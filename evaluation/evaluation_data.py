from shutil import copy, copytree, ignore_patterns
import torchaudio
from datetime import datetime
import json

from utils import save_kaldi_format, create_clean_dir


def prepare_evaluation_data(dataset_list, anon_wav_scps, orig_data_path, anon_vectors_path, output_path):
    for dataset in dataset_list:
        for anon in {True, False}:
            anon_suffix = '_anon' if anon else ''
            orig_data_split = orig_data_path / dataset
            out_data_split = output_path / f'{dataset}{anon_suffix}'
            out_data_split.mkdir(exist_ok=True, parents=True)

            copy_files = ['spk2utt', 'text', 'utt2spk']
            copy_files += ['trials'] if 'trials' in dataset else ['enrolls']

            if anon:
                anon_vec_split = anon_vectors_path / f'{dataset}'
                copy(anon_vec_split / 'spk2gender', out_data_split / 'spk2gender')
                save_kaldi_format(anon_wav_scps[dataset], out_data_split / 'wav.scp')
                save_kaldi_format(get_utterance_durations(anon_wav_scps[dataset]), out_data_split / 'utt2dur')
            else:
                copy_files += ['spk2gender', 'wav.scp', 'utt2dur']

            for file in copy_files:
                copy(orig_data_split / file, out_data_split / file)

        if '_all' in dataset:
            # for vctk, the 'all' tag combines two splits: one for common and one for diverse sentences
            # we have to copy the original data for these splits to the output directory
            common_split = dataset.replace('all', 'common')  # same sentences for all speakers
            diverse_split = dataset.replace('_all', '')  # different sentences for each speaker
            for split in {common_split, diverse_split}:
                (output_path / split).mkdir(exist_ok=True, parents=True)
                for file in ['spk2utt', 'text', 'utt2dur', 'utt2spk', 'spk2gender', 'wav.scp', 'trials']:
                    copy(orig_data_path / split / file, output_path / split / file)


def get_utterance_durations(wav_scp):
    utt2dur = {}
    for utt, wav_path in wav_scp.items():
        metadata = torchaudio.info(wav_path)
        duration = metadata.num_frames / metadata.sample_rate
        utt2dur[utt] = duration
    return utt2dur


def copy_evaluation_results(results_dir, eval_dir, settings, copy_all=False):
    results_dir = results_dir / datetime.strftime(datetime.today(), '%d-%m-%y_%H:%M')
    create_clean_dir(results_dir)

    exp_results_dir = max(list(eval_dir.parent.glob('exp/results-*')))  # exp directory that was created latest
    settings['exp_path'] = str(exp_results_dir)

    with open(results_dir / 'settings.json', 'w') as f:
        json.dump(settings, f)

    if copy_all:  # copy all files and directories from the evaluation but the 'exp_files'
        for test_dir in exp_results_dir.glob('*'):
            if test_dir.is_dir():
                copytree(test_dir, results_dir / test_dir.name, ignore=ignore_patterns('exp_files'))
            else:
                copy(test_dir, results_dir / test_dir.name)
    else:  # copy only the summary results.txt
        copy(exp_results_dir / 'results.txt', results_dir / 'results.txt')

