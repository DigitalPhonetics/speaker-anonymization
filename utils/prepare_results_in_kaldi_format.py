from shutil import copy
import torchaudio
from collections import defaultdict
import os

from utils import save_kaldi_format, create_clean_dir, read_kaldi_format


def get_anon_wav_scps(input_folder):
    audio_dicts = {}

    # List all subfolders in the input folder
    subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    for subfolder in subfolders:
        audio_dict = {}
        subfolder_path = os.path.join(input_folder, subfolder)
        audio_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
        for audio in audio_files:
            name = os.path.splitext(audio)[0]
            pat = os.path.join(subfolder_path, audio)
            audio_dict[name] = pat
        audio_dicts[subfolder] = audio_dict

    return audio_dicts

def prepare_evaluation_data(dataset_dict, anon_wav_scps, anon_vectors_path, output_path, anon_suffix='_anon',
                            emb_level='spk'):
    trials_subs = defaultdict(list)
    for dataset, orig_dataset_path in dataset_dict.items():
        for anon in {True, False}:
            suffix = anon_suffix if anon else ''
            orig_data_path = orig_dataset_path.parent
            out_data_split = output_path / f'{dataset}{suffix}'
            out_data_split.mkdir(exist_ok=True, parents=True)

            copy_files = ['spk2utt', 'text', 'utt2spk']

            if 'trials' in dataset:
                copy_files += ['trials']
                split_dataset_name = dataset.split('_')
                trial_combined_name = f'{split_dataset_name[0]}_{split_dataset_name[1]}_asr{suffix}'
                trials_subs[output_path / trial_combined_name].append(out_data_split)
                if (orig_dataset_path / 'trials_balanced').exists():
                    copy_files += ['trials_balanced']
                if (orig_dataset_path / 'trials_constant').exists():
                    copy_files += ['trials_constant']
            elif 'enrolls' in dataset:
                copy_files += ['enrolls']

            if anon:
                anon_vec_split = anon_vectors_path / f'{dataset}' / emb_level
                if dataset == 'train-clean-360':
                    spk2gender = read_kaldi_format(anon_vec_split / 'spk2gender')
                    if '-' in list(spk2gender.keys())[0]:  # spk2gender contains utts as keys, not speakers
                        utt2spk = read_kaldi_format(orig_dataset_path / 'utt2spk')
                        revised_spk2gender = {utt2spk[utt]: gender for utt, gender in spk2gender.items()}
                        save_kaldi_format(revised_spk2gender, out_data_split / 'spk2gender')
                    else:
                        save_kaldi_format(spk2gender, out_data_split / 'spk2gender')
                else:
                    copy(anon_vec_split / 'spk2gender', out_data_split / 'spk2gender')
                save_kaldi_format(anon_wav_scps[dataset], out_data_split / 'wav.scp')
                save_kaldi_format(get_utterance_durations(anon_wav_scps[dataset]), out_data_split / 'utt2dur')
            else:
                copy_files += ['spk2gender', 'wav.scp', 'utt2dur']

            for file in copy_files:
                copy(orig_dataset_path / file, out_data_split / file)

            if 'train-clean-360' in dataset:
                orig_data_asv = orig_data_path / f'{dataset}-asv'
                out_data_asv = output_path / f'{dataset}-asv{suffix}'
                out_data_asv.mkdir(exist_ok=True, parents=True)

                for file in ['text', 'wav.scp', 'utt2dur']:
                    copy(out_data_split / file, out_data_asv / file)

                for file in ['spk2utt', 'utt2spk']:
                    copy(orig_data_asv / file, out_data_asv / file)

                spk2gender = transform_spk2gender(session_spk2gender=read_kaldi_format(out_data_split / 'spk2gender'),
                                                  session_utt2spk=read_kaldi_format(out_data_split / 'utt2spk'),
                                                  global_utt2spk=read_kaldi_format(orig_data_asv / 'utt2spk'))
                save_kaldi_format(spk2gender, out_data_asv / 'spk2gender')

            if 'vctk' in dataset and '_all' in dataset:
                split_vctk_into_common_and_diverse(dataset=dataset, output_path=output_path,
                                                   orig_data_path=orig_data_path, copy_files=copy_files, anon=anon,
                                                   out_data_split=out_data_split, anon_suffix=suffix)
    for combined_dir, trial_dirs in trials_subs.items():
        combine_asr_data(input_dirs=trial_dirs, output_dir=combined_dir)


def get_utterance_durations(wav_scp):
    utt2dur = {}
    for utt, wav_path in wav_scp.items():
        metadata = torchaudio.info(wav_path)
        duration = metadata.num_frames / metadata.sample_rate
        utt2dur[utt] = duration
    return utt2dur


def transform_spk2gender(session_spk2gender, session_utt2spk, global_utt2spk):
    utt2gender = {utt: session_spk2gender[spk] for utt, spk in session_utt2spk.items()}
    global_spk2gender = {spk: utt2gender[utt] for utt, spk in global_utt2spk.items()}
    return global_spk2gender


def split_vctk_into_common_and_diverse(dataset, output_path, orig_data_path, copy_files, anon, out_data_split,
                                       anon_suffix):
    # for vctk, the 'all' tag combines two splits: one for common and one for diverse sentences
    # we have to copy the original data for these splits to the output directory
    common_split = dataset.replace('all', 'common')  # same sentences for all speakers
    diverse_split = dataset.replace('_all', '')  # different sentences for each speaker
    for split in {common_split, diverse_split}:
        (output_path / f'{split}{anon_suffix}').mkdir(exist_ok=True, parents=True)
        for file in copy_files:
            copy(orig_data_path / split / file, output_path / f'{split}{anon_suffix}' / file)
    if anon:
        spk2gender = read_kaldi_format(out_data_split / 'spk2gender')
        wav_scp = read_kaldi_format(out_data_split / 'wav.scp')
        utt2dur = read_kaldi_format(out_data_split / 'utt2dur')

        for split in {common_split, diverse_split}:
            utt2spk = read_kaldi_format(orig_data_path / split / 'utt2spk')
            speakers = set(utt2spk.values())
            anon_split = f'{split}{anon_suffix}'
            split_spk2gender = {spk: gender for spk, gender in spk2gender.items() if spk in speakers}
            save_kaldi_format(split_spk2gender, output_path / anon_split / 'spk2gender')
            split_wav_scp = {utt: wav_path for utt, wav_path in wav_scp.items() if utt in utt2spk}
            save_kaldi_format(split_wav_scp, output_path / anon_split / 'wav.scp')
            split_utt2dur = {utt: dur for utt, dur in utt2dur.items() if utt in utt2spk}
            save_kaldi_format(split_utt2dur, output_path / anon_split / 'utt2dur')


def combine_asr_data(input_dirs, output_dir):
    create_clean_dir(output_dir)

    files = ['text', 'utt2spk', 'spk2gender', 'wav.scp', 'utt2dur']

    for file in files:
        data = {}
        for in_dir in input_dirs:
            data.update(read_kaldi_format(in_dir / file))
        save_kaldi_format(data, output_dir / file)

    spk2utt = defaultdict(list)
    for in_dir in input_dirs:
        for spk, utt_list in read_kaldi_format(in_dir / 'spk2utt').items():
            spk2utt[spk].extend(utt_list)
    spk2utt = {spk: sorted(utt_list) for spk, utt_list in spk2utt.items()}
    save_kaldi_format(spk2utt, output_dir / 'spk2utt')


