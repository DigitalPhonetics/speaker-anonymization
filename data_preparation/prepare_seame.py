from pathlib import Path
import pandas as pd

from .seame_utils.data_reading_and_cleaning import prepare_transcript_and_meta_data
from .seame_utils.audio_splitting import split_audio_files
from .seame_utils.data_splitting import split_data_into_subsets
from .seame_utils.data_characteristics import extract_data_characteristics
from .shared_utils.enroll_trial_splitting import split_into_enroll_and_trial, create_trial_experiment_files
from .shared_utils.concatenate_audios import concatenate_audios
from .shared_utils.data_saving import save_data, save_train_data

def prepare_seame(dataset_path, output_path):
    # dataset_path : top most Seame folder
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # get transcript and metadata, clean transcript
    data = prepare_transcript_and_meta_data(dataset_path)

    # split audios into single per-sentence audios
    segment_audio_dir = output_path / 'wavs'
    segment_audio_dir.mkdir(exist_ok=True, parents=True)
    wav_scp = split_audio_files(data, segment_audio_dir)
    data['segment_path'] = data['utt'].apply(lambda x: wav_scp.get(x, None))
    # data['segment_path'] = data['utt'].apply(lambda x: str(segment_audio_dir / f'{x}.wav'))

    # split data into train, dev and test
    train, dev, test = split_data_into_subsets(data)
    splits = {
        'train': train,
        'dev': dev,
        'test': test
    }

    kaldi_out_dir = output_path / 'kaldi'
    # the concatenated audio dir will only store the audios that are concatenated, the rest remains in segment audio dir
    concatenated_audio_dir = output_path / 'concatenated_wavs'
    concatenated_audio_dir.mkdir(exist_ok=True, parents=True)

    selected_data = []

    for split, split_data in splits.items():
        for lang in ['en', 'zh', 'cs']:
            sub_data = split_data[split_data['lang'] == lang]
            if split == 'train':  # we don't have enrollment data in our training split, we call everything trial
                trials = concatenate_audios(sub_data, concatenated_audio_dir)
                save_train_data(trials, f'seame-{lang}_{split}', kaldi_out_dir)
                selected_data.append(trials)
            else:
                # split into enrollment and trials
                enrolls, trials = split_into_enroll_and_trial(sub_data, enroll_utts_per_spk=10)
                # concatenate short audios to have a min length of 2 seconds
                enrolls = concatenate_audios(enrolls, concatenated_audio_dir)
                trials = concatenate_audios(trials, concatenated_audio_dir)
                # create the ASV trial experiment files
                # if balanced=True, the trial utterances per speaker are reduced to have the same number of utterances for all speakers
                # alternatively, you can set n=<int> to select a specific number of utterances per speaker
                trial_exp_f, trial_exp_m = create_trial_experiment_files(trials, balanced=False, n=None)
                # save the data in kaldi format
                save_data(enrolls, trials, trial_exp_f, trial_exp_m, f'seame-{lang}_{split}', kaldi_out_dir)
                selected_data.append(enrolls)
                selected_data.append(trials)

    selected_data = pd.concat(selected_data)
    data_characteristics = extract_data_characteristics(selected_data)
    data_characteristics.to_csv(output_path / 'data_characteristics.csv', index=False, sep='\t')