from pathlib import Path
import pandas as pd

from .miami_utils.data_reading_and_cleaning import prepare_transcript_and_meta_data
from .miami_utils.audio_splitting import split_audio_files
from .miami_utils.speaker_selection import select_speakers
from .miami_utils.data_characteristics import extract_data_characteristics
from .shared_utils.enroll_trial_splitting import split_into_enroll_and_trial, create_trial_experiment_files
from .shared_utils.concatenate_audios import concatenate_audios
from .shared_utils.data_saving import save_data

def prepare_miami(dataset_path, output_path):
    # dataset_path : top most Bangor Miami folder
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # get transcript and metadata
    # clean the data: remove overlapping utterances, clean transcript, get language settings
    data, speakers = prepare_transcript_and_meta_data(dataset_path)

    # split audios into single per-sentence audios
    segment_audio_dir = output_path / 'wavs'
    segment_audio_dir.mkdir(exist_ok=True, parents=True)
    wav_scp = split_audio_files(data, segment_audio_dir)
    data['segment_path'] = data['utt'].apply(lambda x: wav_scp.get(x, None))
    #data['segment_path'] = data['utt'].apply(lambda x: str(segment_audio_dir / f'{x}.wav'))

    # select speakers with at least 20 utterances in each language setting (eng, spa, cs)
    # segment data to only contain utterances of those speakers
    data, speakers = select_speakers(data, speakers)

    # separate per language setting
    data_per_lang = {
        'en': data[(data['eng']) & (~data['spa'])],
        'es': data[(data['spa']) & (~data['eng'])],
        'cs': data[data['codeswitching']]
    }

    kaldi_out_dir = output_path / 'kaldi'
    # the concatenated audio dir will only store the audios that are concatenated, the rest remains in segment audio dir
    concatenated_audio_dir = output_path / 'concatenated_wavs'
    concatenated_audio_dir.mkdir(exist_ok=True, parents=True)

    selected_data = []

    for lang, sub_data in data_per_lang.items():
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
        save_data(enrolls, trials, trial_exp_f, trial_exp_m, f'miami-{lang}_test', kaldi_out_dir)
        selected_data.append(enrolls)
        selected_data.append(trials)

    selected_data = pd.concat(selected_data)
    data_characteristics = extract_data_characteristics(selected_data)
    data_characteristics.to_csv(output_path / 'data_characteristics.csv', index=False, sep='\t')
