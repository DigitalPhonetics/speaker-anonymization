from collections import defaultdict

from utils import save_kaldi_format


def save_data(enrolls, trials, trial_exp_f, trial_exp_m, dataset_name, out_dir):
    # for each dataset, we create three folders: enrolls, trials_f and trials_m

    # save enrolls data and "enroll" file (list of all enrollment utterances)
    _save_data_as_kaldi_files(enrolls, out_dir / f'{dataset_name}_enrolls')
    _save_enroll_file(enrolls, out_dir / f'{dataset_name}_enrolls')

    # save trials_f data and "trials" files (format: <enroll_spk> <trial_utt> <target/non-target>)
    _save_data_as_kaldi_files(trials[trials['gender'] == 'female'], out_dir / f'{dataset_name}_trials_f')
    _save_trials(trial_exp_f, out_dir / f'{dataset_name}_trials_f', filename='trials')

    # save trials_m data and "trials" files (format: <enroll_spk> <trial_utt> <target/non-target>)
    _save_data_as_kaldi_files(trials[trials['gender'] == 'male'], out_dir / f'{dataset_name}_trials_m')
    _save_trials(trial_exp_m, out_dir / f'{dataset_name}_trials_m', filename='trials')


def save_train_data(trials, dataset_name, out_dir):
    # in the train split, we only have "trials" folders
    _save_data_as_kaldi_files(trials[trials['gender'] == 'female'], out_dir / f'{dataset_name}_trials_f')
    _save_data_as_kaldi_files(trials[trials['gender'] == 'male'], out_dir / f'{dataset_name}_trials_m')


def _save_data_as_kaldi_files(df, data_dir):
    data_dir.mkdir(exist_ok=True, parents=True)
    spk_df = df.set_index('spk')
    utt_df = df.set_index('utt')

    spk2gender = {spk: 'f' if gender == 'female' else 'm' for spk, gender in spk_df['gender'].to_dict().items()}
    text = utt_df['clean_text'].to_dict()
    utt2dur = utt_df['duration'].to_dict()
    utt2spk = utt_df['spk'].to_dict()
    wav_scp = utt_df['segment_path'].to_dict()

    spk2utt = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk2utt[spk].append(utt)

    save_kaldi_format(spk2gender, data_dir / 'spk2gender')
    save_kaldi_format(text, data_dir / 'text')
    save_kaldi_format(utt2dur, data_dir / 'utt2dur')
    save_kaldi_format(utt2spk, data_dir / 'utt2spk')
    save_kaldi_format(wav_scp, data_dir / 'wav.scp')
    save_kaldi_format(spk2utt, data_dir / 'spk2utt')


def _save_enroll_file(enrolls_df, data_path):
    utts = sorted(enrolls_df['utt'].to_list())
    with open(data_path / 'enrolls', 'w') as f:
        for utt in utts:
            f.write(f'{utt}\n')


def _save_trials(trials_df, data_path, filename='trials'):
    trials_df = trials_df[['enroll_spk', 'utt', 'target']].copy()
    trials_df = trials_df.sort_values(by=['enroll_spk', 'utt'])
    filepath = data_path / filename
    trials_df.to_csv(filepath, sep=' ', index=False, header=False)