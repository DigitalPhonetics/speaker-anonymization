def split_into_enroll_and_trial(data, enroll_utts_per_spk=10):
    enrolls = data.groupby('spk').apply(lambda x: x.sample(enroll_utts_per_spk), include_groups=False).reset_index(
        level='spk').reset_index(drop=True)
    trials = data[~data['utt'].isin(enrolls['utt'])]
    return enrolls, trials


def create_trial_experiment_files(trials, balanced=False, n=None):
    trials_f = _create_trials(trials, gender='female')
    trials_m = _create_trials(trials, gender='male')

    if n is not None:
        trials_f = _reduce_trials(trials_f, n=n)
        trials_m = _reduce_trials(trials_m, n=n)
    elif balanced is True:
        trials_f = _reduce_trials(trials_f, n=_get_lowest_utt_count(trials_f))
        trials_m = _reduce_trials(trials_m, n=_get_lowest_utt_count(trials_m))

    return trials_f, trials_m


def _create_trials(trials_df, gender):
    trials_df = trials_df[trials_df['gender'] == gender].copy()
    enroll_spks = list(set(trials_df['spk'].to_list()))
    trials_df['enroll_spk'] = [enroll_spks] * len(trials_df)
    trials = trials_df.explode('enroll_spk')
    trials['target'] = trials.apply(lambda x: 'target' if x['spk'] == x['enroll_spk'] else 'nontarget', axis=1)
    trials = trials[['enroll_spk', 'utt', 'target', 'spk']].copy()
    return trials


def _reduce_trials(trials_df, n=10):
    selected = trials_df[trials_df['target'] == 'target'].groupby('spk').apply(lambda x: x.sample(n), include_groups=False)
    trials_df = trials_df[trials_df['utt'].isin(selected['utt'])]
    return trials_df


def _get_lowest_utt_count(trials_df):
    return  trials_df[trials_df['target'] == 'target'].groupby('spk').count().min()

