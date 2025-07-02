import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data_into_subsets(data, dev_n=20, test_n=20, random_state=42):
    # Get unique speaker IDs
    unique_speakers = data['spk'].unique()

    # Step 1: select speakers with enough datapoints to get test and dev from this; the remaining will go into train
    eligible_speakers = _find_eligible_speakers(data, min_n_utterances=20)
    # print(f'Unique speakers ({len(unique_speakers)}): {unique_speakers}')
    # print(f'Eligible speakers ({len(eligible_speakers)}): {eligible_speakers}')

    dev_size = (dev_n / len(eligible_speakers))
    test_size = (test_n / len(eligible_speakers))

    # Step 2: Split eligible speakers into train, dev, and test
    train_speakers, temp_speakers = train_test_split(eligible_speakers, test_size=test_size + dev_size,
                                                     random_state=random_state)
    dev_speakers, test_speakers = train_test_split(temp_speakers, test_size=test_size / (test_size + dev_size),
                                                   random_state=random_state)

    # Step 3: Add all remaining speakers (i.e. with less than min_n_utterances) to train
    for x in unique_speakers:
        if x not in eligible_speakers:
            train_speakers = np.append(train_speakers, x)

    print(f'Dev speakers: required: {dev_n}, obtained: {len(dev_speakers)}')
    print(f'Test speakers: required: {test_n}, obtained: {len(test_speakers)}')
    print(f'Train speakers: required: {len(unique_speakers) - dev_n - test_n}, obtained: {len(train_speakers)}')

    # Step 4: Assign data to train, dev, and test sets based on speaker IDs
    train_df = data[data['spk'].isin(train_speakers)]
    dev_df = data[data['spk'].isin(dev_speakers)]
    test_df = data[data['spk'].isin(test_speakers)]

    return train_df, dev_df, test_df


def _find_eligible_speakers(df, min_n_utterances=20):
    """
    Counts number of utterances per speaker and filters for those speakers that have many utterances for train:
    we want at least min_n_utterances=20 in each language for train.
    """
    summary = df.groupby(['spk', 'lang'])['utt'].count()
    summary_df = pd.DataFrame(summary)
    summary_df['more_than_n'] = summary_df['utt'] >= min_n_utterances
    filtered_summary_df = summary_df.groupby('spk').sum().query('more_than_n > 2')
    
    filtered_summary_df.reset_index(inplace=True)
    return filtered_summary_df['spk'].unique()
