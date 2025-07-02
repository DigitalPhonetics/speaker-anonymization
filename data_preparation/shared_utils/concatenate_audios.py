import soundfile
import numpy as np
import pandas as pd

from .loudness import compute_loudness


def concatenate_audios(data, output_dir, loudness_limit=5.0, duration_limit=2.0):
    # loudness_limit: only concatenate audios that have a max. loudness difference of this limit
    # duration_limit: concatenate audios shorter than this limit until this limit is reached

    # not all information is needed anymore, we reduce the dataframe to contain only what we still need to consider
    data = data[['utt', 'spk', 'gender', 'raw_text', 'clean_text', 'duration', 'segment_path']].copy()

    data_not_concat = data[data['duration'] >= duration_limit]
    data_short_utts = data[data['duration'] < duration_limit].copy()

    data_short_utts['loudness'] = data_short_utts['segment_path'].apply(compute_loudness)
    concatenated_data_per_spk = []
    for spk, group in data_short_utts.groupby('spk'):
        concatenated_data_per_spk.append(_concatenate_spk_data(group, output_dir, loudness_limit, duration_limit))
    concatenated_data = pd.concat(concatenated_data_per_spk)
    data = pd.concat([data_not_concat, concatenated_data])
    return data


def _concatenate_spk_data(data, output_dir, loudness_limit, duration_limit):
    spk = data['spk'].to_list()[0]
    gender = data['gender'].to_list()[0]

    utts2delete = set(data[data['loudness'].isna()]['utt'].to_list())
    data = data[~data['loudness'].isna()]
    data = data.set_index('utt')
    concatenation_pairs, more_utt2delete = _find_concatenation_pairs(data, loudness_limit, duration_limit)
    utts2delete.update(more_utt2delete)

    concat_data = {'utt': [], 'spk': [], 'gender': [], 'raw_text': [], 'clean_text': [], 'duration': [],
                   'segment_path': []}

    for audio_pair in concatenation_pairs:
        concatenated_name = '-'.join(audio_pair)
        audio_paths = [data.loc[utt, 'segment_path'] for utt in audio_pair]
        concat_audio_path = _concatenate_audio_files(audio_paths, concatenated_name, output_dir)
        concat_data['utt'].append(concatenated_name)
        concat_data['spk'].append(spk)
        concat_data['gender'].append(gender)
        concat_data['raw_text'].append(' '.join([data.loc[utt]['raw_text'] for utt in audio_pair]))
        concat_data['clean_text'].append(' '.join([data.loc[utt]['clean_text'] for utt in audio_pair]))
        concat_data['duration'].append(sum([data.loc[utt]['duration'] for utt in audio_pair]))
        concat_data['segment_path'].append(concat_audio_path)

    return pd.DataFrame(concat_data)


def _find_concatenation_pairs(data, loudness_limit, duration_limit):
    utts2delete = set()
    concatenation_pairs = []

    utt2loudness = data['loudness'].to_dict()
    utt2dur = data['duration'].to_dict()
    loudness_ranking = sorted([(utt, loudness) for utt, loudness in utt2loudness.items()], key=lambda x: x[1])

    current_utts = set()
    combined_dur = 0
    last_loudness = None

    while len(loudness_ranking) > 0:
        utt, loud = loudness_ranking.pop()

        if last_loudness is None:
            last_loudness = loud
            current_utts.add(utt)
            combined_dur = utt2dur[utt]

        elif abs(last_loudness - loud) > loudness_limit:
            utts2delete.update(current_utts)
            current_utts = {utt}
            last_loudness = loud
            combined_dur = utt2dur[utt]

        else:
            current_utts.add(utt)
            combined_dur += utt2dur[utt]

            if combined_dur >= duration_limit:
                concatenation_pairs.append(tuple(current_utts))
                current_utts = set()
                combined_dur = 0
                last_loudness = None
            else:
                last_loudness = loud
    utts2delete.update(current_utts)
    return concatenation_pairs, utts2delete


def _concatenate_audio_files(audio_paths, concatenated_name, output_dir):
    audios = []
    srs = []
    for audio_path in audio_paths:
        audio, sr = soundfile.read(audio_path)
        audios.append(audio)
        srs.append(sr)
    assert len(set(srs)) == 1; f'Audios have different sampling rates:{srs}!'
    audio = np.concatenate(audios, axis=0)
    out_path = output_dir / f'{concatenated_name}.wav'
    soundfile.write(file=out_path, data=audio, samplerate=srs[0])
    return out_path
