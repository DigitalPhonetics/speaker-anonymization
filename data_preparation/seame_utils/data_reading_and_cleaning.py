import pandas as pd
from pathlib import Path
import re
import warnings


AUDIO_NAME_PATTERN = re.compile(
        r'(?P<conversation_group_id>\d{,2})?'  # Conversation group number id (1-46)
        r'(?P<survey_code>NC|NI|UI)'  # Recording location: N (Nanyang Technological University) / U (Universities Sains Malaysia); Speaking style: C (Conversation) / I (Interview)
        r'(?P<speaker_raw_id>\d{,2}|X)'  # Speaker id number (NC: 0 - 61; NI: 01 - 67, UI: 01 - 29)
        r'(?P<gender>F|M)'  # Gender: F: Female, M: Male
        r'(?P<nationality>A|B)'  # Nationality of speakers: A: Malaysian, B: Singaporean
        r'(?P<microphone_channel>P|Q|X|Y|Z)_'  # Microphone channel: (N: P, Q, X,Y. U: Z)
        r'(?P<session>\d{2})'  # The sequence of recording of that speaker
        r'(?P<part_of_session>\d{2})'  # The order of actual file in the recording.
    )

SURVEY_CODES = {'UI': 'Interview-Malaysia', 'NC': 'Conversation', 'NI': 'Interview-Singapore'}
SURVEY2TAG = {'Interview-Malaysia':  'IM', 'Conversation': 'C', 'Interview-Singapore': 'IS'}


def prepare_transcript_and_meta_data(dataset_path):
    data = _load_seame_data(dataset_path)

    # Keep interviews and disregard conversations, only keep utterances of at least 500ms duration
    # Remove all utterances that consist of only hesitation or only discourse particle
    data = _filter_data(data)

    return data


def _load_seame_data(dataset_path):
    # speaker_df = _load_speaker_data(dataset_path)
    transcripts_df = _load_transcripts(dataset_path)

    new_keys = ['conversation_group_id', 'survey_id', 'spk', 'gender', 'nationality', 'microphone_channel',
                'session', 'part_of_session']
    transcripts_df[new_keys] = transcripts_df.apply(_parse_audio_file_name, axis=1, result_type='expand')
    transcripts_df['utt'] = transcripts_df.apply(lambda x: f'{x["audio_id"]}_{x["start"]}_{x["end"]}', axis=1)
    transcripts_df['audio_path'] = transcripts_df['transcript_path'].apply(lambda x: str(Path(x).parent.parent.parent / 'audio' / f'{Path(x).stem}.flac'))
    return transcripts_df


def _create_spk_id(spk_idx, survey_name):
    survey_tag = SURVEY2TAG[survey_name]
    return f'{survey_tag}{int(spk_idx)}'


def _load_transcripts(dataset_path):
    dfs = []

    for survey_type in ['conversation', 'interview']:
        transcription_dir = dataset_path / 'data' / survey_type / 'transcript' / 'phaseII'
        for transcript_file in transcription_dir.glob('*.txt'):
            df = pd.read_csv(transcript_file, delimiter='\t', names=['audio_id', 'start', 'end', 'lang', 'raw_text'])
            df['lang'] = df['lang'].str.lower()
            df['survey_type'] = survey_type
            df['transcript_path'] = str(transcript_file)
            df['duration'] = (df['end'] - df['start']) / 1000
            dfs.append(df)

    detailed_df = pd.concat(dfs, ignore_index=True)
    detailed_df.reset_index(drop=True, inplace=True)
    return detailed_df


def _parse_audio_file_name(column):
    filename = column['transcript_path']
    parses = re.findall(AUDIO_NAME_PATTERN, Path(filename).stem)
    if not parses:
        warnings.warn(f'Could not parse name: {filename}')
        return [None] * 8
    if len(parses) > 1:
        warnings.warn(f'Ambiguous parsing for name: {filename}')

    parse = parses[0]
    conversation_group_id, survey_id, speaker_id, gender, nationality, microphone_channel, session, part_of_session = parse
    survey_id = SURVEY_CODES[survey_id]
    speaker_id = _create_spk_id(speaker_id, survey_id)
    gender = 'female' if gender == 'F' else 'male'
    nationality = 'Malaysian' if nationality == 'A' else 'Singaporean'
    return conversation_group_id, survey_id, speaker_id, gender, nationality, microphone_channel, session, part_of_session


def _filter_data(df):
    """
    Filters out following utterances:
        - all utterances belonging to conversations (keeping only interviews=
        - all utterances shorter than 500 ms (keeping only longer utterances)
        - all utterances that only contain discourse particles and/or hesitations (keeping only utterances with actual content)
    """
    hesitation_only_pattern = r'^\(.*?\)$'
    discourse_particle_only_pattern = r'^\[.*?\]$'

    conditions = [
        'survey_type == "interview"',
        f'~(raw_text.str.contains(r"{hesitation_only_pattern}") or raw_text.str.contains(r"{discourse_particle_only_pattern}"))',
        'duration > 0.5',
    ]
    filter = '&'.join(conditions)
    filtered_df = df.query(filter).copy()

    # remove everything that is only hesitation or only discourse particle
    remove_bracketed_substrings = lambda x: re.sub(r'\s+', ' ', re.sub(r'\(.*?\)', '',
                                                                       re.sub(r'\[.*?\]', '', x))).strip()
    filtered_df['clean_text'] = filtered_df['raw_text'].apply(remove_bracketed_substrings)
    filtered_df = filtered_df.query('~(clean_text == "")').copy()

    return filtered_df
