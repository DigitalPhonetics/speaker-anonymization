import re
import pandas as pd

def prepare_transcript_and_meta_data(dataset_path):
    data = []
    speakers = []

    audio_dir = dataset_path / 'audio'
    for lang_dir in (dataset_path / 'transcripts').glob('*'):  # English and Spanish
        for transcript_file in lang_dir.glob('*.cha'):
            if 'maria' in transcript_file.name:  # exclude the maria files (only contain Marías speech)
                continue
            if not (audio_dir / lang_dir.name / '0wav' / f'{transcript_file.stem}.wav').exists():
                continue

            df, spks, languages = _read_cha_file(transcript_file)
            df = _remove_overlapping_rows(df)
            df['clean_text'] = df['raw_text'].apply(_clean_text)

            # annotate language setting
            df[['lang', 'clean_text']] = df.apply(lambda x: _find_languages_in_utt(x['clean_text'], languages), axis=1,
                                                  result_type='expand')
            df['codeswitching'] = df['lang'].apply(len) > 1

            data.append(df)
            speakers.append(spks)

    data = pd.concat(data)
    speakers = pd.concat(speakers)

    # delete empty clean text rows and clean text rows with only one word
    data = data[data['clean_text'].str.split(' ').str.len() > 1]
    lang_values = list(set(data['lang'].explode().to_list()))
    data[lang_values] = data.apply(lambda x: [l in x['lang'] for l in lang_values], axis=1, result_type='expand')

    return data, speakers


def _read_cha_file(filename):
    languages = []
    speakers = {'name': [], 'langs': [], 'age': [], 'corpus': [], 'gender': [], 'type': []}
    audio_name = None
    data = {'utt': [], 'spk': [], 'audio': [], 'start': [], 'end': [], 'duration': [], 'transcript': [], 'raw_text': []}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('@Languages'):
                languages = [lang.strip() for lang in line.split('\t')[1].split(',')]
            elif line.startswith('@ID'):
                spk_info = [info.strip() for info in line.split('\t')[1].split('|')]
                speakers['name'].append(spk_info[2])
                speakers['langs'].append(spk_info[0])
                speakers['age'].append(spk_info[3].strip(';'))
                speakers['corpus'].append(spk_info[1])
                speakers['gender'].append(spk_info[4])
                speakers['type'].append(spk_info[7])
            elif line.startswith('@Media'):
                audio_name = line.split('\t')[1].split(',')[0].strip()
            elif line.startswith('*'):
                content = line.split('\t')[1]
                text = ' '.join(content.split(' ')[:-1])
                time = ''.join([c for c in content.split(' ')[-1] if c.isprintable()])
                start, end = time.split('_') if '_' in time else [None, None]
                spk = line.split('\t')[0].strip('*').strip(':')
                data['utt'].append(f'{spk}_{audio_name}_{start}-{end}')
                data['spk'].append(spk)
                data['audio'].append(audio_name)
                data['start'].append(start)
                data['end'].append(end)
                data['duration'].append((int(end) - int(start)) / 1000 if start is not None else None)
                data['transcript'].append(str(filename))
                data['raw_text'].append(text)
    return pd.DataFrame(data), pd.DataFrame(speakers), languages


def _remove_overlapping_rows(data):
    df = data.copy()
    df['overlaps_with_previous'] = df['raw_text'].str.startswith('+<')
    df['overlaps_with_next'] = (df['overlaps_with_previous'].shift(-1) == True)
    df = df[~df['overlaps_with_previous'] & ~df['overlaps_with_next']]
    df = df.drop(columns=['overlaps_with_previous', 'overlaps_with_next'])
    return df


def _clean_text(text):
    text = text.replace('www', '')  # denotes something that is not transcribed (e.g., because no consent)
    text = text.replace('xxx', '')  # something unintelligible -> corresponds to <unk>
    text = text.replace('(.)', '').replace('(..)', '').replace('(...)', '')  # pauses
    text = text.replace('[/]', '').replace('[//]', '')  # word or phrase repetition
    text = text.replace('[///]', '')  # retracing
    text = text.replace('[/-]', '')  # false start without retracing
    text = text.replace('+"/', '').replace('+"', '')  # quotes
    text = re.sub(r'&=[\w-]+:?[\w-]*', '', text)  # &= non word sounds and  &=0 omitted words
    text = re.sub(r'&-\w+', '', text)  # fillers
    text = re.sub(r'\(\w+\)', '', text)  # noncomplete words
    text = re.sub(r'_', ' ', text)  # compounds and multi-word tokens like Great_Britain
    text = text.replace('+..', '')  # trailing off, incomplete utterance
    text = text.replace('+!?', '?')  # question with exclamation
    text = text.replace('+//', '')  # self-interruption
    text = text.replace('+/', '')  # interruption by other interlocutor
    text = text.replace('+.', '')  # transcription break
    text = text.replace('+,', '').replace('++', '')  # completions of earlier utterances
    text = re.sub(r'\[[^-]+]', '', text)  # paralinguistic stuff
    text = text.replace('@l', '')  # @l: single letter, as in "this says x@l"
    text = text.replace('@n', '')  # @n: neologism
    text = text.replace('@o', '')  # @o: onomatopoeia
    text = text.replace('@q', '')  # @q: metalinguistic use
    text = text.replace('@si', '')  # @si: singing
    text = text.replace('+^', '')  # quick uptake after previous utterance
    text = text.replace('[+', '')
    insert_spaces = lambda x: ' '.join([a for a in x.group(1)])
    text = re.sub(r'(\w+)@k', insert_spaces, text)  # @k: multiple letters, are said separately

    text = re.sub(r'&~(\w+)', _handle_nonwords, text)  # &~: nonwords
    text = text.replace(':', '')  # stuff like "m:hm"
    text = text.replace('<', '').replace('>', '')
    text = text.rstrip('.').rstrip('!').rstrip('?')  # remove punctuation at end of utterance
    text = re.sub(r'\s+', r' ', text).strip()  # remove subsequent or trailing whitespaces

    return text


def _handle_nonwords(matchobj):
    valid_characters = 'abcdefghijklmnopqrstuvwxyzñüáéíóú'  # characters in English and Spanish
    match = matchobj.group(1)
    if all([c in valid_characters for c in match]) and len(
            match) > 2:  # keep words that have at least 3 characters and are not IPA
        return match
    else:
        return ''


def _find_languages_in_utt(text, possible_langs):
    pattern = re.compile('|'.join([f'\[-\s*({lang})\s*\]' for lang in possible_langs]))
    lang_matches = re.search(pattern, text)
    lang_match = [lang for lang in lang_matches.groups() if lang is not None] if lang_matches is not None else []
    cs_matches = set(re.findall(r'@s:?\w*\+?\w*', text))
    if lang_match and cs_matches:
        match_lang = lang_match[0]  # we assume that there is only one match per utterance
        other_languages = [lang for lang in possible_langs if lang != match_lang]
        cs_languages = [cs_match.split(':')[1] if ':' in cs_match else other_languages[0] for cs_match in cs_matches]
        languages = list(set([match_lang] + cs_languages))
    elif lang_match:
        languages = [lang_match[0]]
    elif cs_matches:
        other_languages = possible_langs[1:] # first one is default lang
        cs_languages = [cs_match.split(':')[1] if ':' in cs_match else other_languages[0] for cs_match in cs_matches]
        if len(cs_matches) == len(text.split()): # all words have the @s tag:
            languages = list(set(cs_languages))
        else:
            languages = list(set([possible_langs[0]] + cs_languages))
    else:
        languages = [possible_langs[0]] # default language

    text = re.sub(r'\[.+]', '', text)
    text = re.sub(r'@s:?\w*\+?\w*', '', text)
    text = re.sub(r'\s+', r' ', text).strip()

    return languages, text