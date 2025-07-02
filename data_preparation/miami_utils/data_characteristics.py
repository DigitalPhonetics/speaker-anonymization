import re

from ..shared_utils.loudness import compute_loudness

def extract_data_characteristics(data):
    data['loudness'] = data['segment_path'].apply(compute_loudness)
    loudness_mean = data['loudness'].mean()
    loudness_std = data['loudness'].std()

    data['has_repetitions'] = data['raw_text'].apply(_has_repetitions)
    data['has_hesitations_discourse'] = data['raw_text'].apply(_has_hesitations_and_discourse_particles)
    data['has_incomplete_utt'] = data['raw_text'].apply(_is_incomplete_utterance)
    data['has_foreign_tokens'] = data['raw_text'].apply(_has_foreign_language_tokens)
    data['has_abbreviation'] = data['raw_text'].apply(_has_abbreviations)
    data['has_unknown'] = data['raw_text'].apply(_has_unknown)
    data['is_very_short'] = data['clean_text'].apply(_is_very_short_audio)
    data['is_very_long'] = data['duration'].apply(_is_very_long_audio)
    data['suspicious_dur_word_ratio'] = data.apply(lambda x: _has_suspicious_word_count_to_duration_ratio(x['clean_text'], x['duration']), axis=1)
    data[['low_loudness', 'high_loudness']] = data.apply(lambda x: _find_audios_with_suspicious_loudness(x['loudness'], loudness_mean, loudness_std),
                                                         axis=1, result_type='expand')
    data['has_no_problems'] = data.apply(_check_for_problems, axis=1)
    return data

def _has_repetitions(text):
    # Extracted from unfiltered transcript, annotated as [/] and [//]
    pattern = r'(?:\b(\w+)\b(?:\s+\1\b)+|\[/{1,2}\])'
    return bool(re.search(pattern, text.lower()))


def _has_hesitations_and_discourse_particles(text):
    # Extracted from unfiltered transcript
    hesitation_pattern = r'(?:@fp|&-)'  # filled pause as @fp or &-
    pause_pattern = r'\((?:\.{1,3}|…|(?:\d{1,2}:)?\d{1,2}\.\d{1,2})\)'  # pause with length mark like (.), (..), (...), (2.4)
    discourse_token_pattern = r'@i'  # interjection as @i
    satellite_pattern = r'[‡„]'  # initial and final satellites
    return (bool(re.search(hesitation_pattern, text.lower())) or
            bool(re.search(discourse_token_pattern, text.lower())) or
            bool(re.search(pause_pattern, text.lower())) or
            bool(re.search(satellite_pattern, text.lower())))


def _is_incomplete_utterance(text):
    # Incomplete utterances, trailing off
    # Extracted from unfiltered transcript
    incomplete_patterns = [
        r'\+…', r'\+\.\.\.',  # Trailing off
        r'\+\.\.\?', # Trailing off question
        r'\+\+',    # invited interrupt
        r'\+/\.',   # uninvited interruption
        r'\+/\?',   # interruption of question
        r'\+//\.',  # self-interruption
        r'\+//\?',  # self-interruption question
        r'&=0\w+',  # Omitted words
        r'\(\w+\)'  # Partial word omissions
    ]
    incomplete_regex = re.compile('|'.join(incomplete_patterns))
    return bool(incomplete_regex.search(text.lower()))


def _has_foreign_language_tokens(text):
    # Words in foreign languages other than the languages of the dataset
    # Extracted from unfiltered transcript, annotated as @s: (we exclude English and Spanish)
    pattern = r'@s:(?!eng\b|spa\b)\w+'
    return bool(re.search(pattern, text.lower()))


def _has_abbreviations(text):
    # Extracted from unfiltered transcript
    pattern = r'(?:\w{1}\.|[A-Z]{2,})'
    return bool(re.search(pattern, text.lower()))


def _has_unknown(text):
    # Words are marked as unknown if the annotator could not understand them
    # Extracted from unfiltered text, annotated as xxx
    pattern = r'xxx'
    return bool(re.search(pattern, text.lower()))


def _is_very_short_audio(text):
    # definition of very short: only one word / character in filtered transcript
    return len(text.split()) == 1


def _is_very_long_audio(duration):
    # definition of very long: more than 20 seconds
    return duration > 20.0


def _has_suspicious_word_count_to_duration_ratio(text, duration):
    # A word count to duration ratio is suspicious if the duration of the audio is much longer than number of words would suggest
    # We set a generous threshold of 2 seconds per word if the audio has less than 10 seconds, and 1.5 seconds otherwise
    n_words = len(text.split())
    if n_words > 1:
        dur_per_word = round(duration / n_words, 2)
        if duration < 10.0:
            return dur_per_word > 2.0
        else:
            return dur_per_word > 1.5
    else:
        return False


def _find_audios_with_suspicious_loudness(loudness, loudness_mean, loudness_std):
    suspicious_loudness = (loudness - loudness_mean) // (loudness_std + 1)
    if suspicious_loudness < -1:  # low loudness
        return True, False
    elif suspicious_loudness > 1: # high loudness
        return False, True
    else:
        return False, False


def _check_for_problems(row):
    columns = ['has_repetitions', 'has_hesitations_discourse', 'has_incomplete_utt', 'has_foreign_tokens', 
               'has_abbreviation', 'has_unknown', 'is_very_short', 'is_very_long', 'suspicious_dur_word_ratio', 
               'low_loudness', 'high_loudness']
    for column in columns:
        if row[column] is True:
            return False
    return True