from dragonmapper import hanzi
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

from utils import LANGS_SHORT_TO_LONG_CODE


class LanguageTextProcessing:

    def __init__(self, cs_languages, device='cpu'):
        cs_languages = [LANGS_SHORT_TO_LONG_CODE[lang] if len(lang) == 2 else lang for lang in cs_languages]
        if set(cs_languages) == {'eng', 'cmn'}:
            print('CS Languages: eng, cmn')
            self.separate_lang_substrings = self.separate_chinese_and_english_substrings
            self.lid = None
        elif set(cs_languages) == {'eng', 'spa'}:
            print('CS Languages: eng, spa')
            self.separate_lang_substrings = self.separate_spanish_and_english_substrings
            self.tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-spaeng-lid-lince")
            self.model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-spaeng-lid-lince")
            self.lid = pipeline('ner', model=self.model, tokenizer=self.tokenizer, device=device)
        else:
            raise ValueError(
                f'Selected cs languages are not supported; selected: {cs_languages}; available language pairs: (eng, cmn), (eng, spa).')

    def separate_chinese_and_english_substrings(self, text):
        # we assume that we have only either Chinese or English text
        substrings = []
        current_substring = ''
        current_lang = None  # eng or cmn
        text = text.strip('#').strip('~')

        for char in text:
            if char.isspace():
                current_substring += char
                continue
            is_chinese = hanzi.has_chinese(char)
            if is_chinese and current_lang != 'cmn':
                if len(current_substring) > 0:
                    substrings.append((current_lang, current_substring.strip()))
                    current_substring = ''
                current_lang ='cmn'
            elif not is_chinese and current_lang != 'eng':
                if len(current_substring) > 0:
                    substrings.append((current_lang, current_substring.strip()))
                    current_substring = ''
                current_lang = 'eng'
            current_substring += char
        if current_lang is None:
            return []
        else:
            substrings.append((current_lang, current_substring.strip()))
        return substrings

    def separate_spanish_and_english_substrings(self, text):
        substrings = []
        current_substring = ''
        current_lang = None  # eng or spa
        text = text.strip('#').strip('~')
        lid_out = self.lid(text)

        for i in range(len(lid_out)):
            word = lid_out[i]['word']
            cs_lang = lid_out[i]['entity']

            if cs_lang == 'spa' or cs_lang == 'other' or cs_lang == 'ne':
                lang = 'spa'
            elif cs_lang == 'en':
                lang = 'eng'
            else:
                lang = 'spa'

            if i == 0:
                current_lang = lang
                current_substring = word
                continue

            if word.startswith('##') or word.startswith("'") or word == "s":
                lang = current_lang  # wordpieces of one word should all have the same language
            elif not word.isalpha():
                lang = current_lang  # punctuation should belong to previous substring regardless of language

            if lang == current_lang:
                current_substring += " " + word
            else:
                substrings.append((current_lang, current_substring.strip()))
                current_substring = word
                current_lang = lang

        if current_lang is None:
            return []
        else:
            substrings.append((current_lang, current_substring.strip()))

        cleaned_substrings = []
        for lang, substring in substrings:
            substring = substring.replace(' ##', '')
            substring = substring.replace(" d ' ", " d'").replace(" ' s", "'s").replace(" ' ll", "'ll").replace(" ' ve", "'ve").replace(" ' d", "'d")
            substring = substring.replace(" ' re", "'re").replace(" ' m", "").replace('. .', '..')
            cleaned_substrings.append((lang, substring))

        return cleaned_substrings
