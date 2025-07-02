from speechbrain.utils.metric_stats import ErrorRateStats
from pypinyin import pinyin
import chinese_converter

from anonymization.modules.tts.IMSToucan.Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from utils import LanguageTextProcessing, LANGS_SHORT_TO_LONG_CODE


class ASRMetrics:

    def __init__(self, languages, device, convert_to_pinyin=False, convert_to_phones=False):
        self.languages = [LANGS_SHORT_TO_LONG_CODE[lang] if len(lang) == 2 else lang for lang in languages]
        print(f'Compute ASR metrics based on languages: {self.languages}')
        self.convert_to_pinyin = convert_to_pinyin
        self.convert_to_phones = convert_to_phones

        self.langtextprocess = LanguageTextProcessing(self.languages, device=device)

        if self.convert_to_phones:
            self.phonemizer = {}
            for lang in self.languages:
                self.phonemizer[lang] = ArticulatoryCombinedTextFrontend(language=lang, use_word_boundaries=False,
                                                                         add_silence_to_end=False)
        else:
            self.phonemizer = None


    def compute_wer(self, ref_texts, hyp_texts, out_file):
        wer_stats = ErrorRateStats()

        ids = []
        predicted = []
        targets = []
        for utt_id, ref in ref_texts.items():
            if utt_id not in hyp_texts:  # skip the problematic samples that we skipped during inference (they were too long)
                continue
            ref_tokens = self._plain_text_key(ref)
            pred_tokens = self._plain_text_key(hyp_texts[utt_id])
            if len(ref_tokens) > 0 and len(pred_tokens) > 0:
                ids.append(utt_id)
                targets.append(ref_tokens)
                predicted.append(pred_tokens)
            else:
                print(f'Empty prediction for {utt_id}: {hyp_texts[utt_id]}')

        wer_stats.append(ids=ids, predict=predicted, target=targets)
        out_file.parent.mkdir(exist_ok=True, parents=True)

        with open(out_file, 'w') as f:
            wer_stats.write_stats(f)

        return wer_stats

    def _plain_text_key(self, utterance):
        utterance = ' '.join([t for t in utterance.split(' ') if not '<unk>' in t])
        if self.convert_to_pinyin:
            converted_utterance = " ".join([a[0] for a in pinyin(utterance)])
            token_list = [character for character in converted_utterance if character.isalnum() or character.isspace()]
            tokens = ''.join(token_list).lower().strip().split()
        else:
            # an utterance might consist only of English text, Chinese characters, or a mix
            # depending on what it is, we need to change our strategy: English text is written with spaces, Chinese text is not necessarily, mixed text might be
            # Whisper does not return spaces in Chinese transcriptions, so we will have spaces only if English text is involved
            # so, first, let's split the utterance into Chinese and English substrings (format: [(lang, substring)]
            if 'zh' in self.languages:
                simplified_utterance = chinese_converter.to_simplified(utterance)  # if it is English only, it has no effect
            else:
                simplified_utterance = utterance

            lang_substrings = self.langtextprocess.separate_lang_substrings(simplified_utterance)
            if self.convert_to_phones:
                tokens = self._phonemize_tokens(lang_substrings)
            else:
                tokens = []
                for lang, substring in lang_substrings:
                    lang = LANGS_SHORT_TO_LONG_CODE.get(lang, lang)
                    if lang == 'zh':
                        characters = [t for t in substring if t.isalnum()]
                        if characters:
                            tokens.extend(characters)
                    else:
                        # we first exclude all special characters, i.e., let's becomes lets, what? becomes what etc.
                        substring = ''.join([c.lower() for c in substring if c.isalnum() or c.isspace()])
                        words = substring.split(' ')
                        # delete empty strings
                        words = [w for w in words if len(w) > 0]
                        if words:
                            tokens.extend(words)
        return tokens

    def _phonemize_tokens(self, lang_substrings):
        tokens = []
        for lang, substring in lang_substrings:
            phones = self.phonemizer[lang].get_phone_string(substring, include_eos_symbol=False)
            phones = phones.replace('~', '')
            tokens.extend([p for p in phones if p.isalnum()])
        return tokens
