import torch
import resampy
import numpy as np

from .AnonFastSpeech2_multi import AnonFastSpeech2
from .AnonFastSpeech2_codeswitching import AnonCSFastSpeech2
from utils import setup_logger, LANGS_SHORT_TO_LONG_CODE

logger = setup_logger(__name__)

class ImsTTS:

    def __init__(self, hifigan_path, fastspeech_path, device, embedding_path=None, output_sr=16000, lang='en',
                 accent_lang=None, cs_languages=None):
        self.device = device
        self.output_sr = output_sr

        if lang == 'cs':
            if cs_languages is not None:
                cs_languages = [LANGS_SHORT_TO_LONG_CODE[lang] if len(lang) == 2 else lang for lang in cs_languages]
            if accent_lang and len(accent_lang) == 2:
                accent_lang = LANGS_SHORT_TO_LONG_CODE[accent_lang]
            self.model = AnonCSFastSpeech2(device=self.device, vocoder_model_path=hifigan_path,
                                           tts_model_path=fastspeech_path, embedding_model_path=embedding_path,
                                           accent_lang=accent_lang, cs_languages=cs_languages)
        else:
            if len(lang) == 2:
                lang = LANGS_SHORT_TO_LONG_CODE[lang]
            self.model = AnonFastSpeech2(device=self.device, vocoder_model_path=hifigan_path,
                                         tts_model_path=fastspeech_path, embedding_model_path=embedding_path,
                                         language=lang)

    def read_text(self, text, speaker_embedding, text_is_phones=True, duration=None, pitch=None, energy=None,
                  start_silence=None, end_silence=None):
        if pitch is not None:
            pitch = pitch.transpose(0, 1)
        if energy is not None:
            energy = energy.transpose(0, 1)

        self.model.default_utterance_embedding = speaker_embedding.to(self.device)
        try:
            wav, sr = self.model(text=text, text_is_phonemes=text_is_phones, durations=duration, pitch=pitch, energy=energy)
        except IndexError:
            logger.info(f'Index Error for utterance {text}')
            return None
        except RuntimeError:
            logger.info(f'Runtime Error for utterance {text}')
            return None

        i = 0
        while wav.shape[0] < (0.5 * sr):  # 0.5 s
            # sometimes, the speaker embedding is so off that it leads to a practically empty audio
            # then, we need to sample a new embedding
            if i > 0 and i % 10 == 0:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-40, 40).to(self.device)
            else:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-2, 2).to(self.device)
            speaker_embedding = speaker_embedding * mask
            self.model.default_utterance_embedding = speaker_embedding.to(self.device)
            wav, sr = self.model(text=text, text_is_phonemes=text_is_phones, durations=duration, pitch=pitch, energy=energy)
            i += 1
            if i > 30:
                break
        if i > 0:
            logger.info(f'Synthesized utt in {i} takes')

        # start and end silence are computed for 16000, so we have to adapt this to different output sr
        factor = self.output_sr // 16000
        if start_silence is not None:
            start_sil = np.zeros([int(start_silence * factor)])
            wav = np.concatenate([start_sil, wav], axis=0)
        if end_silence is not None:
            end_sil = np.zeros([int(end_silence * factor)])
            wav = np.concatenate([wav, end_sil], axis=0)

        if self.output_sr != sr:
            wav = resampy.resample(wav, sr, self.output_sr)

        return wav
