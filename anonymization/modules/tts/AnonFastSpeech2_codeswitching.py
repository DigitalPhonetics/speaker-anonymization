import os
import librosa
import pyloudnorm
import soundfile
import torch
from speechbrain.inference import EncoderClassifier
from torchaudio.transforms import Resample
from huggingface_hub import hf_hub_download
import sys
from pathlib import Path
import logging

sys.path.insert(0, str((Path(__file__).parent / 'IMSToucan').absolute()))

from .IMSToucan.Modules.ToucanTTS.InferenceToucanTTS import ToucanTTS
from anonymization.modules.tts.toucan_codeswitching.CSInferenceToucanTTS import CSToucanTTS
from .IMSToucan.Modules.Vocoder.HiFiGAN_Generator import HiFiGAN
from .IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor
from .IMSToucan.Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend, get_language_id
from .IMSToucan.Preprocessing.articulatory_features import get_feature_to_index_lookup
from utils import LanguageTextProcessing

logger = logging.getLogger(__name__)

class AnonCSFastSpeech2(torch.nn.Module):


    def __init__(self, vocoder_model_path, tts_model_path, embedding_model_path, device='cpu', accent_lang=None,
                 cs_languages=None):
        super().__init__()
        self.device = device
        self.accent_lang = accent_lang
        print(f'Accent lang: {self.accent_lang}')

        if tts_model_path is None:
            tts_model_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt")
        if vocoder_model_path is None:
            vocoder_model_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="Vocoder.pt")

        ################################
        #   build text to phone        #
        ################################
        if cs_languages is None:
            cs_languages = ['eng', 'cmn']

        self.text2phone = {}
        for lang in cs_languages:
            self.text2phone[lang] = ArticulatoryCombinedTextFrontend(language=lang, add_silence_to_end=True, device=self.device)

        self.langtextprocess = LanguageTextProcessing(cs_languages, device=self.device)

        ################################
        #   load weights               #
        ################################
        checkpoint = torch.load(tts_model_path, map_location='cpu')
        if self.accent_lang is None:
            logger.info('Select language per phone for accent pronunciation')
            self.phone2mel = CSToucanTTS(weights=checkpoint['model'], config=checkpoint['config'])
            self.accent_lang_id = None
        else:
            logger.info(f'Use language for accent pronunciation: {self.accent_lang}')
            self.phone2mel = ToucanTTS(weights=checkpoint['model'], config=checkpoint['config'])
            self.accent_lang_id = get_language_id(self.accent_lang)

        with torch.no_grad():
            self.phone2mel.store_inverse_all()  # this also removes weight norm
        self.phone2mel = self.phone2mel.to(torch.device(device))

        ######################################
        #  load features to style models     #
        ######################################
        self.speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                                           run_opts={'device': str(device)},
                                                                           savedir=embedding_model_path)

        ################################
        #  load mel to wave model      #
        ################################
        vocoder_checkpoint = torch.load(vocoder_model_path, map_location='cpu')
        self.vocoder = HiFiGAN()
        self.vocoder.load_state_dict(vocoder_checkpoint)
        self.vocoder = self.vocoder.to(device).eval()
        self.vocoder.remove_weight_norm()
        self.meter = pyloudnorm.Meter(24000)

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint['default_emb'].to(self.device)
        self.ap = AudioPreprocessor(input_sr=100, output_sr=16000, device=device)
        self.phone2mel.eval()
        self.vocoder.eval()
        # self.lang_id = get_language_id(language)
        self.to(torch.device(device))
        self.eval()

        self.silence_index = get_feature_to_index_lookup()['silence']

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        if type(path_to_reference_audio) != list:
            path_to_reference_audio = [path_to_reference_audio]

        if len(path_to_reference_audio) > 0:
            for path in path_to_reference_audio:
                assert os.path.exists(path)
            speaker_embs = list()
            for path in path_to_reference_audio:
                wave, sr = soundfile.read(path)
                if len(wave.shape) > 1:  # oh no, we found a stereo audio!
                    if len(wave[0]) == 2:  # let's figure out whether we need to switch the axes
                        wave = wave.transpose()  # if yes, we switch the axes.
                wave = librosa.to_mono(wave)
                wave = Resample(orig_freq=sr, new_freq=16000).to(self.device)(
                    torch.tensor(wave, device=self.device, dtype=torch.float32))
                speaker_embedding = self.speaker_embedding_func_ecapa.encode_batch(
                    wavs=wave.to(self.device).squeeze().unsqueeze(0)).squeeze()
                speaker_embs.append(speaker_embedding)
            self.default_utterance_embedding = sum(speaker_embs) / len(speaker_embs)

    def forward(self,
                text,
                view=False,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                energy_variance_scale=1.0,
                pause_duration_scaling_factor=1.0,
                durations=None,
                pitch=None,
                energy=None,
                text_is_phonemes=False,
                return_plot_as_filepath=False,
                loudness_in_db=-29.0,
                prosody_creativity=0.1):
        """
        duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
        energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        with torch.inference_mode():
            # step 1: separate string into different languages
            lang_substrings = self.langtextprocess.separate_lang_substrings(text)
            # step 2: phonemize each language string; concatenate all phones into one tensor but keep track of which phone is in which language
            phones = []
            lang_per_phone = []
            n = len(lang_substrings)
            for i, (lang, substring) in enumerate(lang_substrings):
                sub_phones = self.text2phone[lang].string_to_tensor(substring, input_phonemes=text_is_phonemes).to(torch.device(self.device))
                lang_id = get_language_id(lang)  # lang_id is a tensor with a single number
                # silence is added at the beginning and end of each phone string, as well as the EOS token

                # we keep the silence at the beginning if:
                # - it is the first phone string in the sequence OR
                # - the phone string has at least 3 words
                # we only insert the pause if the phone string before did not end with a pause (we don't want a double pause)
                # otherwise, we remove the pause at the beginning
                if i > 0: # phone string is not at the beginning
                    if sub_phones.shape[0] == 2:  # phone consists only of pause and EOS, probably because it is something like ","
                        sub_phones = sub_phones  # don't cut silence
                    elif len(substring.split()) < 3: # substring has less than 3 words
                        sub_phones = sub_phones[1:]  # cut silence at beginning
                    elif phones[-1][-1][self.silence_index] == 1: # previous phone string ended with silence
                        sub_phones = sub_phones[1:]  # cut silence at beginning

                # we keep the silence at the end if:
                # - it is the last phone string in the sequence (in that case we also keep the EOS) OR
                # - the phone string has at least 3 words
                # otherwise, we remove the pause at the end
                if i < (n - 1): # phone string is not the last
                    if sub_phones.shape[0] == 2:  # phone consists only of pause and EOS, probably because it is something like ","
                        sub_phones = sub_phones[:-1]  # cut only EOS
                    elif len(substring.split()) < 3: # substring has less than 3 words
                        sub_phones = sub_phones[:-2]  # cut silence and EOS at the end
                    else:  # phone string has at least 3 words
                        sub_phones = sub_phones[:-1]  # cut only the EOS (keep silence at the end)

                phones.append(sub_phones)
                lang_per_phone.append(lang_id.expand(len(sub_phones))) # repeat value of lang_id for each phone in phone string

            if len(phones) == 0:
                return None, None

            phones = torch.concatenate(phones)
            lang_per_phone = torch.concatenate(lang_per_phone)

            mel, durations, pitch, energy = self.phone2mel(phones,
                                                           return_duration_pitch_energy=True,
                                                           utterance_embedding=self.default_utterance_embedding,
                                                           durations=durations,
                                                           pitch=pitch,
                                                           energy=energy,
                                                           lang_id=lang_per_phone if self.accent_lang is None else self.accent_lang_id,
                                                           duration_scaling_factor=duration_scaling_factor,
                                                           pitch_variance_scale=pitch_variance_scale,
                                                           energy_variance_scale=energy_variance_scale,
                                                           pause_duration_scaling_factor=pause_duration_scaling_factor,
                                                           prosody_creativity=prosody_creativity)

            wave = self.vocoder(mel.unsqueeze(0))
            wave = wave.squeeze().cpu()
        wave = wave.numpy()
        sr = 24000
        try:
            loudness = self.meter.integrated_loudness(wave)
            wave = pyloudnorm.normalize.loudness(wave, loudness, loudness_in_db)
        except ValueError:
            # if the audio is too short, a value error will arise
            pass

        return wave, sr
