import torch
torch.set_num_threads(1)
import soundfile as sf
from huggingface_hub import hf_hub_download
import librosa

from anonymization.modules.tts.IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor
from anonymization.modules.tts.IMSToucan.Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from anonymization.modules.tts.IMSToucan.Preprocessing.articulatory_features import get_feature_to_index_lookup
from anonymization.modules.tts.IMSToucan.Modules.Aligner.Aligner import Aligner
from anonymization.modules.tts.IMSToucan.Modules.ToucanTTS.DurationCalculator import DurationCalculator
from anonymization.modules.tts.IMSToucan.Modules.ToucanTTS.EnergyCalculator import EnergyCalculator
from anonymization.modules.tts.IMSToucan.Modules.ToucanTTS.PitchCalculator import Parselmouth

from utils import setup_logger, LANGS_SHORT_TO_LONG_CODE

logger = setup_logger(__name__)

class ImsProsodyExtractor:

    def __init__(self, aligner_path, device, on_line_fine_tune=True, language='en'):
        self.on_line_fine_tune = on_line_fine_tune

        if len(language) == 2:
            language = LANGS_SHORT_TO_LONG_CODE[language]

        self.ap = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=False)
        self.tf = ArticulatoryCombinedTextFrontend(language=language, device=device)
        self.device = device

        if not aligner_path:
            aligner_path = hf_hub_download(repo_id='Flux9665/ToucanTTS', filename='Aligner.pt')
        self.acoustic_model = Aligner()
        self.aligner_weights = torch.load(aligner_path, map_location=self.device)['asr_model']
        self.acoustic_model.load_state_dict(self.aligner_weights)
        self.acoustic_model = self.acoustic_model.to(self.device)
        self.acoustic_model.eval()

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
        # careful: assumes 16kHz or 8kHz audio
        self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  onnx=False,
                                                  verbose=False)
        (self.get_speech_timestamps, _, _, _, _) = utils
        torch.set_grad_enabled(True)  # finding this issue was very infuriating: silero sets
        # this to false globally during model loading rather than using inference mode or no_grad
        self.parsel = Parselmouth(reduction_factor=1, fs=16000)
        self.energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        self.dc = DurationCalculator(reduction_factor=1)

    def extract_prosody(self, transcript, ref_audio_path, input_is_phones=False):
        # remove special characters
        transcript = transcript.replace('«', ''). replace('»', '')
        # we need to reinitialize the aligner weights for each utterance again if we fine tune it for each utterance
        if self.on_line_fine_tune:
            self.acoustic_model.load_state_dict(self.aligner_weights)
            self.acoustic_model.eval()

        wave, sr = sf.read(ref_audio_path)
        if len(wave.shape) > 1:  # oh no, we found a stereo audio!
            if len(wave[0]) == 2:  # let's figure out whether we need to switch the axes
                wave = wave.transpose()  # if yes, we switch the axes.
        wave = librosa.to_mono(wave)

        if self.ap.sr != sr:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=False)
        try:
            norm_wave = self.ap.normalize_audio(audio=wave)
        except ValueError:
            logger.error('Something went wrong, the reference wave might be too short.')
            raise RuntimeError

        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(norm_wave, self.silero_model, sampling_rate=16000)
        if len(speech_timestamps) == 0:
            speech_timestamps = [{'start': 0, 'end': len(norm_wave)}]
        start_silence = speech_timestamps[0]['start']
        end_silence = len(norm_wave) - speech_timestamps[-1]['end']
        norm_wave = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = self.tf.string_to_tensor(transcript, handle_missing=True, input_phonemes=input_is_phones).squeeze(0)
        features = self.ap.audio_to_mel_spec_tensor(audio=norm_wave, explicit_sampling_rate=16000).transpose(0, 1)
        feature_length = torch.LongTensor([len(features)]).numpy()

        if self.on_line_fine_tune:
            # we fine-tune the aligner for a couple steps using SGD. This makes cloning pretty slow, but the results are greatly improved.
            steps = 4
            tokens = self.tf.text_vectors_to_id_sequence(text_vector=text)  # we need an ID sequence for training rather than a sequence of phonological features
            tokens = torch.LongTensor(tokens).squeeze().to(self.device)
            tokens_len = torch.LongTensor([len(tokens)]).to(self.device)
            mel = features.unsqueeze(0).to(self.device)
            mel_len = torch.LongTensor([len(mel[0])]).to(self.device)
            # actual fine-tuning starts here
            optim_asr = torch.optim.Adam(self.acoustic_model.parameters(), lr=0.00001)
            self.acoustic_model.train()
            for _ in list(range(steps)):
                pred = self.acoustic_model(mel.clone())
                loss = self.acoustic_model.ctc_loss(pred.transpose(0, 1).log_softmax(2), tokens, mel_len, tokens_len)
                optim_asr.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acoustic_model.parameters(), 1.0)
                optim_asr.step()
            self.acoustic_model.eval()

        # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
        # We note the index of word boundaries and insert durations of 0 afterwards
        text_without_word_boundaries = list()
        indexes_of_word_boundaries = list()
        for phoneme_index, vector in enumerate(text):
            if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                text_without_word_boundaries.append(vector.numpy().tolist())
            else:
                indexes_of_word_boundaries.append(phoneme_index)
        matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)

        alignment_path = self.acoustic_model.inference(features=features.to(self.device),
                                                       tokens=matrix_without_word_boundaries.to(self.device),
                                                       return_ctc=False)

        duration = self.dc(torch.LongTensor(alignment_path), vis=None).cpu()

        for index_of_word_boundary in indexes_of_word_boundaries:
            duration = torch.cat([duration[:index_of_word_boundary],
                                  torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                                  duration[index_of_word_boundary:]])

        energy = self.energy_calc(input_waves=norm_wave.unsqueeze(0),
                                  input_waves_lengths=norm_wave_length,
                                  feats_lengths=feature_length,
                                  text=text,
                                  durations=duration.unsqueeze(0),
                                  durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()

        pitch = self.parsel(input_waves=norm_wave.unsqueeze(0),
                            input_waves_lengths=norm_wave_length,
                            feats_lengths=feature_length,
                            text=text,
                            durations=duration.unsqueeze(0),
                            durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()

        return duration, pitch, energy, start_silence, end_silence
