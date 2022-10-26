import soundfile as sf
import torch
from torch.optim import SGD
from .Preprocessing.AudioPreprocessor import AudioPreprocessor

from .InferenceInterfaces.AnonFastSpeech2 import AnonFastSpeech2
from .Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from .TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from .TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from .TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from .TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Parselmouth


class UtteranceCloner:

    def __init__(self,
                 path_to_fastspeech_model,
                 path_to_hifigan_model,
                 path_to_aligner_model,
                 path_to_embed_model, device):
        self.tts = AnonFastSpeech2(device=device,
                                   path_to_fastspeech_model=path_to_fastspeech_model,
                                   path_to_hifigan_model=path_to_hifigan_model,
                                   path_to_embed_model=path_to_embed_model)
        self.ap = AudioPreprocessor(input_sr=16000, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        self.tf = ArticulatoryCombinedTextFrontend(language="en")
        self.device = device
        self.aligner_weights = torch.load(path_to_aligner_model, map_location='cpu')["asr_model"]
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

    def extract_prosody(self,
                        transcript,
                        ref_audio_path,
                        lang="en",
                        on_line_fine_tune=True,
                        input_is_phones=False,
                        random_offset_lower=None,
                        random_offset_higher=None):
        acoustic_model = Aligner()
        acoustic_model.load_state_dict(self.aligner_weights)
        acoustic_model = acoustic_model.to(self.device)
        parsel = Parselmouth(reduction_factor=1, fs=16000)
        energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        dc = DurationCalculator(reduction_factor=1)
        wave, sr = sf.read(ref_audio_path)
        if self.tf.language != lang:
            self.tf = ArticulatoryCombinedTextFrontend(language=lang)
        if self.ap.sr != sr:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        try:
            norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
        except ValueError:
            print('Something went wrong, the reference wave might be too short.')
            raise RuntimeError

        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(norm_wave, self.silero_model, sampling_rate=16000)
        start_silence = speech_timestamps[0]['start']
        end_silence = len(norm_wave) - speech_timestamps[-1]['end']
        norm_wave = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = self.tf.string_to_tensor(transcript, handle_missing=True, input_phonemes=input_is_phones).squeeze(0)
        melspec = self.ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
        melspec_length = torch.LongTensor([len(melspec)]).numpy()

        if on_line_fine_tune:
            # we fine-tune the aligner for a couple steps using SGD. This makes cloning pretty slow, but the results are greatly improved.
            steps = 3
            tokens = list()  # we need an ID sequence for training rather than a sequence of phonological features
            for vector in text:
                if vector[21] == 0:  # we don't include word boundaries when performing alignment, since they are not always present in audio.
                    for phone in self.tf.phone_to_vector:
                        if vector.numpy().tolist()[13:] == self.tf.phone_to_vector[phone][13:]:
                            # the first 12 dimensions are for modifiers, so we ignore those when trying to find the phoneme in the ID lookup
                            tokens.append(self.tf.phone_to_id[phone])
                            # this is terribly inefficient, but it's fine
                            break
            tokens = torch.LongTensor(tokens).squeeze().to(self.device)
            tokens_len = torch.LongTensor([len(tokens)]).to(self.device)
            mel = melspec.unsqueeze(0).to(self.device)
            mel.requires_grad = True
            mel_len = torch.LongTensor([len(mel[0])]).to(self.device)
            # actual fine-tuning starts here
            optim_asr = SGD(acoustic_model.parameters(), lr=0.1)
            acoustic_model.train()
            for _ in list(range(steps)):
                pred = acoustic_model(mel)
                loss = acoustic_model.ctc_loss(pred.transpose(0, 1).log_softmax(2), tokens, mel_len, tokens_len)
                optim_asr.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(acoustic_model.parameters(), 1.0)
                optim_asr.step()
            acoustic_model.eval()

        # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
        # We note the index of word boundaries and insert durations of 0 afterwards
        text_without_word_boundaries = list()
        indexes_of_word_boundaries = list()
        for phoneme_index, vector in enumerate(text):
            if vector[21] == 0:
                text_without_word_boundaries.append(vector.numpy().tolist())
            else:
                indexes_of_word_boundaries.append(phoneme_index)
        matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)

        alignment_path = acoustic_model.inference(mel=melspec.to(self.device),
                                                  tokens=matrix_without_word_boundaries.to(self.device),
                                                  return_ctc=False)

        duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

        for index_of_word_boundary in indexes_of_word_boundaries:
            duration = torch.cat([duration[:index_of_word_boundary],
                                  torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                                  duration[index_of_word_boundary:]])

        last_vec = None
        for phoneme_index, vec in enumerate(text):
            if last_vec is not None:
                if last_vec.numpy().tolist() == vec.numpy().tolist():
                    # we found a case of repeating phonemes!
                    # now we must repair their durations by giving the first one 3/5 of their sum and the second one 2/5 (i.e. the rest)
                    dur_1 = duration[phoneme_index - 1]
                    dur_2 = duration[phoneme_index]
                    total_dur = dur_1 + dur_2
                    new_dur_1 = int((total_dur / 5) * 3)
                    new_dur_2 = total_dur - new_dur_1
                    duration[phoneme_index - 1] = new_dur_1
                    duration[phoneme_index] = new_dur_2
            last_vec = vec

        energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                             input_waves_lengths=norm_wave_length,
                             feats_lengths=melspec_length,
                             text=text,
                             durations=duration.unsqueeze(0),
                             durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        if random_offset_lower is not None and random_offset_higher is not None:
            #print(f'use random offset lower of {random_offset_lower}')
            scales = torch.randint(low=random_offset_lower, high=random_offset_higher, size=energy.size()).float() / 100
            energy = energy * scales
        pitch = parsel(input_waves=norm_wave.unsqueeze(0),
                       input_waves_lengths=norm_wave_length,
                       feats_lengths=melspec_length,
                       text=text,
                       durations=duration.unsqueeze(0),
                       durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        if random_offset_lower is not None and random_offset_higher is not None:
            #print(f'use random offset higher of {random_offset_higher}')
            scales = torch.randint(low=random_offset_lower, high=random_offset_higher, size=pitch.size()).float() / 100
            pitch = pitch * scales

        return duration, pitch, energy, start_silence, end_silence

    def clone_utterance(self,
                        path_to_reference_audio,
                        reference_transcription,
                        clone_speaker_identity=True,
                        speaker_embedding=None,
                        lang="en",
                        input_is_phones=False,
                        random_offset_lower=None,
                        random_offset_higher=None):
        # speaker_embedding can be either a path to a reference audio or a pre-extracted embedding
        if clone_speaker_identity or speaker_embedding is not None:
            prev_speaker_embedding = self.tts.default_utterance_embedding.clone().detach()
            if clone_speaker_identity:
                self.tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio)
            elif type(speaker_embedding) == str:
                self.tts.set_utterance_embedding(path_to_reference_audio=speaker_embedding)
            else:
                # remaining case should be that it's already a tensor, but we don't explicitly check
                #print('use given speaker embedding')
                self.tts.default_utterance_embedding = speaker_embedding.to(self.device)

        duration, pitch, energy, silence_frames_start, silence_frames_end = self.extract_prosody(reference_transcription,
                                                                                                 path_to_reference_audio,
                                                                                                 lang=lang,
                                                                                                 input_is_phones=input_is_phones,
                                                                                                 random_offset_lower=random_offset_lower,
                                                                                                 random_offset_higher=random_offset_higher)
        self.tts.set_language(lang)
        start_sil = torch.zeros([silence_frames_start * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        end_sil = torch.zeros([silence_frames_end * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        cloned_speech = self.tts(reference_transcription, view=False, durations=duration, pitch=pitch, energy=energy,
                                 text_is_phonemes=input_is_phones)
        cloned_utt = torch.cat((start_sil, cloned_speech, end_sil), dim=0)
        if clone_speaker_identity or speaker_embedding is not None:
            self.tts.default_utterance_embedding = prev_speaker_embedding.to(self.device)  # return to normal
        return cloned_utt

    def biblical_accurate_angel_mode(self,
                                     path_to_reference_audio,
                                     reference_transcription,
                                     list_of_speaker_references_for_ensemble,
                                     lang="en",
                                     input_is_phones=False,
                                     random_offset_lower=None,
                                     random_offset_higher=None):
        # list_of_speaker_references_for_ensemble can be a list of filepaths or a list of pre-extracted embeddings
        prev_speaker_embedding = self.tts.default_utterance_embedding.clone().detach()

        duration, pitch, energy, silence_frames_start, silence_frames_end = self.extract_prosody(reference_transcription,
                                                                                                 path_to_reference_audio,
                                                                                                 lang=lang,
                                                                                                 input_is_phones=input_is_phones,
                                                                                                 random_offset_lower=random_offset_lower,
                                                                                                 random_offset_higher=random_offset_higher)
        self.tts.set_language(lang)
        start_sil = torch.zeros([silence_frames_start * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        end_sil = torch.zeros([silence_frames_end * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        list_of_cloned_speeches = list()
        if type(list_of_speaker_references_for_ensemble[0]) == str:
            for path in list_of_speaker_references_for_ensemble:
                self.tts.set_utterance_embedding(path_to_reference_audio=path)
                list_of_cloned_speeches.append(self.tts(reference_transcription, view=False, durations=duration,
                                                        pitch=pitch, energy=energy, text_is_phonemes=input_is_phones))
        else:
            for embed in list_of_speaker_references_for_ensemble:
                self.tts.set_utterance_embedding(embedding=embed)
                list_of_cloned_speeches.append(self.tts(reference_transcription, view=False, durations=duration,
                                                        pitch=pitch, energy=energy, text_is_phonemes=input_is_phones))
        cloned_speech = torch.stack(list_of_cloned_speeches).mean(dim=0)
        cloned_utt = torch.cat((start_sil, cloned_speech, end_sil), dim=0)
        self.tts.default_utterance_embedding = prev_speaker_embedding.to(self.device)  # return to normal
        return cloned_utt


def integration_test():
    """
    This has to be called from one directory higher up so that the relative imports work.

    Exemplary calls for all features are included.
    """
    uc = UtteranceCloner(path_to_fastspeech_model="IMSToucan/Models/FastSpeech2_VP/best.pt",
                         path_to_hifigan_model="IMSToucan/Models/HiFiGAN_combined/best.pt",
                         path_to_aligner_model="IMSToucan/Models/Aligner/aligner.pt",
                         path_to_embed_model="IMSToucan/Models/Embedding/embedding_function.pt",
                         device="cpu")

    wav = uc.clone_utterance(path_to_reference_audio="IMSToucan/audios/test.wav",
                             reference_transcription="Hello world, this is a test.",
                             clone_speaker_identity=False,  # if True, clones the prosody of the sample AND the voice from the same sample. For
                             # using a different voice, use next argument.
                             speaker_embedding=None,
                             # speaker_embedding can be either a path to a reference audio or a pre-extracted embedding
                             lang="en")
    sf.write(file="IMSToucan/audios/integration_test_normal.wav", data=wav.cpu().numpy(), samplerate=48000)

    wav = uc.clone_utterance(path_to_reference_audio="IMSToucan/audios/test.wav",
                             reference_transcription="Hello world, this is a test.",
                             clone_speaker_identity=False,  # if True, clones the prosody of the sample AND the voice from the same sample. For
                             # using a different voice, use next argument.
                             speaker_embedding=None,
                             # speaker_embedding can be either a path to a reference audio or a pre-extracted embedding
                             lang="en",
                             random_offset_lower=50,  # corresponds to value * 0.5
                             random_offset_higher=150)  # corresponds to value * 1.5
    sf.write(file="IMSToucan/audios/integration_test_randomly_modded.wav", data=wav.cpu().numpy(), samplerate=48000)

    wav = uc.biblical_accurate_angel_mode(path_to_reference_audio="IMSToucan/audios/test.wav",
                                          reference_transcription="Hello world, this is a test.",
                                          list_of_speaker_references_for_ensemble=["IMSToucan/audios/zischler.wav",
                                                                                   "IMSToucan/audios/nadja_sample.wav",
                                                                                   "IMSToucan/audios/test.wav"],  # paths as strings
                                          # or embeddings in torch.tensor can go here, but don't mix
                                          lang="en")
    sf.write(file="IMSToucan/audios/integration_test_angelic.wav", data=wav.cpu().numpy(), samplerate=48000)
