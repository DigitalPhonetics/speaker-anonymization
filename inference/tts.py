from tqdm import tqdm
import soundfile
import torch
import resampy

from IMSToucan.InferenceInterfaces.AnonFastSpeech2 import AnonFastSpeech2
from IMSToucan.UtteranceCloner import UtteranceCloner
from utils import create_clean_dir, read_kaldi_format


class InferenceTTS:

    def __init__(self, hifigan_model_name, fastspeech_model_name, aligner_model_name, embedding_model_name,
                 model_dir, results_dir, device, data_dir, clone_prosody=False, output_sr=16000):
        self.device = device
        hifigan_path = model_dir / 'tts' / 'HiFiGAN_combined' / hifigan_model_name
        fastspeech_path = model_dir / 'tts' / 'FastSpeech2_Multi' / fastspeech_model_name
        aligner_path = model_dir / 'tts' / 'Aligner' / aligner_model_name
        embedding_path = model_dir / 'tts' / 'Embedding' / embedding_model_name if embedding_model_name else None
        self.data_dir = data_dir

        self.clone_prosody = clone_prosody
        self.output_sr = output_sr

        self.results_dir = results_dir

        if self.clone_prosody:
            self.model = UtteranceCloner(path_to_hifigan_model=hifigan_path, path_to_fastspeech_model=fastspeech_path,
                                         path_to_aligner_model=aligner_path, path_to_embed_model=embedding_path,
                                         device=self.device)
        else:
            self.model = AnonFastSpeech2(device=self.device, path_to_hifigan_model=hifigan_path,
                                         path_to_fastspeech_model=fastspeech_path, path_to_embed_model=embedding_path)


    def read_texts(self, dataset, texts, anon_embeddings, utt2spk, text_is_phonemes=False, force_compute=False,
                   save_wav=True, emb_level='spk', random_offset_lower=None, random_offset_higher=None):
        dataset_results_dir = self.results_dir / dataset
        wav_scp = {}
        wavs = {}

        if dataset_results_dir.exists() and not force_compute:
            already_anon_utts = {x.stem: str(x.absolute()) for x in dataset_results_dir.glob('*.wav')}
            if already_anon_utts:
                print(f'No synthesis necessary for {len(already_anon_utts)} of {len(texts)} utterances...')
                texts = {utt: text for utt, text in texts.items() if utt not in already_anon_utts.keys()}
                wav_scp = already_anon_utts

        if texts:
            print(f'Synthesize {len(texts)} utterances...')
            new = True
            if force_compute:
                create_clean_dir(dataset_results_dir)
            elif not dataset_results_dir.exists():
                dataset_results_dir.mkdir(parents=True)

            if self.clone_prosody:
                original_audios = read_kaldi_format(self.data_dir / dataset / 'wav.scp')
                wav_scp.update(
                    self._synthesize_with_prosody_cloning(texts, original_audios, anon_embeddings, utt2spk,
                                                          dataset_results_dir, emb_level,
                                                          text_is_phonemes, save_wav, random_offset_lower,
                                                          random_offset_higher))
            else:
                wav_scp.update(self._synthesize(texts, anon_embeddings, utt2spk, dataset_results_dir, emb_level,
                                                text_is_phonemes, save_wav))

        return wav_scp, wavs

    def _synthesize(self, texts, anon_embeddings, utt2spk, dataset_results_dir, emb_level, text_is_phonemes, save_wav):
        print('Synthesize without prosody cloning')
        wav_scp = {}

        for utt, text in tqdm(texts.items()):
            if emb_level == 'spk':
                speaker = utt2spk[utt]
                speaker_embedding = anon_embeddings.get_embedding_for_speaker(speaker)
            else:
                speaker_embedding = anon_embeddings.get_embedding_for_speaker(utt)
            out_file = str((dataset_results_dir / f'{utt}.wav').absolute())

            self.model.default_utterance_embedding = speaker_embedding.to(self.device)
            wav = self.model(text=text, text_is_phonemes=text_is_phonemes)

            i = 0
            while wav.shape[0] < 24000:  # 0.5 s
                # sometimes, the speaker embedding is so off that it leads to a practically empty audio
                # then, we need to sample a new embedding
                if i > 0 and i % 10 == 0:
                    mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-40, 40).to(self.device)
                else:
                    mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-2, 2).to(self.device)
                speaker_embedding = speaker_embedding * mask
                self.model.default_utterance_embedding = speaker_embedding.to(self.device)
                wav = self.model(text=text, text_is_phonemes=text_is_phonemes)
                i += 1
                if i > 30:
                    break

            if i > 0:
                print(f'Synthesized utt {utt} in {i} takes')

            if save_wav:
                if self.output_sr != 48000:
                    wav = resampy.resample(wav.cpu().numpy(), 48000, self.output_sr)
                else:
                    wav = wav.cpu().numpy()
                soundfile.write(file=out_file, data=wav, samplerate=self.output_sr)
            wav_scp[utt] = out_file

        return wav_scp

    def _synthesize_with_prosody_cloning(self, texts, original_audios, anon_embeddings, utt2spk, dataset_results_dir,
                                         emb_level, text_is_phonemes, save_wav, random_offset_lower,
                                         random_offset_higher):
        print('Synthesize with prosody cloning')
        wav_scp = {}

        for utt, text in tqdm(texts.items()):
            if emb_level == 'spk':
                speaker = utt2spk[utt]
                speaker_embedding = anon_embeddings.get_embedding_for_speaker(speaker)
            else:
                speaker_embedding = anon_embeddings.get_embedding_for_speaker(utt)

            out_file = str((dataset_results_dir / f'{utt}.wav').absolute())
            reference_audio_path = original_audios[utt]

            wav = self.model.clone_utterance(path_to_reference_audio=reference_audio_path,
                                             reference_transcription=text,
                                             clone_speaker_identity=False, lang="en",
                                             input_is_phones=text_is_phonemes,
                                             random_offset_lower=random_offset_lower,
                                             random_offset_higher=random_offset_higher,
                                             speaker_embedding=speaker_embedding
                                             )

            i = 0
            while wav.shape[0] < 24000:  # 0.5 s
                if i > 0 and i % 10 == 0:
                    mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-40, 40).to(self.device)
                else:
                    mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-2, 2).to(self.device)
                speaker_embedding = speaker_embedding * mask
                wav = self.model.clone_utterance(path_to_reference_audio=reference_audio_path,
                                                 reference_transcription=text,
                                                 clone_speaker_identity=False, lang="en",
                                                 input_is_phones=text_is_phonemes,
                                                 random_offset_lower=random_offset_lower,
                                                 random_offset_higher=random_offset_higher,
                                                 speaker_embedding=speaker_embedding
                                                 )
                i += 1
                if i == 30:
                    break

            if i > 0:
                print(f'Synthesized utt {utt} in {i} takes')

            if save_wav:
                if self.output_sr != 48000:
                    wav = resampy.resample(wav.cpu().numpy(), 48000, self.output_sr)
                else:
                    wav = wav.cpu().numpy()
                soundfile.write(file=out_file, data=wav, samplerate=self.output_sr)
            wav_scp[utt] = out_file
        return wav_scp



