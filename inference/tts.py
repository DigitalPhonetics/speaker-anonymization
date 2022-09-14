from tqdm import tqdm
import soundfile
import torch

from IMSToucan.InferenceInterfaces.AnonFastSpeech2 import AnonFastSpeech2
from utils import create_clean_dir


class InferenceTTS:

    def __init__(self, hifigan_model_name, fastspeech_model_name, anon_model_name, asr_model_name, model_dir,
                 results_dir, device, force_compute=False):
        self.force_compute = force_compute
        self.device = device

        model_name = f'{hifigan_model_name}_{fastspeech_model_name}'
        hifigan_path = model_dir / 'tts' / 'HiFiGAN_combined' / hifigan_model_name
        fastspeech_path = model_dir / 'tts' / 'FastSpeech2_Multi' / fastspeech_model_name

        self.results_dir = results_dir / 'speech' / model_name / anon_model_name / asr_model_name

        self.model = AnonFastSpeech2(device=self.device, path_to_hifigan_model=hifigan_path,
                                     path_to_fastspeech_model=fastspeech_path)


    def read_texts(self, dataset, texts, anon_embeddings, utt2spk, text_is_phonemes=False, force_compute=False,
                   save_wav=True, emb_level='spk'):
        dataset_results_dir = self.results_dir / dataset
        wav_scp = {}
        wavs = {}
        force_compute = force_compute or self.force_compute

        if dataset_results_dir.exists() and not force_compute:
            already_anon_utts = {x.stem: str(x.absolute()) for x in dataset_results_dir.glob('*.wav')}
            if already_anon_utts:
                print(f'No synthesis necessary for {len(already_anon_utts)} of {len(texts)} utterances...')
                texts = {utt: text for utt, text in texts.items() if utt not in already_anon_utts.keys()}
                wav_scp = already_anon_utts

        if texts:
            print(f'Synthesize {len(texts)} utterances...')
            new = True
            if self.force_compute:
                create_clean_dir(dataset_results_dir)
            elif not dataset_results_dir.exists():
                dataset_results_dir.mkdir(parents=True)

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

                if i > 0:
                    print(f'Synthesized utt {utt} in {i} takes')

                if save_wav:
                    soundfile.write(file=out_file, data=wav.cpu().numpy(), samplerate=48000)
                wav_scp[utt] = out_file

        return wav_scp, wavs





