import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from speechbrain.pretrained import EncoderClassifier

from utils import read_kaldi_format, save_kaldi_format
from IMSToucan.TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor


VALID_VEC_TYPES = {'xvector', 'ecapa', 'ecapa+xvector', 'style-embed'}


class SpeakerEmbeddings:

    def __init__(self, vec_type='xvector', emb_level='spk', device=torch.device('cpu')):
        self.vec_type = vec_type
        assert self.vec_type in VALID_VEC_TYPES, f'Invalid vec_type {self.vec_type}, must be one of {VALID_VEC_TYPES}'
        self.emb_level = emb_level
        self.device = device

        self.speakers = None  # in the case of utt-level embeddings, this will be utterances, else speakers
        self.utt2spk = {}
        self.genders = None
        self.idx2speakers = None
        self.speaker_vectors = None

    def __iter__(self):
        assert (self.speakers is not None) and (self.speaker_vectors is not None), \
            'Speaker vectors need to be extracted or loaded before they can be iterated!'
        assert len(self.speakers) == self.speaker_vectors.shape[0], \
            f'Not same amount of speakers and vectors, #speakers: {len(self.speakers)}, #vectors:' \
            f' {self.speaker_vectors.shape[0]}!'

        for speaker, idx in sorted(self.speakers.items(), key=lambda x: x[1]):
            yield speaker, self.speaker_vectors[idx]

    def __len__(self):
        return len(self.speakers) if self.speakers else 0

    def __getitem__(self, item):
        assert (self.speakers is not None) and (self.speaker_vectors is not None), \
            'Speaker vectors need to be extracted or loaded before they can be accessed!'
        assert item <= len(self), 'Index needs to be smaller or equal the number of speakers!'
        return self.idx2speakers[item], self.speaker_vectors[item]

    def extract_vectors_from_audio(self, data_dir: Path, model_path=Path('pretrained_models')):
        # The following lines download and load the corresponding speaker embedding model from huggingface and store
        # it in the corresponding savedir. If a model has been previously downloaded and stored already,
        # it is loaded from savedir instead of downloading it again.
        encoders = []
        if self.vec_type == 'style-embed':
            self._extract_style_embeddings_from_audio(data_dir=data_dir, model_path=model_path)
            return

        if 'ecapa' in self.vec_type:
            encoders.append(EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                           savedir=model_path / 'spkrec-ecapa-voxceleb',
                                                           run_opts={'device': self.device}))
        if 'xvector' in self.vec_type:
            encoders.append(EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb',
                                                           savedir=model_path / 'spkrec-xvect-voxceleb',
                                                           run_opts={'device': self.device}))

        recordings = read_kaldi_format(data_dir / 'wav.scp')
        utt2spk = read_kaldi_format(data_dir / 'utt2spk')
        spk2gender = read_kaldi_format(data_dir / 'spk2gender')

        spk2utt_ids = defaultdict(list)
        vectors = []

        i = 0
        for rec_name, rec_file in recordings.items():
            if self.emb_level == 'utt':
                speaker = rec_name
            else:  # speaker-level anonymization
                speaker = utt2spk[rec_name]
            self.utt2spk[rec_name] = utt2spk[rec_name]
            signal, fs = torchaudio.load(rec_file)
            vector = self._extract_embedding(wave=signal, sr=fs, encoders=encoders)

            spk2utt_ids[speaker].append(i)
            vectors.append(vector)
            i += 1

        if self.emb_level == 'utt':
            self.speakers = {speaker: id_list[0] for speaker, id_list in spk2utt_ids.items()}
            self.speaker_vectors = torch.stack(vectors, dim=0).to(self.device)
            spk2gender = {utt: spk2gender[speaker] for utt, speaker in utt2spk.items()}
        else:
            self.speakers, self.speaker_vectors = self._get_speaker_level_vectors(spk2utt_ids, torch.stack(
                vectors, dim=0).to(self.device))
        self.genders = [spk2gender[speaker] for speaker in self.speakers]
        self.idx2speakers = {idx: spk for spk, idx in self.speakers.items()}

    def set_vectors(self, speakers, vectors, genders, utt2spk):
        if not isinstance(speakers, dict):
            self.speakers = {speaker: idx for idx, speaker in enumerate(speakers)}
        else:
            self.speakers = speakers
        self.speaker_vectors = vectors
        self.genders = genders
        self.idx2speakers = {idx: spk for spk, idx in self.speakers.items()}
        self.utt2spk = utt2spk

    def add_vectors(self, speakers, vectors, genders, utt2spk):
        assert (self.speakers is not None) and (self.speaker_vectors is not None), \
            'Speaker vectors need to be extracted or loaded before new vectors can be added to them!'
        if not isinstance(speakers, dict):
            speakers = {speakers: idx for idx, speaker in enumerate(speakers)}

        new_speakers = list(speakers.keys() - self.speakers.keys())
        indices = [speakers[speaker] for speaker in new_speakers]
        last_known_index = len(self.speaker_vectors)

        new_speaker_dict = {speaker: last_known_index + i for i, speaker in enumerate(new_speakers)}
        self.speakers.update(new_speaker_dict)
        self.idx2speakers.update({idx: speaker for speaker, idx in new_speaker_dict.items()})
        self.speaker_vectors = torch.cat((self.speaker_vectors, vectors[indices]), dim=0)
        self.genders.extend([genders[idx] for idx in indices])
        if utt2spk:
            self.utt2spk.update({utt: utt2spk[utt] for utt in new_speaker_dict.keys()})

    def load_vectors(self, in_dir:Path):
        assert ((in_dir / f'spk2idx').exists() or (in_dir / f'utt2idx').exists()) and \
               ((in_dir / f'speaker_vectors.pt').exists()), \
            f'speaker_vectors.pt and either spk2idx or utt2idx must exist in {in_dir}!'

        spk2gender = read_kaldi_format(in_dir / 'spk2gender')
        self.speaker_vectors = torch.load(in_dir / f'speaker_vectors.pt', map_location=self.device)

        if self.emb_level == 'spk':
            spk2idx = read_kaldi_format(in_dir / f'spk2idx')
            self.idx2speakers = {int(idx): spk for spk, idx in spk2idx.items()}
            self.speakers = {spk: idx for idx, spk in self.idx2speakers.items()}
            self.genders = [spk2gender[spk] for spk in self.speakers]

        elif self.emb_level == 'utt':
            utt2idx = read_kaldi_format(in_dir / f'utt2idx')
            self.idx2speakers = {int(idx): spk for spk, idx in utt2idx.items()}
            self.utt2spk = read_kaldi_format(in_dir / 'utt2spk')
            self.speakers = {spk: idx for idx, spk in self.idx2speakers.items()}
            self.genders = [spk2gender[self.utt2spk[utt]] for utt in self.speakers]


    def save_vectors(self, out_dir:Path):
        assert (self.speakers is not None) and (self.speaker_vectors is not None), \
            'Speaker vectors need to be extracted or loaded before they can be stored!'
        out_dir.mkdir(exist_ok=True, parents=True)

        if self.emb_level == 'spk':
            spk2idx = {spk: idx for idx, spk in self.idx2speakers.items()}
            save_kaldi_format(spk2idx, out_dir / f'spk2idx')
        elif self.emb_level == 'utt':
            utt2idx = {spk: idx for idx, spk in self.idx2speakers.items()}
            save_kaldi_format(utt2idx, out_dir / f'utt2idx')
            save_kaldi_format(self.utt2spk, out_dir / 'utt2spk')

        spk2gender = self.get_spk2gender()
        save_kaldi_format(spk2gender, out_dir / 'spk2gender')
        torch.save(self.speaker_vectors, out_dir / f'speaker_vectors.pt')

    def get_embedding_for_speaker(self, speaker):
        idx = self.speakers[speaker]
        return self.speaker_vectors[idx]

    def get_spk2gender(self):
        if self.emb_level == 'spk':
            speaker_list = [speaker for speaker, idx in sorted(self.speakers.items(), key=lambda x: x[1])]
            spk2gender = {speaker: gender for speaker, gender in zip(speaker_list, self.genders)}
        elif self.emb_level == 'utt':
            speaker_list = [self.utt2spk[utt] for utt, idx in sorted(self.speakers.items(), key=lambda x: x[1])]
            spk2gender = {speaker: gender for speaker, gender in zip(speaker_list, self.genders)}
        else:
            spk2gender = {}
        return spk2gender

    def _get_speaker_level_vectors(self, spk2utt_ids, vectors):
        # speaker-level x-vector: mean of utterance-level x-vectors
        speakers = {}
        speaker_level_vecs = []

        i = 0
        for speaker, utt_list in spk2utt_ids.items():
            utt_vecs = vectors[utt_list]
            spk_vec = torch.mean(utt_vecs, dim=0)
            speakers[speaker] = i
            speaker_level_vecs.append(spk_vec)
            i += 1

        return speakers, torch.stack(speaker_level_vecs, dim=0).to(self.device)


    def _extract_embedding(self, wave, sr, encoders):
        # adapted from IMSToucan/Preprocessing/AudioPreprocessor
        norm_wave = self._normalize_wave(wave, sr)
        norm_wave = torch.tensor(np.trim_zeros(norm_wave.numpy()))

        spk_embs = [encoder.encode_batch(wavs=norm_wave.unsqueeze(0)).squeeze() for encoder in encoders]
        if len(spk_embs) == 1:
            return spk_embs[0]
        else:
            return torch.cat(spk_embs, dim=0)

    def _normalize_wave(self, wave, sr):
        # adapted from IMSToucan/Preprocessing/AudioPreprocessor
        dur = wave.shape[1] / sr
        wave = wave.squeeze().cpu().numpy()

        # normalize loudness
        try:
            meter = pyln.Meter(sr, block_size=min(dur-0.0001, abs(dur - 0.1)) if dur < 0.4 else 0.4)
            loudness = meter.integrated_loudness(wave)
            loud_normed = pyln.normalize.loudness(wave, loudness, -30.0)
            peak = np.amax(np.abs(loud_normed))
            norm_wave = np.divide(loud_normed, peak)
        except ZeroDivisionError:
            norm_wave = wave

        wave = torch.Tensor(norm_wave).to(self.device)

        if sr != 16000:
            wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)(wave)

        return wave.cpu()

    def _extract_style_embeddings_from_audio(self, data_dir: Path, model_path: Path):
        encoder = StyleEmbedding()
        check_dict = torch.load(model_path, map_location='cpu')
        encoder.load_state_dict(check_dict['style_emb_func'])
        encoder.to(self.device)

        audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True, device=self.device)

        recordings = read_kaldi_format(data_dir / 'wav.scp')
        utt2spk = read_kaldi_format(data_dir / 'utt2spk')
        spk2gender = read_kaldi_format(data_dir / 'spk2gender')

        spk2utt_ids = defaultdict(list)
        vectors = []

        i = 0
        for rec_name, rec_file in recordings.items():
            if self.emb_level == 'utt':
                speaker = rec_name
            else:  # speaker-level anonymization
                speaker = utt2spk[rec_name]
            self.utt2spk[rec_name] = utt2spk[rec_name]
            signal, fs = torchaudio.load(rec_file)
            if fs != audio_preprocessor.sr:
                audio_preprocessor = AudioPreprocessor(input_sr=fs, output_sr=16000, cut_silence=True,
                                                       device=self.device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                norm_wave = self._normalize_wave(wave=signal, sr=fs).to(self.device)
                norm_wave = audio_preprocessor.cut_silence_from_audio(norm_wave).cpu()
                spec = audio_preprocessor.logmelfilterbank(norm_wave, 16000).transpose(0, 1)
                spec_len = torch.LongTensor([len(spec)])
                vector = encoder(spec.unsqueeze(0).to(self.device), spec_len.unsqueeze(0).to(self.device)).squeeze().detach()

            spk2utt_ids[speaker].append(i)
            vectors.append(vector)

            i += 1

        if self.emb_level == 'utt':
            self.speakers = {speaker: id_list[0] for speaker, id_list in spk2utt_ids.items()}
            self.speaker_vectors = torch.stack(vectors, dim=0).to(self.device)
            spk2gender = {utt: spk2gender[speaker] for utt, speaker in utt2spk.items()}
        else:
            self.speakers, self.speaker_vectors = self._get_speaker_level_vectors(spk2utt_ids, torch.stack(
                vectors, dim=0).to(self.device))
        self.genders = [spk2gender[speaker] for speaker in self.speakers]
        self.idx2speakers = {idx: spk for spk, idx in self.speakers.items()}
        
