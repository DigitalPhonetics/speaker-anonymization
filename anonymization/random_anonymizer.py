import json
from pathlib import Path
import torch
import numpy as np

from .speaker_embeddings import SpeakerEmbeddings
from .base_anonymizer import BaseAnonymizer
from utils import create_clean_dir


class RandomAnonymizer(BaseAnonymizer):

    def __init__(self, vec_type='xvector', device=None, model_name=None, in_scale=False,  **kwargs):
        super().__init__(vec_type=vec_type, device=device)
        self.model_name = model_name if model_name else f'random_{vec_type}'

        self.in_scale = in_scale
        self.dim_ranges = None

    def load_parameters(self, model_dir):
        with open(model_dir / 'settings.json') as f:
            settings = json.load(f)
        self.vec_type = settings['vec_type'] if 'vec_type' in settings else self.vec_type
        self.in_scale = settings.get('in_scale', self.in_scale)

        if self.in_scale:
            with open(model_dir / 'stats_per_dim.json') as f:
                dim_ranges = json.load(f)
                self.dim_ranges = [(v['min'], v['max']) for k, v in sorted(dim_ranges.items(), key=lambda x: int(x[0]))]

    def save_parameters(self, model_dir):
        create_clean_dir(model_dir)
        settings = {
            'vec_type': self.vec_type
        }
        with open(model_dir / 'settings.json', 'w') as f:
            json.dump(settings, f)

    def anonymize_data(self, data_dir: Path, vector_dir: Path, emb_level='spk'):
        speaker_embeddings = self._get_speaker_embeddings(data_dir, vector_dir / f'{emb_level}_level_{self.vec_type}',
                                                          emb_level=emb_level)

        if self.dim_ranges:
            print('Anonymize vectors in scale!')
            return self._anonymize_data_in_scale(speaker_embeddings)
        else:
            speakers = []
            anon_vectors = []
            genders = speaker_embeddings.genders
            for speaker, vector in speaker_embeddings:
                mask = torch.zeros(vector.shape[0]).float().random_(-40, 40).to(self.device)
                anon_vec = vector * mask
                speakers.append(speaker)
                anon_vectors.append(anon_vec)

            anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, device=self.device)
            anon_embeddings.set_vectors(speakers=speakers, vectors=torch.stack(anon_vectors, dim=0), genders=genders,
                                        utt2spk=speaker_embeddings.utt2spk)

            return anon_embeddings

    def _anonymize_data_in_scale(self, speaker_embeddings):
        speakers = []
        anon_vectors = []
        genders = speaker_embeddings.genders

        for speaker, vector in speaker_embeddings:
            anon_vec = torch.tensor([np.random.uniform(*dim_range) for dim_range in self.dim_ranges]).to(self.device)
            speakers.append(speaker)
            anon_vectors.append(anon_vec)

        anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, device=self.device)
        anon_embeddings.set_vectors(speakers=speakers, vectors=torch.stack(anon_vectors, dim=0), genders=genders,
                                    utt2spk=speaker_embeddings.utt2spk)

        return anon_embeddings
