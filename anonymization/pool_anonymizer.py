from pathlib import Path
import numpy as np
import torch
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

from .base_anonymizer import BaseAnonymizer
from .plda_model import PLDAModel
from .speaker_embeddings import SpeakerEmbeddings
from utils import create_clean_dir

REVERSED_GENDERS = {'m': 'f', 'f': 'm'}


class PoolAnonymizer(BaseAnonymizer):

    def __init__(self, pool_data_dir=Path('libritts_train_other_500'), vec_type='xvector', N=200, N_star=100,
                 distance='plda', cross_gender=False, proximity='farthest', device=None, model_name=None, **kwargs):
        # Pool anonymization method based on the primary baseline of the Voice Privacy Challenge 2020.
        # Given a speaker vector, the N most distant vectors in an external speaker pool are extracted,
        # and an average of a random subset of N_star vectors is computed and taken as new speaker vector.
        # Default distance measure is PLDA.
        super().__init__(vec_type=vec_type, device=device)

        self.model_name = model_name if model_name else f'pool_{vec_type}'

        self.pool_data_dir = pool_data_dir  # data for external speaker pool
        self.N = N  # number of most distant vectors to consider
        self.N_star = N_star  # number of vectors to include in averaged vector
        self.distance = distance  # distance measure, either 'plda' or 'cosine'
        self.proximity = proximity  # proximity method, either 'farthest' (distant vectors), 'nearest', or 'closest'
        self.cross_gender = cross_gender  # Whether to reverse the genders of the speakers

        self.pool_embeddings = None
        self.pool_genders = {}
        self.plda = None

    def load_parameters(self, model_dir: Path):
        self._load_settings(model_dir / 'settings.json')
        self.pool_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='spk', device=self.device)
        self.pool_embeddings.load_vectors(model_dir / 'pool_embeddings')
        self.pool_genders = {gender: [i for i, spk_gender in enumerate(self.pool_embeddings.genders)
                                      if spk_gender == gender] for gender in set(self.pool_embeddings.genders)}
        if self.distance == 'plda':
            self.plda = PLDAModel(train_embeddings=self.pool_embeddings, results_path=model_dir)

    def save_parameters(self, model_dir: Path):
        create_clean_dir(model_dir)
        self.pool_embeddings.save_vectors(model_dir / 'pool_embeddings')
        self._save_settings(model_dir / 'settings.json')
        if self.plda:
            self.plda.save_parameters(model_dir)

    def anonymize_data(self, data_dir: Path, vector_dir: Path, emb_level='spk'):
        print('Load original speaker embeddings...')
        speaker_embeddings = self._get_speaker_embeddings(data_dir, vector_dir / f'{emb_level}_level_{self.vec_type}',
                                                          emb_level=emb_level)
        if not self.pool_embeddings:
            print('Compute speaker embeddings for speaker pool...')
            self.pool_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='spk', device=self.device)
            self.pool_embeddings.extract_vectors_from_audio(self.pool_data_dir, model_path=self.embed_model_path)
            self.pool_genders = {gender: [i for i, spk_gender in enumerate(self.pool_embeddings.genders)
                                          if spk_gender == gender] for gender in set(self.pool_embeddings.genders)}
        if self.distance == 'plda' and not self.plda:
            print('Train PLDA model...')
            self.plda = PLDAModel(train_embeddings=self.pool_embeddings)

        print('pool embeddings', self.pool_embeddings.speaker_vectors.shape)
        print('speaker embeddings', speaker_embeddings.speaker_vectors.shape)
        distance_matrix = self._compute_distances(vectors_a=self.pool_embeddings.speaker_vectors,
                                                  vectors_b=speaker_embeddings.speaker_vectors)

        print(f'Anonymize embeddings of {len(speaker_embeddings)} speakers...')
        speakers = []
        anon_vectors = []
        genders = []
        for i in tqdm(range(len(speaker_embeddings))):
            speaker, _ = speaker_embeddings[i]
            gender = speaker_embeddings.genders[i]
            distances_to_speaker = distance_matrix[:, i]
            candidates = self._get_pool_candidates(distances_to_speaker, gender)
            selected_anon_pool = np.random.choice(candidates, self.N_star, replace=False)
            anon_vec = torch.mean(self.pool_embeddings.speaker_vectors[selected_anon_pool], dim=0)
            speakers.append(speaker)
            anon_vectors.append(anon_vec)
            genders.append(gender if not self.cross_gender else REVERSED_GENDERS[gender])

        anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, device=self.device)
        anon_embeddings.set_vectors(speakers=speakers, vectors=torch.stack(anon_vectors, dim=0), genders=genders,
                                    utt2spk=speaker_embeddings.utt2spk)

        return anon_embeddings

    def _compute_distances(self, vectors_a, vectors_b):
        if self.distance == 'plda':
            return 1 - self.plda.compute_distance(enrollment_vectors=vectors_a, trial_vectors=vectors_b)
        elif self.distance == 'cosine':
            return cosine_distances(X=vectors_a.cpu(), Y=vectors_b.cpu())
        else:
            return []

    def _get_pool_candidates(self, distances, gender):
        if self.cross_gender is True:
            distances = distances[self.pool_genders[REVERSED_GENDERS[gender]]]
        else:
            distances = distances[self.pool_genders[gender]]

        if self.proximity == 'farthest':
            return np.argpartition(distances, -self.N)[-self.N:]
        elif self.proximity == 'nearest':
            return np.argpartition(distances, self.N)[:self.N]
        elif self.proximity == 'center':
            sorted_distances = np.sort(distances)
            return sorted_distances[len(sorted_distances)//2:(len(sorted_distances)//2)+self.N]

    def _save_settings(self, filename):
        settings = {
            'vec_type': self.vec_type,
            'N': self.N,
            'N*': self.N_star,
            'distance': self.distance,
            'proximity': self.proximity,
            'cross_gender': self.cross_gender
        }
        with open(filename, 'w') as f:
            json.dump(settings, f)

    def _load_settings(self, filename):
        with open(filename, 'r') as f:
            settings = json.load(f)

        self.N = settings['N'] if 'N' in settings else self.N
        self.N_star = settings['N*'] if 'N*' in settings else self.N_star
        self.distance = settings['distance'] if 'distance' in settings else self.distance
        self.proximity = settings['proximity'] if 'proximity' in settings else self.proximity
        self.cross_gender = settings['cross_gender'] if 'cross_gender' in settings else self.cross_gender
        self.vec_type = settings['vec_type'] if 'vec_type' in settings else self.vec_type



# for every source x-vector, an anonymized x-vector is computed by finding the N farthest x-
# vectors in an external pool (LibriTTS train-other-500) accord-
# ing to the PLDA distance, and by averaging N ∗ randomly se-
# lected vectors among them. In the baseline, we use N = 200 and N ∗ = 100
