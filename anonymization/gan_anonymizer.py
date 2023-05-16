from pathlib import Path
import torch
import numpy as np
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm

from .base_anonymizer import BaseAnonymizer
from .speaker_embeddings import SpeakerEmbeddings
from WGAN.embeddings_generator import EmbeddingsGenerator
from utils import create_clean_dir


class GANAnonymizer(BaseAnonymizer):

    def __init__(self, vec_type='xvector', device=None, model_name=None, vectors_file=None, sim_threshold=0.7,
                 **kwargs):
        super().__init__(vec_type=vec_type, device=device)

        self.model_name = model_name if model_name else f'gan_{vec_type}'
        self.vectors_file = vectors_file
        self.sim_threshold = sim_threshold

        self.gan_vectors = None
        self.unused_indices = None

    def load_parameters(self, model_dir: Path):
        with open(model_dir / 'settings.json') as f:
            settings = json.load(f)
        self.vec_type = settings.get('vec_type', self.vec_type)
        self.vectors_file = settings.get('vectors_file', self.vectors_file)
        self.embed_model_path = settings.get('embed_model_path', Path('pretrained_models'))

        if (model_dir / self.vectors_file).is_file():
            self.gan_vectors = torch.load(model_dir / self.vectors_file, map_location=self.device)
            self.unused_indices = torch.load(model_dir / f'unused_indices_{self.vectors_file}', map_location='cpu')
        else:
            self.gan_model_name = settings.get('gan_model_name', 'gan.pt')
            self.n = settings.get('num_sampled', 1000)
            self._generate_artificial_embeddings(model_dir, self.gan_model_name, self.n)

    def save_parameters(self, model_dir: Path):
        create_clean_dir(model_dir)
        settings = {
            'vec_type': self.vec_type,
            'vectors_file': self.vectors_file,
            'embed_model_path': self.embed_model_path,
            'gan_model_name': self.gan_model_name,
            'num_sampled': self.n
        }
        with open(model_dir / 'settings.json', 'w') as f:
            json.dump(settings, f)

        torch.save(self.unused_indices, model_dir / f'unused_indices_{self.vectors_file}')

    def anonymize_data(self, data_dir: Path, vector_dir: Path, emb_level='spk'):
        print('Load original speaker embeddings...')
        speaker_embeddings = self._get_speaker_embeddings(data_dir, vector_dir / f'{emb_level}_level_{self.vec_type}',
                                                          emb_level=emb_level)

        if emb_level == 'spk':
            print(f'Anonymize embeddings of {len(speaker_embeddings)} speakers...')
        elif emb_level == 'utt':
            print(f'Anonymize embeddings of {len(speaker_embeddings)} utterances...')

        speakers = []
        anon_vectors = []
        genders = []
        for i in tqdm(range(len(speaker_embeddings))):
            speaker, vector = speaker_embeddings[i]
            gender = speaker_embeddings.genders[i]
            anon_vec = self._select_gan_vector(spk_vec=vector)
            speakers.append(speaker)
            anon_vectors.append(anon_vec)
            genders.append(gender)

        anon_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, device=self.device, emb_level=emb_level)
        anon_embeddings.set_vectors(speakers=speakers, vectors=torch.stack(anon_vectors, dim=0), genders=genders,
                                    utt2spk=speaker_embeddings.utt2spk)

        return anon_embeddings

    def _generate_artificial_embeddings(self, model_dir, gan_model_name, n):
        print(f'Generate {n} artificial speaker embeddings...')
        generator = EmbeddingsGenerator(gan_path=model_dir / gan_model_name, device=self.device)
        self.gan_vectors = generator.generate_embeddings(n=n)
        self.unused_indices = np.arange(len(self.gan_vectors))

        torch.save(self.gan_vectors, model_dir / self.vectors_file)
        torch.save(self.unused_indices, model_dir / f'unused_indices_{self.vectors_file}')

    def _select_gan_vector(self, spk_vec):
        i = 0
        limit = 20
        while i < limit:
            idx = np.random.choice(self.unused_indices)
            anon_vec = self.gan_vectors[idx]
            sim = 1 - cosine(spk_vec.cpu().numpy(), anon_vec.cpu().numpy())
            if sim < self.sim_threshold:
                break
            i += 1
        self.unused_indices = self.unused_indices[self.unused_indices != idx]
        return anon_vec