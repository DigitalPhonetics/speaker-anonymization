from pathlib import Path
import torch

from .speaker_embeddings import SpeakerEmbeddings


class BaseAnonymizer:

    def __init__(self, vec_type='xvector', device=None, emb_level='spk', **kwargs):
        # Base class for speaker embedding anonymization.
        self.vec_type = vec_type
        self.emb_level = emb_level
        self.embed_model_path = Path('pretrained_models')

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, int):
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def load_parameters(self, model_dir: Path):
        # Template method for loading parameters special to the anonymization method. Not implemented.
        raise NotImplementedError('load_parameters')

    def save_parameters(self, model_dir: Path):
        # Template method for saving parameters special to the anonymization method. Not implemented.
        raise NotImplementedError('save_parameters')

    def load_embeddings(self, emb_dir: Path):
        # Load previously extracted or generated speaker embeddings from disk.
        embeddings = SpeakerEmbeddings(self.vec_type, device=self.device, emb_level=self.emb_level)
        embeddings.load_vectors(emb_dir)
        return embeddings

    def save_embeddings(self, embeddings, emb_dir):
        # Save speaker embeddings to disk.
        embeddings.save_vectors(emb_dir)

    def anonymize_data(self, data_dir: Path, vector_dir: Path, emb_level='spk'):
        # Template method for anonymizing a dataset. Not implemented.
        raise NotImplementedError('anonymize_data')

    def _get_speaker_embeddings(self, data_dir: Path, vector_dir: Path, emb_level='spk'):
        # Retrieve original speaker embeddings, either by extracting or loading them.
        vectors = SpeakerEmbeddings(vec_type=self.vec_type, emb_level=emb_level, device=self.device)
        if vector_dir.exists():
            vectors.load_vectors(in_dir=vector_dir)
        else:
            vectors.extract_vectors_from_audio(data_dir=data_dir)
            vectors.save_vectors(out_dir=vector_dir)
        return vectors
