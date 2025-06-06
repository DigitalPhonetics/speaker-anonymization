from pathlib import Path
import torch
import numpy as np
from scipy.spatial.distance import cosine
from os import PathLike
from tqdm import tqdm
from typing import Union

from .base_anon import BaseAnonymizer
from ..speaker_embeddings import SpeakerEmbeddings
from .utils.WGAN import EmbeddingsGenerator
from utils import setup_logger

logger = setup_logger(__name__)

class GANAnonymizer(BaseAnonymizer):
    """
        Implementation of the anonymizer proposed in the paper "Anonymizing
        speech with generative adversarial networks to preserve speaker
        privacy" (https://arxiv.org/pdf/2210.07002.pdf).
    """
    def __init__(
        self,
        vec_type: str = "xvector",
        device: Union[str, torch.device, int] = "cuda:0",  # TODO
        model_name: Union[str, PathLike] = None,
        vectors_file: Union[str, PathLike] = None,
        sim_threshold: float = 0.7,
        gan_model_path: Union[str, PathLike] = None,
        num_sampled: int = 1000,
        save_intermediate: bool = False,
        suffix: str = '_anon',
        **kwargs,
    ):
        """
        Args:
            vec_type: The type of the speaker embedding to anonymize. Valid
                values are 'xvector', 'style-embed', 'ecapa'
            device: The computation device to use for the anonymization.
            model_name: The filename of the model used for the anonymization.
                Defaults to 'gan_{vec_type}'.
            vectors_file: The path to the file containing the GAN vectors.
                Defaults to 'gan_vectors_{vec_type}.pt'.
            sim_threshold: The minimum cosine similarity between the original
                speaker embedding and the anonymized embedding.
            gan_model_path: The path to the GAN model.
            num_sampled: The number of GAN vectors to sample.
            save_intermediate: If True, the GAN vectors and the unused indices
                will be saved to files.
            suffix: The suffix to append to the output files.
        """
        super().__init__(vec_type=vec_type, device=device, suffix=suffix)

        self.model_name = model_name if model_name else f"gan_{vec_type}"
        self.vectors_file = Path(vectors_file)
        self.unused_indices_file = self.vectors_file.with_name(
            f"unused_indices_{self.vectors_file.name}"
        )
        self.sim_threshold = sim_threshold
        self.save_intermediate = save_intermediate
        self.n = num_sampled

        if self.vectors_file.is_file():
            self.gan_vectors = torch.load(self.vectors_file, map_location=self.device)
            #print(self.gan_vectors)
            logger.info(f'Gan vectors: {self.gan_vectors.shape}')
            if self.unused_indices_file.is_file():
                self.unused_indices = torch.load(
                    self.unused_indices_file, map_location="cpu"
                )
                logger.info(f'Unused indices: {self.unused_indices.shape}')
            else:
                self.unused_indices = np.arange(len(self.gan_vectors))
        else:
            (
                self.gan_vectors,
                self.unused_indices,
            ) = self._generate_artificial_embeddings(gan_model_path, self.n)

    def anonymize_embeddings(
        self, speaker_embeddings: torch.Tensor, emb_level: str = "spk"
    ):
        """
            Anonymize speaker embeddings using the GAN model.
        Args:
            speaker_embeddings: [n_embeddings, n_channels] Speaker
                embeddings to be anonymized.
            emb_level: Embedding level ('spk' for speaker level
                or 'utt' for utterance level).
        """
        if emb_level == "spk":
            logger.info(f"Anonymize embeddings of {len(speaker_embeddings)} speakers...")
        elif emb_level == "utt":
            logger.info(f"Anonymize embeddings of {len(speaker_embeddings)} utterances...")

        identifiers = []
        speakers = []
        anon_vectors = []
        genders = []
        for i in tqdm(range(len(speaker_embeddings))):
            identifier, vector = speaker_embeddings[i]
            speaker = speaker_embeddings.original_speakers[i]
            gender = speaker_embeddings.genders[i]
            anon_vec = self._select_gan_vector(spk_vec=vector)
            identifiers.append(identifier)
            speakers.append(speaker)
            anon_vectors.append(anon_vec)
            genders.append(gender)

        anon_embeddings = SpeakerEmbeddings(
            vec_type=self.vec_type, device=self.device, emb_level=emb_level
        )
        anon_embeddings.set_vectors(
            identifiers=identifiers,
            vectors=torch.stack(anon_vectors, dim=0),
            speakers=speakers,
            genders=genders,
        )
        if self.save_intermediate:
            torch.save(self.unused_indices, self.unused_indices_file)

        return anon_embeddings

    def _generate_artificial_embeddings(self, gan_model_path: Path, n: int):
        logger.info(f"Generate {n} artificial speaker embeddings...")
        generator = EmbeddingsGenerator(gan_path=gan_model_path, device=self.device)
        gan_vectors = generator.generate_embeddings(n=n)
        unused_indices = np.arange(len(gan_vectors))

        if self.save_intermediate:
            torch.save(gan_vectors, self.vectors_file)
            torch.save(unused_indices, self.unused_indices_file)
        return gan_vectors, unused_indices

    def _select_gan_vector(self, spk_vec: torch.Tensor):
        i = 0
        limit = 20
        while i < limit:
            if len(self.unused_indices) == 0:
                self.unused_indices = np.arange(len(self.gan_vectors))  # reset indices
            idx = np.random.choice(self.unused_indices)
            anon_vec = self.gan_vectors[idx]
            sim = 1 - cosine(spk_vec.cpu().numpy(), anon_vec.cpu().numpy())
            if sim < self.sim_threshold:
                break
            i += 1
        self.unused_indices = self.unused_indices[self.unused_indices != idx]
        return anon_vec
