import torch

from .init_wgan import create_wgan


class EmbeddingsGenerator:

    def __init__(self, gan_path, device):
        self.device = device
        self.gan_path = gan_path

        self.mean = None
        self.std = None
        self.wgan = None

        self._load_model(self.gan_path)

    def generate_embeddings(self, n=1000):
        return self.wgan.sample_generator(num_samples=n, nograd=True, return_intermediate=False)

    def _load_model(self, path):
        gan_checkpoint = torch.load(path, map_location="cpu")

        self.wgan = create_wgan(parameters=gan_checkpoint['model_parameters'], device=self.device)
        self.wgan.G.load_state_dict(gan_checkpoint['generator_state_dict'])
        self.wgan.D.load_state_dict(gan_checkpoint['critic_state_dict'])

        self.mean = gan_checkpoint["mean"]
        self.std = gan_checkpoint["std"]

