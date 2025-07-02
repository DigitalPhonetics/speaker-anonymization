import warnings
import torch
from torchaudio.transforms import Resample

from anonymization.modules.tts.IMSToucan.Modules.EmbeddingModel.StyleEmbedding import StyleEmbedding
from anonymization.modules.tts.IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor


class StyleEmbeddings:

    def __init__(self, model_path, device):
        self.device = device

        self.extractor = StyleEmbedding()
        check_dict = torch.load(model_path, map_location='cpu')
        self.extractor.load_state_dict(check_dict['style_emb_func'])
        self.extractor.to(self.device)

        self.sr = 16000
        self.audio_preprocessor = AudioPreprocessor(input_sr=self.sr, output_sr=self.sr, cut_silence=True,
                                                    device=self.device)

    def extract_vector(self, audio, sr):
        if sr != self.sr:
            resample = Resample(orig_freq=sr, new_freq=self.sr).to(self.device)
            audio = resample(torch.tensor(audio, device=self.device, dtype=torch.float32))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = self.audio_preprocessor.audio_to_mel_spec_tensor(audio, explicit_sampling_rate=self.sr).transpose(0, 1)
            spec_len = torch.LongTensor([len(spec)])
            vector = self.extractor(spec.unsqueeze(0).to(self.device), spec_len.unsqueeze(0).to(self.device))
        return vector.squeeze().detach()
