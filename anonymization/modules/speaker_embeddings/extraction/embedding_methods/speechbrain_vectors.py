from pathlib import Path
import numpy as np
import torch
from speechbrain.inference import EncoderClassifier


class SpeechBrainVectors:

    VEC_PATHS = {
        'xvector': 'spkrec-xvect-voxceleb',
        'ecapa': 'spkrec-ecapa-voxceleb'
    }

    def __init__(self, vec_type, device, model_path: Path = None):
        self.device = device

        if model_path is not None and model_path.exists():
            model_path = Path(model_path).absolute()
            if model_path.is_file():
                savedir = model_path.parent
            else:
                savedir = model_path
            self.extractor = EncoderClassifier.from_hparams(
                    source=str(model_path), 
                    savedir=str(savedir),
                    run_opts={'device': self.device}
                )
        else:
            if model_path is None:
                model_path = Path('')
            vec_path = self.VEC_PATHS[vec_type]
            # The following line downloads and loads the corresponding speaker embedding model from huggingface and store
            # it in the corresponding savedir. If a model has been previously downloaded and stored already,
            # it is loaded from savedir instead of downloading it again.
            self.extractor = EncoderClassifier.from_hparams(source=f'speechbrain/{vec_path}',
                                                            savedir=Path(model_path, vec_path),
                                                            run_opts={'device': self.device})

    def extract_vector(self, audio, sr):
        audio = torch.tensor(np.trim_zeros(audio.cpu().numpy()))
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        return self.extractor.encode_batch(wavs=audio).squeeze()
