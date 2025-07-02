from utils import save_kaldi_format
from copy import deepcopy

from speechbrain.inference import EncoderASR, EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
import torch
import torchaudio

from utils import read_kaldi_format


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, wav_scp_file, asr_model):
        data = []
        for utt_id, wav_file in read_kaldi_format(wav_scp_file).items():
            signal, sr = torchaudio.load(str(wav_file))
            wav = asr_model.audio_normalizer(signal.squeeze(), sr)
            #wav = asr_model.load_audio(wav_file)
            wav_len = len(wav.squeeze())
            data.append((utt_id, wav, wav_len))

        # Sort the data based on audio length
        self.data = sorted(data, key=lambda x: x[2], reverse=True)

    def __getitem__(self, idx):
        wavname, wav, wav_len = self.data[idx]
        return wav, wavname, wav_len

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):  ## make them all the same length with zero padding
        wavs, wavnames, wav_lens = zip(*batch)
        wavs = list(wavs)
        batch_wav = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
        lens = torch.Tensor(wav_lens) / batch_wav.shape[1]

        return wavnames, batch_wav, lens


class InferenceSpeechBrainASR:

    def __init__(self, model_path, device):
        self.device = device
        print(f'Use ASR model for evaluation: {model_path}')
        self.asr_model = EncoderDecoderASR.from_hparams(source=model_path,
                                                        hparams_file='transformer_inference.yaml',
                                                        savedir=model_path, run_opts={'device': self.device})

    def plain_text_key(self, path):
        tokens = []  # key: token_list
        for token in path:
            tokens.append(token.strip().split(' '))
        return tokens

    def transcribe_audios_old(self, audio_paths, out_file):
        texts = {}

        for utt_id, wav_file in audio_paths.items():
            text = self.asr_model.transcribe_file(wav_file)
            texts[utt_id] = text

        out_file.parent.mkdir(exist_ok=True, parents=True)
        save_kaldi_format(texts, out_file)
        return texts

    def transcribe_audios(self, data, out_file):
        texts = {}
        for batch in data:
            filenames, inputs, lengths = batch
            inputs = inputs.to(self.device)
            lengths = lengths.to(self.device)
            predicts, _ = self.asr_model.transcribe_batch(inputs, lengths)
            for i, utt_id in enumerate(filenames):
                texts[deepcopy(utt_id)] = str(predicts[i])

        out_file.parent.mkdir(exist_ok=True, parents=True)
        save_kaldi_format(texts, out_file)
        return texts

    def compute_wer(self, ref_texts, hyp_texts, out_file):
        wer_stats = ErrorRateStats()

        ids = []
        predicted = []
        targets = []
        for utt_id, ref in ref_texts.items():
            ids.append(utt_id)
            targets.append(ref)
            predicted.append(hyp_texts[utt_id])

        wer_stats.append(ids=ids, predict=self.plain_text_key(predicted), target=self.plain_text_key(targets))
        out_file.parent.mkdir(exist_ok=True, parents=True)

        with open(out_file, 'w') as f:
            wer_stats.write_stats(f)

        return wer_stats





