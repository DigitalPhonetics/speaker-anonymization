from tqdm import tqdm
from espnet2.bin.asr_inference import Speech2Text
import soundfile
import resampy
import torch
from espnet_model_zoo.downloader import ModelDownloader

from utils import read_kaldi_format, save_kaldi_format


class InferenceASR:

    def __init__(self, model_name, results_dir, data_dir, model_dir, device):
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.device = device

        model_dir = model_dir / 'asr' / model_name

        d = ModelDownloader()

        self.speech2text = Speech2Text(
            **d.download_and_unpack(str(model_dir)),
            device=str(device),
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=0.2,
            beam_size=15,
            batch_size=1,
            nbest=1
        )

    def recognize_speech(self, dataset, utterance_list=None):
        dataset_results_dir = self.results_dir / dataset
        utt2spk = read_kaldi_format(self.data_dir / dataset / 'utt2spk')
        texts = {}
        new = False

        if (dataset_results_dir / 'text').exists():
            # if the text created from this ASR model already exists for this dataset and a computation is not
            # forced, simply load the text
            with open(dataset_results_dir / f'text', 'r', encoding='utf8') as f:
                for line in f:
                    splitted_line = line.strip().split(' ')
                    texts[splitted_line[0].strip()] = ' '.join(splitted_line[1:])

        if len(texts) == len(utt2spk):
            print('No speech recognition necessary; load existing text instead...')
        else:
            if texts:
                print(f'No speech recognition necessary for {len(texts)} of {len(utt2spk)} utterances')
            # otherwise, recognize the speech
            print(f'Recognize speech of {len(utt2spk)} utterances...')
            new = True
            dataset_results_dir.mkdir(exist_ok=True)
            wav_scp = read_kaldi_format(self.data_dir / dataset / 'wav.scp')

            i = 0
            for utt, spk in tqdm(utt2spk.items()):
                if utt in texts:
                    continue
                if utterance_list and utt not in utterance_list:
                    continue
                if utt in wav_scp:
                    speech, rate = soundfile.read(wav_scp[utt])
                    speech = torch.tensor(resampy.resample(speech, rate, 16000)).to(torch.device('cpu'))
                    rate = 16000

                    nbests = self.speech2text(speech)
                    text, *_ = nbests[0]
                    texts[utt] = text
                i += 1
                if i % 1000 == 0 and not utterance_list:
                    save_kaldi_format(texts, dataset_results_dir / f'text')

            if not utterance_list:
                save_kaldi_format(texts, dataset_results_dir / 'text')

        return texts, utt2spk, new
