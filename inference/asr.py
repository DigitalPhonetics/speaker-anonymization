from tqdm import tqdm
from espnet2.bin.asr_inference import Speech2Text
import soundfile
import resampy
from espnet_model_zoo.downloader import ModelDownloader

from utils import create_clean_dir, read_kaldi_format, save_kaldi_format


class InferenceASR:

    def __init__(self, model_name, results_dir, data_dir, model_dir, device, force_compute=False):
        self.force_compute = force_compute
        self.results_dir = results_dir / 'transcription' / model_name
        self.data_dir = data_dir

        model_dir = model_dir / 'asr' / model_name

        d = ModelDownloader()

        self.speech2text = Speech2Text(
            **d.download_and_unpack(str(model_dir)),
            device=str(device),
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=0.4,
            beam_size=15,
            batch_size=1,
            nbest=1
        )

    def recognize_speech(self, dataset, utterance_list=None):
        dataset_results_dir = self.results_dir / dataset
        utt2spk = read_kaldi_format(self.data_dir / dataset / 'utt2spk')
        new = False

        if (dataset_results_dir / 'text').exists() and not self.force_compute:
            # if the text created from this ASR model already exists for this dataset and a computation is not
            # forced, simply load the text
            print('No speech recognition necessary; load existing text instead...')
            texts = {}
            with open(dataset_results_dir / 'text', 'r') as f:
                for line in f:
                    splitted_line = line.strip().split(' ')
                    texts[splitted_line[0].strip()] = ' '.join(splitted_line[1:])
        else:
            # otherwise, recognize the speech
            print(f'Recognize speech of {len(utt2spk)} utterances...')
            new = True
            create_clean_dir(dataset_results_dir)
            texts = {}
            wav_scp = read_kaldi_format( self.data_dir / dataset / 'wav.scp')

            for utt, spk in tqdm(utt2spk.items()):
                if utterance_list and utt not in utterance_list:
                    continue
                if utt in wav_scp:
                    speech, rate = soundfile.read(wav_scp[utt])
                    speech = resampy.resample(speech, rate, 16000)
                    rate = 16000

                    nbests = self.speech2text(speech)
                    text, *_ = nbests[0]
                    texts[utt] = text

            if not utterance_list:
                save_kaldi_format(texts, dataset_results_dir / 'text')

        return texts, utt2spk, new
