from pathlib import Path
import shutil
import torch
from torch.utils.data import DataLoader
import pandas as pd

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from .speechbrain_asr import InferenceSpeechBrainASR
from anonymization.modules import SpeechRecognition
from .speechbrain_asr.inference import MyDataset
from .asr_metrics import ASRMetrics
from utils import read_kaldi_format, save_yaml, setup_logger

logger = setup_logger(__name__)


def evaluate_asr(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device, backend, anon=True, n=None,
                 **kwargs):
    if backend == 'speechbrain':
        return asr_eval_speechbrain(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                                    model_path=model_path, anon_data_suffix=anon_data_suffix, device=device)
    elif backend in ('whisper', 'wav2vec2', 'mms'):
        return asr_eval_hf(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                           model_path=model_path, anon_data_suffix=anon_data_suffix, device=device, anon=anon, n=n)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: speechbrain, whisper, wav2vec2, mms.')


def asr_eval_speechbrain(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device):
    print(f'Use ASR model for evaluation: {model_path}')
    model = InferenceSpeechBrainASR(model_path=model_path, device=device)
    results_dir = params['results_dir']
    test_sets = eval_datasets + [f'{asr_dataset}_{anon_data_suffix}' for asr_dataset in eval_datasets]
    results = []


    with torch.no_grad():
        for test_set in test_sets:
            data_path = eval_data_dir / test_set
            if (results_dir / test_set / 'wer').exists() and (results_dir / test_set / 'text').exists():
                logger.info("No WER computation  necessary; print exsiting WER results")
                references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                hypotheses = read_kaldi_format(Path(results_dir, test_set, 'text'), values_as_string=True)
                scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=Path(results_dir,test_set, 'wer'))
            else:
                dataset = MyDataset(wav_scp_file=Path(data_path, 'wav.scp'), asr_model=model.asr_model)
                dataloader = DataLoader(dataset, batch_size=params['eval_batchsize'], shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
                hypotheses = model.transcribe_audios(data=dataloader, out_file=Path(results_dir, test_set, 'text'))
                references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=Path(results_dir,
                                                                                     test_set, 'wer'))
            wer = scores.summarize("error_rate")
            test_set_info = test_set.split('_')
            results.append({'dataset': test_set_info[0], 'split': test_set_info[1],
                            'asr': 'anon' if 'anon' in test_set else 'original', 'WER': round(wer, 3)})
            print(f'{test_set} - WER: {wer}')
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv(results_dir / 'results.csv')
        save_yaml(params, results_dir / 'config.yaml')
        return results_df


def asr_eval_hf(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device, anon=True, n=None):
    print(f'Use ASR model for evaluation: {model_path}')
    results_dir = Path(params['results_dir'])
    model = SpeechRecognition(devices=[device], save_intermediate=True, settings=params, force_compute=True,
                              results_dir=results_dir)
    if anon:
        test_sets = eval_datasets + [f'{asr_dataset}_{anon_data_suffix}' for asr_dataset in eval_datasets]
    else:
        test_sets = eval_datasets
    results = []

    languages = params.get('dataset_languages', ['en', 'zh'])
    if 'zh' in languages:
        asr_metrics_wer = ASRMetrics(languages=languages, convert_to_pinyin=True, convert_to_phones=False, device=device)
    else:
        asr_metrics_wer = ASRMetrics(languages=languages, convert_to_pinyin=False, convert_to_phones=False, device=device)

    asr_metrics_per = ASRMetrics(languages=languages, convert_to_pinyin=False, convert_to_phones=True, device=device)

    with torch.no_grad():
        for test_set in test_sets:
            print(test_set)
            data_path = eval_data_dir / test_set
            test_results_path =  Path(results_dir, test_set)
            test_results_path.mkdir(exist_ok=True, parents=True)
            # copy gold text for future reference
            shutil.copy(Path(data_path, 'text'), Path(test_results_path, 'gold_text'))
            if (results_dir / test_set / 'text').exists():
                logger.info("No speech recognition  necessary; print WER based on existing transcripts")
                references = read_kaldi_format(Path(test_results_path, 'gold_text'), values_as_string=True)
                hypotheses = read_kaldi_format(Path(test_results_path, 'text'), values_as_string=True)
            else:
                print('Recognize speech')
                hypotheses = model.recognize_speech(dataset_path=data_path, dataset_name=test_set, n=n)
                references = read_kaldi_format(Path(test_results_path, 'gold_text'), values_as_string=True)

            wer_scores = asr_metrics_wer.compute_wer(ref_texts=references, hyp_texts=hypotheses,
                                                     out_file=Path(test_results_path, 'wer'))
            wer = round(wer_scores.summarize("error_rate"), 3)

            per_scores = asr_metrics_per.compute_wer(ref_texts=references, hyp_texts=hypotheses,
                                                     out_file=Path(test_results_path, 'per'))
            per = round(per_scores.summarize('error_rate'), 3)


            test_set_info = test_set.split('_')
            if len(test_set_info) > 1:
                dataset_name = test_set_info[0]
                split = test_set_info[1]
            else:
                dataset_name, split = test_set_info[0], test_set_info[0]

            results.append({'dataset': dataset_name, 'split': split,
                            'asr': 'anon' if 'anon' in test_set else 'original',
                            'WER': wer, 'PER': per})
            print(f'{test_set} - WER: {wer}')

        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv(results_dir / 'results.csv')
        save_yaml(params, results_dir / 'config.yaml')
        return results_df
