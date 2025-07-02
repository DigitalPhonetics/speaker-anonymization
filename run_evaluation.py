# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch so we will read all arguments directly on startup
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import multiprocessing
parser = ArgumentParser()
parser.add_argument('--config', default='config_eval.yaml')
parser.add_argument('--gpu_ids', default='0')
args = parser.parse_args()

from datetime import datetime

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

import torch
import time
import shutil
import itertools

from evaluation import evaluate_asv, train_asv_eval, evaluate_asr, train_asr_eval, evaluate_gvd
from utils import (parse_yaml, scan_checkpoint, combine_asr_data, split_vctk_into_common_and_diverse,
                   get_datasets, prepare_evaluation_data, get_anon_wav_scps, save_yaml, setup_logger)

logger = setup_logger(__name__)

def get_evaluation_steps(params):
    eval_steps = {}
    if 'privacy' in params:
        eval_steps['privacy'] = list(params['privacy'].keys())
    if 'utility' in params:
        eval_steps['utility'] = list(params['utility'].keys())

    if 'eval_steps' in params:
        param_eval_steps = params['eval_steps']

        for eval_part, eval_metrics in param_eval_steps.items():
            if eval_part not in eval_steps:
                raise KeyError(f'Unknown evaluation step {eval_part}, please specify in config')
            for metric in eval_metrics:
                if metric not in eval_steps[eval_part]:
                    raise KeyError(f'Unknown metric {metric}, please specify in config')
        return param_eval_steps
    return eval_steps


def get_eval_trial_datasets(datasets_list):
    eval_pairs = []

    for dataset in datasets_list:
        eval_pairs.extend([(f'{dataset["data"]}_{dataset["set"]}_{enroll}',
                            f'{dataset["data"]}_{dataset["set"]}_{trial}')
                           for enroll, trial in itertools.product(dataset['enrolls'], dataset['trials'])])
    return eval_pairs


def check_vctk_split(data_trials, eval_data_dir, anon_suffix):
    copy_files_for_orig = ['spk2utt', 'text', 'utt2spk', 'trials', 'spk2gender', 'wav.scp', 'utt2dur']
    copy_files_for_anon = ['spk2utt', 'text', 'utt2spk', 'trials']
    separated_data_trials = []

    for enroll, trial in data_trials:
        if 'vctk' in trial and '_all' in trial:
            common_split = trial.replace('all', 'common')  # same sentences for all speakers
            diverse_split = trial.replace('_all', '')  # different sentences for each speaker
            if (not Path(eval_data_dir, common_split).exists() or \
                    not Path(eval_data_dir, diverse_split).exists()):
                split_vctk_into_common_and_diverse(dataset=trial, output_path=eval_data_dir, orig_data_path=eval_data_dir,
                                                   copy_files=copy_files_for_orig, anon=False, anon_suffix=f'{anon_suffix}',
                                                   out_data_split=Path(eval_data_dir, trial))
            if not Path(eval_data_dir, f'{common_split}{anon_suffix}').exists() or \
                not Path(eval_data_dir, f'{diverse_split}{anon_suffix}').exists():
                split_vctk_into_common_and_diverse(dataset=trial, output_path=eval_data_dir, orig_data_path=eval_data_dir,
                                                   copy_files=copy_files_for_anon, anon=True, anon_suffix=f'{anon_suffix}',
                                                   out_data_split = Path(eval_data_dir, f'{trial}{anon_suffix}'))
            separated_data_trials.append((enroll, common_split))
            separated_data_trials.append((enroll, diverse_split))
        else:
            separated_data_trials.append((enroll, trial))
    return separated_data_trials

def check_anon_dir(datasets_list, eval_data_dir, anon_suffix):
    for dataset in datasets_list:
        if not os.path.exists(f'{eval_data_dir}/{dataset["data"]}_{dataset["set"]}_enrolls_{anon_suffix}'):
            print(f'Not exits:{eval_data_dir}/{dataset["data"]}_{dataset["set"]}_enrolls_{anon_suffix}, try to create')
            return False
        for trial in dataset["trials"]:
            if not os.path.exists(f'{eval_data_dir}/{dataset["data"]}_{dataset["set"]}_{trial}_{anon_suffix}'):
                return False

    return True

def get_eval_asr_datasets(datasets_list, eval_data_dir, anon_suffix):
    # combines the trial subsets (trial_f, trial_m) of each dataset into one asr dataset
    eval_data = set()

    # in case one bigger dataset is divided into smaller parts (e.g. vctk_all into vctk and vctk_common) we need to
    # collect all parts together to create one asr dataset for the whole instead of for each subpart
    collated_datasets = defaultdict(list)
    for dataset in datasets_list:
        asr_dataset_name = f'{dataset["data"]}_{dataset["set"]}_asr'
        collated_datasets[asr_dataset_name].append(dataset)

    for asr_dataset_name, datasets in collated_datasets.items():
        for asr_dataset in (asr_dataset_name, f'{asr_dataset_name}_{anon_suffix}'): # for orig and anon
            if not Path(eval_data_dir / asr_dataset).exists():
                trial_dirs = []
                for dataset in datasets:
                    for trial in dataset['trials']:
                        if anon_suffix in asr_dataset:
                            trial_dirs.append(Path(eval_data_dir / f'{dataset["data"]}_{dataset["set"]}_{trial}_{anon_suffix}'))
                        else:
                            trial_dirs.append(Path(eval_data_dir / f'{dataset["data"]}_{dataset["set"]}_{trial}'))
                output_dir = Path(eval_data_dir / asr_dataset)
                combine_asr_data(input_dirs=trial_dirs, output_dir=output_dir)
        eval_data.add(asr_dataset_name)

    return list(eval_data)

def save_result_summary(out_dir, results_dict, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, out_dir / 'config.yaml')

    with open(out_dir / 'results.txt', 'w') as f:
        f.write(f'---- Time: {datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")} ----\n')
        if 'asv' in results_dict:
            f.write('\n')
            f.write('---- ASV results ----\n')
            f.write(results_dict['asv'].to_string())
        if 'asr' in results_dict:
            f.write('\n')
            f.write('---- ASR results ----\n')
            f.write(results_dict['asr'].to_string())
        if 'gvd' in results_dict:
            f.write('\n')
            f.write('---- GVD results ----\n')
            f.write(results_dict['gvd'].to_string())


if __name__ == '__main__':
    multiprocessing.set_start_method("fork", force=True)

    params = parse_yaml(Path('configs', args.config))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    eval_data_dir = params['eval_data_dir']
    anon_suffix = params['anon_data_suffix']
    eval_steps = get_evaluation_steps(params)
    results = {}

    if "anon_data_dir" in params:
        logger.info("Preparing datadir according to the Kaldi format.")
        now = datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")
        datasets = get_datasets(params)
        anon_wav_scps = get_anon_wav_scps(params['anon_data_dir'])
        output_path = params['exp_dir'] / 'formatted_data' / now
        prepare_evaluation_data(
            dataset_dict=datasets,
            anon_wav_scps=anon_wav_scps,
            anon_vectors_path=params['data_dir'],
            anon_suffix='_' + anon_suffix,
            output_path=output_path,
        )
        eval_data_dir = output_path

    eval_data_trials = get_eval_trial_datasets(params['datasets'])
    eval_data_trials = check_vctk_split(eval_data_trials, eval_data_dir=eval_data_dir, anon_suffix='_'+anon_suffix)
    eval_data_asr = get_eval_asr_datasets(params['datasets'], eval_data_dir=eval_data_dir, anon_suffix=anon_suffix)

    # make sure given paths exist
    assert eval_data_dir.exists(), f'{eval_data_dir} does not exist'

    if 'privacy' in eval_steps:
        if 'asv' in eval_steps['privacy']:
            asv_params = params['privacy']['asv']
            model_dir = params['privacy']['asv']['model_dir']
            if 'training' in asv_params:
                asv_train_params = asv_params['training']
                inference_config = params['privacy']['asv']['training']['inference_config']
                if not model_dir.exists() or asv_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    logger.info('Perform ASV training')
                    train_asv_eval(train_params=asv_train_params, output_dir=asv_params['model_dir'])
                    logger.info("ASV training time: %f min ---" % (float(time.time() - start_time) / 60))
                model_dir = scan_checkpoint(model_dir, 'CKPT')
                shutil.copy(inference_config, params['privacy']['asv']['model_dir'] / 'hyperparams.yaml')
                shutil.copy(inference_config, model_dir / 'hyperparams.yaml')

            if 'evaluation' in asv_params:
                logger.info('Perform ASV evaluation')
                start_time = time.time()
                asv_results = evaluate_asv(eval_datasets=eval_data_trials, eval_data_dir=eval_data_dir,
                                           params=asv_params, device=device,  model_dir=model_dir,
                                           anon_data_suffix=anon_suffix)
                results['asv'] = asv_results
                logger.info("--- EER computation time: %f min ---" % (float(time.time() - start_time) / 60))

    if 'utility' in eval_steps:
        if 'asr' in eval_steps['utility']:
            asr_params = params['utility']['asr']
            
            model_name = asr_params['model_name']
            backend = asr_params['backend'].lower()


            if 'training' in asr_params:
                asr_train_params = asr_params['training']
                model_dir = asr_train_params['model_dir']
                if "anon_libri_360" in params:
                    asr_train_params['train_data_dir'] = output_path
                if not model_dir.exists() or asr_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    print('Perform SpeechBrain ASR training')
                    train_asr_eval(params=asr_train_params)
                    model_dir_old = model_dir
                    model_dir = scan_checkpoint(model_dir, 'CKPT')
                    shutil.copy('evaluation/utility/asr/speechbrain_asr/asr_train/hparams/transformer_inference.yaml', model_dir)
                    shutil.copy(model_dir_old / 'lm.ckpt', model_dir)
                    shutil.copy(model_dir_old / 'tokenizer.ckpt', model_dir)
                    print("--- ASR training time: %f min ---" % (float(time.time() - start_time) / 60))

            if 'evaluation' in asr_params:
                asr_eval_params = asr_params['evaluation']
                asr_model_path = asr_eval_params['model_dir']
                asr_model_path = scan_checkpoint(asr_model_path, 'CKPT')

                if not asr_model_path:
                    # asr_model_path = model_dir
                    asr_model_path = asr_eval_params['model_dir']
                start_time = time.time()
                print('Perform ASR evaluation')
                asr_results = evaluate_asr(eval_datasets=eval_data_asr, eval_data_dir=eval_data_dir,
                                           params=asr_eval_params, model_path=asr_model_path,
                                           anon_data_suffix=anon_suffix, device=device, backend=backend)
                results['asr'] = asr_results
                print("--- ASR evaluation time: %f min ---" % (float(time.time() - start_time) / 60))


        if 'gvd' in eval_steps['utility']:
            gvd_params = params['utility']['gvd']
            start_time = time.time()
            logger.info('Perform GVD evaluation')
            gvd_results = evaluate_gvd(eval_datasets=eval_data_trials, eval_data_dir=eval_data_dir, params=gvd_params,
                                       device=device, anon_data_suffix=anon_suffix)
            results['gvd'] = gvd_results
            logger.info("--- GVD  computation time: %f min ---" % (float(time.time() - start_time) / 60))

    if results:
        now = datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")
        results_summary_dir = params.get('results_summary_dir', Path('exp', 'results_summary', now))
        save_result_summary(out_dir=results_summary_dir, results_dict=results, config=params)

        for eval_type, res in results.items():
            print(f'{eval_type}:')
            print(res)
            print()
