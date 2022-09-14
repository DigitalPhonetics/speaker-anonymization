from argparse import ArgumentParser
from pathlib import Path
import subprocess
import torch
from inference import InferenceAnonymizer, InferenceASR, InferenceTTS
from evaluation import prepare_evaluation_data, copy_evaluation_results

parser = ArgumentParser()
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()


# Settings
gpu = args.gpu     # None for CPU, integer for GPU ID
settings = {
    'datasets': ['libri_dev', 'libri_test', 'vctk_dev', 'vctk_test'],
    'anonymizer': 'pool_minmax_ecapa+xvector',  # name of anonymization model
    'asr': 'asr_tts-phn_en.zip',  # name of ASR model
    'tts_hifigan': 'best.pt',  # name of TTS HiFiGAN model
    'tts_fastspeech': 'trained_on_ground_truth_phonemes.pt'  # name of TTS FastSpeech2 model
}
force_compute = []  # options: 'anon', 'asr', 'tts'
settings['text_is_phonemes'] = '-phn' in settings['asr']

# Some static variables
data_dir = Path('Voice-Privacy-Challenge-2020', 'baseline', 'data')
vectors_dir = Path('original_speaker_embeddings')
models_dir = Path('models')
results_dir = Path('results')

# the challenge's eval scripts require the data to be at a specific location
eval_data_dir = Path('Voice-Privacy-Challenge-2020', 'baseline', 'data')

if not torch.cuda.is_available():
    gpu = None
device = torch.device(f'cuda:{gpu}') if gpu is not None else torch.device('cpu')

dataset_splits = {
    'libri': ['trials_f', 'trials_m', 'enrolls'],
    'vctk': ['trials_f_all', 'trials_m_all', 'enrolls']
}

datasets = [f'{dset}_{split}' for dset in settings['datasets'] for split in dataset_splits[dset.split('_')[0]]]
anon_wav_scps = {}


print('Set up components...')
anonymizer = InferenceAnonymizer(settings['anonymizer'], data_dir=data_dir, model_dir=models_dir,
                                 results_dir=results_dir, vectors_dir=vectors_dir, device=device,
                                 force_compute='anon' in force_compute)
asr = InferenceASR(settings['asr'], device=device, data_dir=data_dir, model_dir=models_dir,
                   results_dir=results_dir, force_compute='asr' in force_compute)
tts = InferenceTTS(hifigan_model_name=settings['tts_hifigan'], fastspeech_model_name=settings['tts_fastspeech'],
                   anon_model_name=settings['anonymizer'], asr_model_name=settings['asr'], model_dir=models_dir,
                   results_dir=results_dir, device=device, force_compute='tts' in force_compute)

with torch.inference_mode():
    for i, dataset in enumerate(datasets):
        print(f'{i+1}/{len(datasets)}: Processing {dataset}...')
        anon_embeddings, new_anon = anonymizer.anonymize_embeddings(dataset=dataset)
        texts, utt2spk, new_text = asr.recognize_speech(dataset=dataset)
        wav_scp, _ = tts.read_texts(dataset=dataset, texts=texts, anon_embeddings=anon_embeddings, utt2spk=utt2spk,
                                    force_compute=(new_anon or new_text),
                                    text_is_phonemes=settings['text_is_phonemes'],
                                    emb_level=anonymizer.anonymizer.emb_level)
        anon_wav_scps[dataset] = wav_scp
print('Done')

# Evaluation
print(f'Prepare evaluation data for {datasets}...')
prepare_evaluation_data(dataset_list=datasets, anon_wav_scps=anon_wav_scps, orig_data_path=data_dir,
                        anon_vectors_path=anonymizer.results_dir, output_path=eval_data_dir)

if 'vctk_dev' in settings['datasets']:
    print('Make anon subsets for vctk_dev...')
    subprocess.run(['./evaluation/run_make_vctk_anon_subsets.sh', '--split', 'dev'], check=True)
if 'vctk_test' in settings['datasets']:
    print('Make anon subsets for vctk_test...')
    subprocess.run(['./evaluation/run_make_vctk_anon_subsets.sh', '--split', 'test'], check=True)

print('Perform evaluation...')
subprocess.run(['./evaluation/run_evaluation.sh', '--mcadams', 'false', '--gpu', str(gpu) or 'cpu',
                *settings['datasets']], check=True)


# Copy the evaluation results to our results directory
copy_evaluation_results(results_dir=results_dir / 'evaluation', eval_dir=eval_data_dir, settings=settings)

subprocess.run(['./utils/run_cleanup.sh'], check=True)
