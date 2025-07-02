from pathlib import Path
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--config', default='anon_config.yaml')
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--force_compute', default=False, type=bool)
parser.add_argument('--anonymize_train_data', default=False, type=bool)
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    print(f'Set CUDA_VISIBLE_DEVICES to {args.gpu_ids}')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

import torch

from anonymization.pipelines.sttts_pipeline import STTTSPipeline
from utils import parse_yaml, get_datasets, setup_logger

PIPELINES = {
    'sttts': STTTSPipeline
}

if __name__ == '__main__':
    config_path = Path('configs', args.config)
    config = parse_yaml(config_path)

    if args.anonymize_train_data:
        datasets = {'train-clean-360': Path(config['root_dir'], 'vpc_data', 'train-clean-360')}
    else:
        datasets = get_datasets(config)

    devices = []
    if torch.cuda.is_available():
        gpus = args.gpu_ids.split(',')
        devices = [torch.device(f'cuda:{i}') for i in range(len(gpus))]
    else:
        devices = [torch.device('cpu')]

    with torch.no_grad():
        logger = setup_logger(__name__)
        logger.info(f'Running pipeline: {config["pipeline"]}')
        pipeline = PIPELINES[config['pipeline']](config=config, force_compute=args.force_compute, devices=devices,
                                                 config_name=config_path.stem)
        pipeline.run_anonymization_pipeline(datasets)
