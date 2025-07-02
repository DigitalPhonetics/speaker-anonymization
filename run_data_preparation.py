from pathlib import Path
from argparse import ArgumentParser

from data_preparation.prepare_miami import prepare_miami
from data_preparation.prepare_seame import prepare_seame

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--miami_path', default='corpora/Bangor/Miami')
    parser.add_argument('--seame_path', default='corpora/seame')
    parser.add_argument('--output_path', default='../data')
    args = parser.parse_args()

    print('Prepare MIAMI data...')
    miami_output_path = Path(args.output_path, 'miami')
    miami_output_path.mkdir(exist_ok=True, parents=True)
    prepare_miami(Path(args.miami_path), miami_output_path)

    print('Prepare SEAME data...')
    seame_output_path = Path(args.output_path, 'seame')
    seame_output_path.mkdir(exist_ok=True, parents=True)
    prepare_seame(Path(args.seame_path), seame_output_path)