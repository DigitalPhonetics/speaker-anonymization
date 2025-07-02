from pathlib import Path
from argparse import ArgumentParser
import torch

from analysis.analyze_codeswitching_points import analyze_codeswitching_points
from analysis.analyze_data_characteristics import analyze_data_characteristics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='../data')
    parser.add_argument('--results_path', default='../exp')
    parser.add_argument('--dataset', default='both', choices=['miami', 'seame', 'both'])
    parser.add_argument('--gpu_id', default='0')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)

    datasets = ['miami', 'seame'] if args.dataset == 'both' else [args.dataset]
    datasets2lang = {'miami': ['en', 'es', 'cs'], 'seame': ['en', 'zh', 'cs']}

    for dataset in datasets:
        print(f'Perform analysis for {dataset.upper()}...')

        print(f'Analyze code-switching points for {dataset.upper()}-cs...')
        # code-switching points are only analyzed for code-switching subsets
        analyze_codeswitching_points(results_path / f'{dataset}_cs', device=device)

        characteristics_file = data_path / dataset / 'data_characteristics.csv'
        for lang in datasets2lang[dataset]:
            print(f'Analyze data characteristics for {dataset.upper()}-{lang}...')
            analyze_data_characteristics(results_path / f'{dataset}_{lang}', characteristics_file, dataset=dataset)
