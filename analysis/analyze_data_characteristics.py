from pathlib import Path
import pandas as pd
import numpy as np

from .analysis_utils import parse_wer_file, get_stat_significance


CHARACTERISTICS = ['has_repetitions', 'has_hesitations_discourse', 'has_incomplete_utt', 'has_foreign_tokens',
				   'has_abbreviation', 'has_unknown', 'is_very_short', 'is_very_long', 'suspicious_dur_word_ratio',
				   'low_loudness', 'high_loudness']

THRESHOLD = 0.025


def analyze_data_characteristics(results_path, characteristics_file, dataset='seame', anon_suffix='_anon', metric='wer',
								 split='test', characteristics=None):
	characteristics = characteristics or CHARACTERISTICS
	charact_df = pd.read_csv(characteristics_file, sep='\t')
	charact_df = charact_df[['utt', *characteristics, 'has_no_problems']]

	results_path = Path(results_path)
	if (results_path / 'asr_orig_whisper').exists():
		results_path = results_path / 'asr_orig_whisper'

	
	for asr_dir in  results_path.glob(f'{dataset}-*_{split}_asr'):
		if not asr_dir.is_dir():
			continue
		lang = asr_dir.name.split('_')[0].split('-')[1]
		orig_results = parse_wer_file(asr_dir / metric)
		anon_results = parse_wer_file(Path(f'{str(asr_dir.absolute())}{anon_suffix}', metric))

		orig_df = pd.DataFrame(orig_results, columns=['utt', 'wer', 'error_words', 'total_words', 'ref', 'hyp'])
		orig_df = orig_df[['utt', 'wer', 'error_words', 'total_words']]
		orig_df['lang'] = lang
		orig_df = orig_df.merge(charact_df, on='utt', how='inner')

		anon_df = pd.DataFrame(anon_results, columns=['utt', 'wer', 'error_words', 'total_words', 'ref', 'hyp'])
		anon_df = anon_df[['utt', 'wer', 'error_words', 'total_words']]
		anon_df['lang'] = lang
		anon_df = anon_df.merge(charact_df, on='utt', how='inner')

		print(f'Statistics for dataset {dataset.upper()} and split {split}:\n\n')
		_print_overall_statistics(orig_df, anon_df, metric=metric.upper())
		for charact in characteristics:
			_print_charact_statistics(charact, orig_df, anon_df, metric=metric.upper())


def _print_overall_statistics(orig_df, anon_df, metric):
	"""
	Overall statistics:

				 | No problems  | Total        |
	| Data | Lang| # utts | WER | # utts | WER |
	| ---- | --- |
	| Orig | EN  |
	| Orig | ZH  |
	| Orig | CS  |
	| Anon | EN  |
	| Anon | ZH  |
	| Anon | CS  |
	"""
	orig_stat = _get_overall_stat(orig_df, 'orig', metric)
	anon_stat = _get_overall_stat(anon_df, 'anon', metric)
	df = pd.concat([orig_stat, anon_stat], ignore_index=True)
	print('Overall statistics: \n')
	print(df)
	print('-'* 20 + '\n')


def _get_overall_stat(df, data='orig', metric='WER'):
	groups = df.groupby('lang')
	np_groups = df[df['has_no_problems'] == True].groupby('lang')
	res = pd.DataFrame(index=groups.groups.keys())

	res[('Total', '# utts')] = groups['utt'].count()
	res[('Total', metric)] = round((groups['error_words'].sum() / groups['total_words'].sum()) * 100, 2)
	res[('No problems', '# utts')] = np_groups['utt'].count()
	res[('No problems', metric)] = round((np_groups['error_words'].sum() / np_groups['total_words'].sum()) * 100, 2)

	res.columns = pd.MultiIndex.from_tuples(res.columns)
	res = res.reset_index().rename(columns={'index': 'lang'})
	res.insert(0, column='Data', value=data)
	return res


def _print_charact_statistics(charact, orig_df, anon_df, metric):
	"""
	Characteristic: has_abbreviation

				 | True         | False        | True >> False  | True >> No problems | True >> Total |
	| Data | Lang| # utts | WER | # utts | WER | pvalue  | sig  | pvalue      | sig   | pvalue  | sig |
	| ---- | --- |
	| Orig | EN  |
	| Orig | ZH  |
	| Orig | CS  |
	| Anon | EN  |
	| Anon | ZH  |
	| Anon | CS  |
	"""
	orig_stat = _get_charact_stat(orig_df, charact, 'orig', metric)
	anon_stat = _get_charact_stat(anon_df, charact, 'anon', metric)
	df = pd.concat([orig_stat, anon_stat], ignore_index=True)
	print(f'Characteristic: {charact}: \n')
	print(df.to_string())
	print('-' * 20 + '\n')


def _get_wer(df):
	return (df['error_words'].sum() / df['total_words'].sum()) * 100


def _get_stat_sig(lang, groups_df1, groups_df2, metric):
	if len(groups_df1) == 0 or len(groups_df2) == 0:
		return None
	df1 = groups_df1.get_group(lang)
	df2 = groups_df2.get_group(lang)
	metric = metric.lower()
	df1_wer = _get_wer(df1)
	df2_wer = _get_wer(df2)
	scenario, stat_sig = get_stat_significance(df1, df2, metric, df1_wer, df2_wer)
	if scenario == '>>':  # expected scenario
		return stat_sig.pvalue
	else:
		return np.inf

def _get_charact_stat(df, charact, data='orig', metric='WER'):
	groups = df.groupby('lang')
	np_groups = df[df['has_no_problems'] == True].groupby('lang')
	true_groups = df[df[charact] == True].groupby('lang')
	false_groups = df[df[charact] == False].groupby('lang')
	res = pd.DataFrame(index=groups.groups.keys())

	res[(f'{charact}: True', '# utts')] = true_groups['utt'].count()
	res[(f'{charact}: True', metric)] = (true_groups['error_words'].sum() / true_groups['total_words'].sum()) * 100
	res[(f'{charact}: False', '# utts')] = false_groups['utt'].count()
	res[(f'{charact}: False', metric)] = (false_groups['error_words'].sum() / false_groups['total_words'].sum()) * 100

	res[('True >> False', 'pvalue')] = res.apply(lambda x: _get_stat_sig(x.name, true_groups, false_groups, metric), axis=1)
	res[('True >> False', 'stat. sig.')] = res[('True >> False', 'pvalue')] < THRESHOLD
	res[('True >> No problems', 'pvalue')] = res.apply(lambda x: _get_stat_sig(x.name, true_groups, np_groups, metric), axis=1)
	res[('True >> No problems', 'stat. sig.')] = res[('True >> No problems', 'pvalue')] < THRESHOLD
	res[('True >> Total', 'pvalue')] = res.apply(lambda x: _get_stat_sig(x.name, true_groups, groups, metric), axis=1)
	res[('True >> Total', 'stat. sig.')] = res[('True >> Total', 'pvalue')] < THRESHOLD

	res.columns = pd.MultiIndex.from_tuples(res.columns)
	res = res.reset_index().rename(columns={'index': 'lang'})
	res.insert(0, column='Data', value=data)
	return res