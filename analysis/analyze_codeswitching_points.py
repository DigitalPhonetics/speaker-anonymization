from pathlib import Path
import pandas as pd

from .analysis_utils import parse_wer_file, get_stat_significance
from utils import LanguageTextProcessing, read_kaldi_format


def analyze_codeswitching_points(results_path, device, anon_suffix='_anon', metric='wer', split='test',
								 languages=None):
	results_path = Path(results_path)
	if (results_path / 'asr_orig_whisper').exists():
		results_path = results_path / 'asr_orig_whisper'

	for asr_dir in  results_path.glob(f'*-cs_{split}_asr{anon_suffix}'):
		if not asr_dir.is_dir():
			continue
		dataset_name = asr_dir.name.split('-')[0]
		orig_text = read_kaldi_format(asr_dir / 'gold_text', values_as_string=True)
		anon_text = read_kaldi_format(asr_dir / 'text', values_as_string=True)
		results = _read_asr_results(asr_dir, metric, orig_text, anon_text)

		if languages:
			textprocessing = LanguageTextProcessing(cs_languages=languages, device=device)
		elif dataset_name == 'seame':
			textprocessing = LanguageTextProcessing(cs_languages={'eng', 'cmn'}, device=device)
		elif dataset_name == 'miami':
			textprocessing = LanguageTextProcessing(cs_languages={'eng', 'spa'}, device=device)
		else:
			raise ValueError(f'Unknown dataset {dataset_name}. Automatic language selection is only available for SEAME and MIAMI data. '
							 f'If you are using a different dataset, you need to specify the languages like languages=["eng", "cmn"].')

		results = _get_codeswitching_points(results, textprocessing)
		print(f'Results for {dataset_name}:')
		_analyze_codeswitching_behavior(results)
		print('-' * 20)
		print()


def _read_asr_results(exp_dir, metric, orig_text, anon_text):
	result_data = []
	file_results = parse_wer_file(exp_dir / metric)

	for utt_id, metric_result, error_words, total_words, reference, hypothesis in file_results:
		utt_results = {}
		utt_results[metric] = metric_result
		utt_results[f'{metric}_errors'] = error_words
		utt_results[f'{metric}_total'] = total_words
		utt_results['orig_text'] = orig_text[utt_id]
		utt_results['anon_text'] = anon_text[utt_id ]
		result_data.append(utt_results)

	return  pd.DataFrame(result_data)


def _get_codeswitching_points(results, textprocessing):
	# get number of codeswitching points in each transcript
	get_n_cs_points = lambda text: max(len(textprocessing.separate_lang_substrings(text)) - 1, 0)
	results['csp_orig'] = results['orig_text'].apply(get_n_cs_points)
	results['csp_anon'] = results['anon_text'].apply(get_n_cs_points)

	# measure the difference in number of codeswitching points between two transcripts
	results['delta_cs_anon_orig'] = results['csp_anon'] - results['csp_orig']
	return results


def _get_wer(df, metric):
	return (df[f'{metric}_errors'].sum() / df[f'{metric}_total'].sum()) * 100


def _analyze_codeswitching_behavior(results, metric='wer', stat_threshold=0.025):
	# test how often CSP(O) == 0
	n_no_csp_orig = len(results[results['csp_orig'] == 0])
	print(f'No CS points in original transcripts in {n_no_csp_orig} samples ({round((n_no_csp_orig / len(results))*100, 2)}%)')

	# only consider samples with CSP(O) > 0
	results = results[results['csp_orig'] > 0]

	total = len(results)
	total_wer = _get_wer(results, metric)
	print(f'Total number of samples: {total}, {metric.upper()}: {round(total_wer, 2)}')

	# mean number of CSP
	mean_csp_o = results['csp_orig'].mean()
	mean_csp_a = results['csp_anon'].mean()
	print(f'Mean number of CS points in original transcripts: avg. CSP(O)={round(mean_csp_o, 2)}')
	print(f'Mean number of CS points in anon transcripts: avg. CSP(A)={round(mean_csp_a, 2)}')

	# reduction in CSP during anonymization
	csp_reduction = results[results['delta_cs_anon_orig'] < 0]
	csp_reduction_wer = _get_wer(csp_reduction, metric)
	scenario, stat_sig = get_stat_significance(csp_reduction, results, metric, csp_reduction_wer, total_wer)
	print(f'Less CS points after anonymization (CSP(A) < CSP(O)) in {len(csp_reduction)} samples ({round((len(csp_reduction) / total)*100, 2)}%)')
	print(f'{metric.upper()} for CSP(A) < CSP(O): {round(csp_reduction_wer, 2)}')
	if stat_sig.pvalue < stat_threshold:
		print(f'{metric.upper()}(CSP(A) < CSP(O)) {scenario} {metric.upper()}(total) is stat. significant with p={round(stat_sig.pvalue, 4)}')
	else:
		print(f'{metric.upper()}(CSP(A) < CSP(O)) {scenario} {metric.upper()}(total) is NOT stat. significant (p={round(stat_sig.pvalue, 4)})')

	# no CSP after anonymization
	no_csp_anon = results[results['csp_anon'] == 0]
	no_csp_anon_wer = _get_wer(no_csp_anon, metric)
	scenario, stat_sig = get_stat_significance(no_csp_anon, results, metric, no_csp_anon_wer, total_wer)
	print(f'No CS points after anonymization (CSP(A)=0)) in {len(no_csp_anon)} samples ({round((len(no_csp_anon) / total)*100, 2)}%)')
	print(f'{metric.upper()} for CSP(A)=0: {round(no_csp_anon_wer, 2)}')
	if stat_sig.pvalue < stat_threshold:
		print(f'{metric.upper()}(CSP(A)=0) {scenario} {metric.upper()}(total) is stat. significant with p={round(stat_sig.pvalue, 4)}')
	else:
		print(f'{metric.upper()}(CSP(A)=0) {scenario} {metric.upper()}(total) is NOT stat. significant (p={round(stat_sig.pvalue, 4)})')

	# same number of CSP in anonymization and original
	same_csp = results[results['delta_cs_anon_orig'] == 0]
	same_csp_wer = _get_wer(same_csp, metric)
	scenario, stat_sig = get_stat_significance(same_csp, results, metric, same_csp_wer, total_wer)
	print(f'Same number of CS points after anonymization (CSP(A) = CSP(O)) in {len(same_csp)} samples ({round((len(same_csp) / total)*100, 2)}%)')
	print(f'{metric.upper()} for CSP(A) = CSP(O): {round(same_csp_wer, 2)}')
	if stat_sig.pvalue < stat_threshold:
		print(
			f'{metric.upper()}(CSP(A) = CSP(O)) {scenario} {metric.upper()}(total) is stat. significant with p={round(stat_sig.pvalue, 4)}')
	else:
		print(
			f'{metric.upper()}(CSP(A) = CSP(O)) {scenario} {metric.upper()}(total) is NOT stat. significant (p={round(stat_sig.pvalue, 4)})')
