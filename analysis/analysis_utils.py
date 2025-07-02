import re
from scipy.stats import mannwhitneyu


def parse_wer_file(filename):
	file_results = []
	# Define regex to parse the file content
	utterance_regex = re.compile(
		r'^((?:\w|\-)+), %WER\s(\d+\.\d{2})\s\[([^\n]+)\]\n([^\n]+)\n([^\n]+)\n([^\n]+)(\n={80})?', re.MULTILINE)
	repeated_space_regex = re.compile(r' +')
	with open(filename, 'r') as f:
		content = f.read()
	content = repeated_space_regex.sub(' ', content)
	matches = utterance_regex.findall(content)
	for match in matches:
		utterance_id = match[0]
		metric_result = float(match[1])
		utt_error_details = match[2].split(',')[0].split('/')
		error_words = int(utt_error_details[0].strip())
		total_words = int(utt_error_details[1].strip())
		reference = match[3]
		hypothesis = match[5]
		file_results.append((utterance_id, metric_result, error_words, total_words, reference, hypothesis))
	return file_results


def get_stat_significance(subdf, df, metric, subdf_wer, total_wer):
	# test if what we observe is stat. significant
	if total_wer > subdf_wer:
		# H0: same distribution, alternative H1: subdf has smaller wer
		scenario = '<<'  # WER(subdf) << WER(whole df)
		stat_significance = mannwhitneyu(df[metric], subdf[metric], method="auto", alternative="greater")
	else:
		# H1: same distribution, alternative H1: subdf has higher wer
		scenario = '>>'  # WER(subdf) >> WER(whole df)
		stat_significance = mannwhitneyu(subdf[metric], df[metric], method="auto", alternative="greater")
	return scenario, stat_significance
