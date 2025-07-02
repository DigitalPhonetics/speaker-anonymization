from collections import defaultdict
from pathlib import Path
import soundfile

# unfortunately, it is not annotated which speaker is using which channel. So, I annotated it manually.
CHANNELS = {
    'herring01': {'LAU': 0, 'CHL': 1},
    'herring02': {'TOM': 0, 'MIG': 1},
    'herring03': {'JAC': 0, 'ASH': 1},
    'herring05': {'NOA': 0, 'MEG': 0},
    'herring06': {'NIC': 0, 'JES': 1},
    'herring07': {'RIC': 0, 'SEB': 1},
    'herring08': {'MEL': 0, 'ROB': 1, 'OSE': 0, 'OSA': 0},
    'herring09': {'CLA': 0, 'LUK': 1},
    'herring10': {'SAR': 0, 'PAI': 1},
    'herring11': {'GRA': 0, 'CAL': 1},
    'herring12': {'MIG': 0, 'TIM': 1},
    'herring13': {'VAN': 0, 'LEA': 1},
    'herring14': {'CON': 0, 'GAB': 1},
    'herring15': {'BRA': 0, 'EVN': 1},
    'herring16': {'IAN': 0, 'ABE': 1},
    'herring17': {'JAM': 0, 'IRI': 1},
    'sastre01': {'KEV': 0, 'SOF': 1},
    'sastre02': {'LUI': 0, 'AVA': 1},
    'sastre03': {'LAN': 0, 'OLI': 1, 'MAS': 1},
    'sastre04': {'EMI': 0, 'GIA': 1},
    'sastre05': {'LIL': 0, 'VIC': 1, 'OSE': 0},
    'sastre06': {'ALY': 0, 'AAR': 1},
    'sastre07': {'JUL': 0, 'JAD': 1},
    'sastre08': {'AUD': 0, 'PAO': 1},
    'sastre09': {'VAL': 0, 'KAY': 1},
    'sastre10': {'JOC': 0, 'JEN': 1},
    'sastre11': {'EVE': 0, 'DIE': 1},
    'sastre12': {'SAM': 0, 'MAD': 1},
    'sastre13': {'ELI': 0, 'COL': 1},
    'zeledon01': {'CAR': 0, 'AME': 1},
    'zeledon02': {'REB': 0, 'MAT': 1},
    'zeledon03': {'ELE': 0, 'FEL': 1},
    'zeledon04': {'HEN': 0, 'ETH': 1},
    'zeledon05': {'ISA': 0, 'MAY': 1},
    'zeledon06': {'ELL': 0, 'ABI': 1},
    'zeledon07': {'NAT': 0, 'JAS': 1},
    'zeledon08': {'FLA': 0, 'MAR': 1},
    'zeledon09': {'CHA': 0, 'GIL': 1},
    'zeledon11': {'SEA': 0, 'ANT': 1},
    'zeledon13': {'AVE': 0, 'ARI': 1},
    'zeledon14': {'LAR': 0, 'HER': 1},
}


def split_audio_files(df, out_dir):
    wav_scp = {}
    splits_per_audio = defaultdict(list)

    for idx, (utt, audio, spk, start, end, transcript) in df[['utt', 'audio', 'spk', 'start', 'end', 'transcript']].iterrows():
        channel = CHANNELS[audio][spk]
        if start is None or end is None:
            continue
        splits_per_audio[transcript].append((utt, channel, int(start), int(end)))

    index_out_of_bounds = 0
    for transcript_file, split_list in splits_per_audio.items():
        transcript_file = Path(transcript_file.replace('transcripts', 'audio'))
        audio_file = transcript_file.parent / '0wav' / f'{transcript_file.stem}.wav'
        audio, sr = soundfile.read(audio_file)
        audio = audio.T  # all audios are stereo in format (time, 2); we transpose it to easily access each channel

        for utt, channel, start, end in split_list:
            start_sample = int((start / 1000) * sr)
            end_sample = int((end / 1000) * sr)
            try:
                segment = audio[channel][start_sample:end_sample + 1]
            except IndexError:
                index_out_of_bounds += 1
                continue
            segment_file = out_dir / f'{utt}.wav'
            soundfile.write(file=segment_file, data=segment, samplerate=sr)
            wav_scp[utt] = str(segment_file)

    print('Index Out Of Bounds:', index_out_of_bounds)

    return wav_scp

