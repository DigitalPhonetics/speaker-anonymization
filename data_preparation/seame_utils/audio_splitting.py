from pathlib import Path
import soundfile


def split_audio_files(df, out_dir):
    wav_scp = {}
    index_out_of_bounds = 0

    # group by audio so that we only need to load each audio once and not per utterance
    for audio_path, group in df.groupby('audio_path'):
        audio, sr = soundfile.read(audio_path)

        for idx, (utt, start, end) in group[['utt', 'start', 'end']].iterrows():
            start_sample = int((start / 1000) * sr)
            end_sample = int((end / 1000) * sr)
            try:
                segment = audio[start_sample:end_sample + 1]
            except IndexError:
                index_out_of_bounds += 1
                continue
            segment_file = out_dir / f'{utt}.wav'
            soundfile.write(file=segment_file, data=segment, samplerate=sr)
            wav_scp[utt] = str(segment_file)

    if index_out_of_bounds:
        print('Index Out Of Bounds:', index_out_of_bounds)

    return wav_scp

