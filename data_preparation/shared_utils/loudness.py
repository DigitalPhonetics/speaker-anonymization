import soundfile
import pyloudnorm as pyln


def compute_loudness(wav_path):
    try:
        audio, sr = soundfile.read(wav_path)
        if audio.shape[0] < (0.4 * sr):
            block_size = (audio.shape[0] / sr) - 0.01
        else:
            block_size = 0.400
        meter = pyln.Meter(sr, block_size=block_size)
        loudness = meter.integrated_loudness(audio)
        if loudness < -80:  # we sometimes get -inf, see https://github.com/csteinmetz1/pyloudnorm/issues/52
            return -80
        return loudness
    except ValueError:
        return None