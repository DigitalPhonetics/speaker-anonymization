from tqdm import tqdm
import soundfile
import time
from torch.multiprocessing import Pool, set_start_method
from itertools import repeat

from .ims_tts import ImsTTS
from utils import create_clean_dir, setup_logger

set_start_method('spawn', force=True)
logger = setup_logger(__name__)

class SpeechSynthesis:

    def __init__(self, devices, settings, results_dir=None, save_output=True, force_compute=False):
        self.devices = devices
        self.output_sr = settings.get('output_sr', 16000)
        self.save_output = save_output
        self.force_compute = force_compute if force_compute else settings.get('force_compute_synthesis', False)
        lang = settings.get('lang', 'en')
        if lang == 'cs':
            cs_languages = settings.get('languages', ['en', 'zh'])
        else:
            cs_languages = None
        accent_lang = settings.get('accent_lang', None)
        logger.info(f'Use language {lang}')

        synthesizer_type = settings.get('synthesizer', 'ims')
        if synthesizer_type == 'ims':
            hifigan_path = settings['hifigan_path']
            fastspeech_path = settings['fastspeech_path']
            embedding_path = settings.get('embeddings_path', None)

            self.tts_models = []
            for device in self.devices:
                self.tts_models.append(ImsTTS(hifigan_path=hifigan_path, fastspeech_path=fastspeech_path,
                                              embedding_path=embedding_path, device=device,
                                              output_sr=self.output_sr, lang=lang, accent_lang=accent_lang,
                                              cs_languages=cs_languages))

        if results_dir:
            self.results_dir = results_dir
        elif 'results_path' in settings:
            self.results_dir = settings['results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_output:
                raise ValueError('Results dir must be specified in parameters or settings!')

    def synthesize_speech(self, dataset_name, texts, speaker_embeddings, prosody=None, emb_level='spk'):
        # depending on whether we save the generated audios to disk or not, we either return a dict of paths to the
        # saved wavs (wav.scp) or the wavs themselves
        dataset_results_dir = self.results_dir / dataset_name if self.save_output else ''
        wavs = {}

        if dataset_results_dir.exists() and not self.force_compute:
            already_synthesized_utts = {wav_file.stem: str(wav_file.absolute())
                                        for wav_file in dataset_results_dir.glob('*.wav')
                                        if wav_file.stem in texts.utterances}

            if len(already_synthesized_utts):
                logger.info(f'No synthesis necessary for {len(already_synthesized_utts)} of {len(texts)} utterances...')
                texts.remove_instances(list(already_synthesized_utts.keys()))
                if self.save_output:
                    wavs = already_synthesized_utts
                else:
                    wavs = {}
                    for utt, wav_file in already_synthesized_utts.items():
                        wav, _ = soundfile.read(wav_file)
                        wavs[utt] = wav

        if texts:
            logger.info(f'Synthesize {len(texts)} utterances...')
            if self.force_compute or not dataset_results_dir.exists():
                create_clean_dir(dataset_results_dir)

            text_is_phones = texts.is_phones

            if len(self.tts_models) == 1:
                instances = []
                for text, utt, speaker in texts:
                    try:
                        if emb_level == 'spk':
                            speaker_embedding = speaker_embeddings.get_embedding_for_identifier(speaker)
                        else:
                            speaker_embedding = speaker_embeddings.get_embedding_for_identifier(utt)

                        if prosody:
                            utt_prosody_dict = prosody.get_instance(utt)
                        else:
                            utt_prosody_dict = {}
                        instances.append((text, utt, speaker_embedding, utt_prosody_dict))
                    except KeyError:
                        logger.warn(f'Key error at {utt}')
                        continue
                wavs.update(synthesis_job(instances=instances, tts_model=self.tts_models[0],
                                          out_dir=dataset_results_dir, sleep=0, text_is_phones=text_is_phones,
                                          save_output=self.save_output))

            else:
                num_processes = len(self.tts_models)
                sleeps = [10 * i for i in range(num_processes)]
                text_iterators = texts.get_iterators(n=num_processes)

                instances = []
                for iterator in text_iterators:
                    job_instances = []
                    for text, utt, speaker in iterator:
                        try:
                            if emb_level == 'spk':
                                speaker_embedding = speaker_embeddings.get_embedding_for_identifier(speaker)
                            else:
                                speaker_embedding = speaker_embeddings.get_embedding_for_identifier(utt)

                            if prosody:
                                utt_prosody_dict = prosody.get_instance(utt)
                            else:
                                utt_prosody_dict = {}
                            job_instances.append((text, utt, speaker_embedding, utt_prosody_dict))
                        except KeyError:
                            logger.warn(f'Key error at {utt}')
                            continue
                    instances.append(job_instances)

                # multiprocessing
                with Pool(processes=num_processes) as pool:
                    job_params = zip(instances, self.tts_models, repeat(dataset_results_dir), sleeps,
                                     repeat(text_is_phones), repeat(self.save_output))
                    new_wavs = pool.starmap(tqdm(synthesis_job), job_params)

                for new_wav_dict in new_wavs:
                    wavs.update(new_wav_dict)
        return wavs


def synthesis_job(instances, tts_model, out_dir, sleep, text_is_phones=False, save_output=False):
    time.sleep(sleep)

    wavs = {}
    for text, utt, speaker_embedding, utt_prosody_dict in tqdm(instances):
        wav = tts_model.read_text(text=text, speaker_embedding=speaker_embedding, text_is_phones=text_is_phones,
                                  **utt_prosody_dict)
        if wav is None:
            continue

        if save_output:
            out_file = str((out_dir / f'{utt}.wav').absolute())
            soundfile.write(file=out_file, data=wav, samplerate=tts_model.output_sr)
            wavs[utt] = out_file
        else:
            wavs[utt] = wav
    return wavs


