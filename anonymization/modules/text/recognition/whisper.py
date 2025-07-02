from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ..text import Text
from utils import LANGS_SHORT_CODE_TO_NAME, UNWANTED_PUNCTUATION


class WhisperASR:

    def __init__(self, model_path, device, utt_start_token='', utt_end_token='', lang=None, batch_size=16,
                 languages=None, fallback_lang=None, **kwargs):
        self.device = device
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_path = model_path
        self.use_flash_attention_2 = False
        self.utt_start_token = utt_start_token
        self.utt_end_token = utt_end_token
        self.lang = LANGS_SHORT_CODE_TO_NAME[lang] if lang is not None else lang
        self.cs_languages = [LANGS_SHORT_CODE_TO_NAME[lang] for lang in languages] if languages else None
        self.fallback_lang = LANGS_SHORT_CODE_TO_NAME[fallback_lang] if fallback_lang else 'english'

        model_id = 'openai/whisper-large-v3'
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                                                          use_safetensors=True, cache_dir=model_path)
        model.to(self.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_path)
        self.speech2text = pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
                                    feature_extractor=processor.feature_extractor, batch_size=batch_size,
                                    return_timestamps=False, torch_dtype=torch_dtype, device=self.device,
                                    max_new_tokens=128, return_language=True, chunk_length_s=30)

        self.output = 'text'

        suppress_file_path = model_path.parent / 'whisper_suppress_token_ids.txt'
        if suppress_file_path.exists():
            self.suppress_tokens = []
            with open(suppress_file_path, 'r') as f:
                for line in f:
                    self.suppress_tokens.append(int(line.strip()))
        else:
            self.suppress_tokens = model.generation_config.suppress_tokens  # Whisper is already suppressing some tokens by default
            self.suppress_tokens += self._define_bad_tokens(tokenizer=processor.tokenizer)
            self.suppress_tokens = sorted(list(set(self.suppress_tokens)))
            with open(suppress_file_path, 'w') as f:
                for token in self.suppress_tokens:
                    f.write(f'{token}\n')

    def _define_bad_tokens(self, tokenizer):
        # Whisper likes to generate some output that is very bad for the TTS, e.g., by transcribing a number word as number instead of string
        unwanted_tokens = UNWANTED_PUNCTUATION + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        suppress_tokens = []
        for token_id in range(tokenizer.vocab_size):
            token_str = tokenizer.decode([token_id]).removeprefix(' ')
            if any(c in unwanted_tokens for c in token_str):
                suppress_tokens.append(token_id)
        return suppress_tokens

    def recognize_speech_of_audio(self, audio_file):
        if self.lang is None:
            text = self.speech2text(audio_file)['text']
        else:
            text = self.speech2text(audio_file, generate_kwargs={'language': self.lang,
                                                                 'suppress_tokens': self.suppress_tokens})['text']
        text = self.utt_start_token + text.strip() + self.utt_end_token
        return text

    def recognize_speech_of_dataset(self, audio_dataset, out_dir, save_intermediate=True, job_id=None):
        texts = Text(is_phones=(self.output == 'phones'))
        allowed_languages = self.cs_languages

        if len(audio_dataset) == 0:
            return texts

        with torch.inference_mode():
            if self.lang is None:
                outputs = self.speech2text(audio_dataset, generate_kwargs={'suppress_tokens': self.suppress_tokens})
            else:
                outputs = self.speech2text(audio_dataset, generate_kwargs={'language': self.lang,
                                                                           'suppress_tokens': self.suppress_tokens})

        if job_id is None:  # single processing
            add_suffix = None
            tqdm_params = {}
        else: # process amongst multiple processes
            add_suffix = f'_{job_id}'
            tqdm_params = {'desc': f'Job {job_id}', 'leave': True}

        i = 0
        for output in tqdm(outputs, **tqdm_params):
            inferred_language = output['chunks'][0]['language']
            utt = output['utt'][0]
            if allowed_languages and inferred_language not in allowed_languages:
                instance = audio_dataset.get_instance(utt)
                output = self.speech2text(instance, generate_kwargs={'suppress_tokens': self.suppress_tokens,
                                                                     'language': self.fallback_lang})
            spk = output['spk'][0]
            sentence = self.utt_start_token + output['text'].strip() + self.utt_end_token
            texts.add_instance(sentence=sentence, utterance=utt, speaker=spk)

            i += 1
            if i % 100 == 0 and save_intermediate:
                texts.save_text(out_dir=out_dir, add_suffix=add_suffix)

        if save_intermediate:
            texts.save_text(out_dir=out_dir, add_suffix=add_suffix)
        return texts
