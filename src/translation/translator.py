import torch
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

#
# Global singleton cache
#
_TRANSLATOR_CACHE = None

LANGUAGE_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
}


class Translator:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = None
        self.model = None

        self.loaded = False

    def load(self):

        if self.loaded:
            return

        print("Loading NLLB Translator...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)

        self.loaded = True

        print("Translator Ready.")

    @property
    def ready(self):
        return self.loaded

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:

        if source_language == target_language:
            return text

        start = time.time()

        if not self.loaded:
            self.load()

        self.tokenizer.src_lang = LANGUAGE_CODES[source_language]

        if len(text.split()) < 8:
            text = (
                "Translate the following mental health query "
                "into natural English while preserving its meaning:\n\n"
                f"{text}"
            )
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():

            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                    LANGUAGE_CODES[target_language]
                ),
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        latency = (time.time() - start) * 1000

        print(
            f"[Translator] "
            f"{source_language}->{target_language} "
            f"{latency:.1f} ms"
        )

        return translated
    
def get_translator() -> Translator:
    global _TRANSLATOR_CACHE

    if _TRANSLATOR_CACHE is None:
        translator = Translator()
        translator.load()
        _TRANSLATOR_CACHE = translator

    return _TRANSLATOR_CACHE