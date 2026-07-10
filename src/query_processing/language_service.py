from dataclasses import dataclass

from lingua import Language, LanguageDetectorBuilder


@dataclass
class LanguageInfo:
    code: str
    confidence: float
    translation_required: bool


class LanguageService:

    def __init__(self):

        self.detector = None

        self.loaded = False

    def load_models(self):

        if self.loaded:
            return

        self.detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH,
            Language.HINDI,
            Language.TELUGU,
        ).build()

        self.loaded = True

    def detect_language(self, text: str):

        if not self.loaded:
            self.load_models()

        language = self.detector.detect_language_of(text)

        if language is None:
            return LanguageInfo(
                code="en",
                confidence=0.0,
                translation_required=False,
            )

        code_map = {
            Language.ENGLISH: "en",
            Language.HINDI: "hi",
            Language.TELUGU: "te",
        }

        code = code_map.get(language, "en")

        return LanguageInfo(
            code=code,
            confidence=1.0,
            translation_required=(code != "en"),
        )

    def translate_to_english(self, text):

        return text

    def translate_from_english(
        self,
        text,
        target_language,
    ):

        return text