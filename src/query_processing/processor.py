from .language_service import LanguageService
from .models import ProcessedQuery
from src.translation.translator import get_translator


class QueryProcessor:

    def __init__(self):
        self.language_service = LanguageService()
        self.translator = get_translator()

    def process(
        self,
        query: str,
    ) -> ProcessedQuery:

        language = self.language_service.detect_language(query)

        #
        # Translate only if necessary
        #
        if language.translation_required:

            english_query = self.translator.translate(
                text=query,
                source_language=language.code,
                target_language="en",
            )

        else:

            english_query = query

        return ProcessedQuery(
            original_query=query,
            english_query=english_query,
            language=language.code,
            translation_required=language.translation_required,
            confidence=language.confidence,
        )

    def restore_response(
        self,
        response: str,
        processed_query: ProcessedQuery,
    ) -> str:

        if not processed_query.translation_required:
            return response

        return self.translator.translate(
            text=response,
            source_language="en",
            target_language=processed_query.language,
        )