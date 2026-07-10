import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.query_processing.language_service import LanguageService

service = LanguageService()

queries = [
    "I have been feeling depressed lately.",
    "मुझे बहुत उदासी महसूस हो रही है।",
    "నాకు చాలా బాధగా ఉంది."
]

for query in queries:
    info = service.detect_language(query)

    print("=" * 60)
    print("Query:", query)
    print("Detected Language:", info.code)
    print("Confidence:", info.confidence)
    print("Needs Translation:", info.translation_required)