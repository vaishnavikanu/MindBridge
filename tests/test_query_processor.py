import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.query_processing.processor import QueryProcessor

processor = QueryProcessor()

queries = [
    "I have been feeling depressed lately.",
    "मुझे बहुत उदासी महसूस हो रही है।",
    "నాకు చాలా బాధగా ఉంది.",
]

for q in queries:

    processed = processor.process(q)

    print("=" * 70)

    print("Original:")
    print(processed.original_query)

    print()

    print("Detected Language:")
    print(processed.language)

    print()

    print("Retriever Query:")
    print(processed.english_query)

    print()

    restored = processor.restore_response(
        "Depression is treatable with therapy and medication.",
        processed,
    )

    print("Restored Response:")
    print(restored)