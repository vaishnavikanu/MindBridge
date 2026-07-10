import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.config import config
from src.utils.logging import setup_logging


def main():

    setup_logging()

    config.load("configs/config.yaml")

    print("=" * 80)
    print("Initializing RAG Pipeline...")
    print("=" * 80)

    pipeline = RAGPipeline(generator_type="ollama")
    pipeline.initialize()

    print("\nPipeline Ready.\n")

    test_queries = [

        (
            "English",
            "What are the symptoms of depression?"
        ),

        (
            "Hindi",
            "डिप्रेशन के लक्षण क्या हैं?"
        ),

        (
            "Telugu",
            "డిప్రెషన్ లక్షణాలు ఏమిటి?"
        ),

    ]

    for language, query in test_queries:

        print("=" * 80)
        print(f"Language : {language}")
        print(f"Query    : {query}")
        print("=" * 80)

        result = pipeline.query(
            query=query,
            role="patient",
            user_id="patient_001",
        )

        print("\nResponse:\n")
        print(result.response)

        print("\nLatency:")
        print(f"{result.latency_ms:.2f} ms")

        print("\nRetrieved Chunks:")
        print(len(result.retrieved_chunks))

        print("\n")


if __name__ == "__main__":
    main()