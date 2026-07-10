import json
import csv
import time
from pathlib import Path

from evaluation.run_evaluation import (
    evaluate,
    load_dataset,
)

from src.pipeline.rag_pipeline import RAGPipeline


RESULT_DIR = Path("evaluation/results/clinician")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

#DATASET = "counselchat_unique"
DATASET = "clinician_qa_1"

EXPERIMENTS = {

    "dense": {
        "graph": False,
        "dual": False,
        "cross_encoder": False,
    },

    "graph": {
        "graph": True,
        "dual": False,
        "cross_encoder": False,
    },

    "role-aware": {
        "graph": True,
        "dual": True,
        "cross_encoder": False,
    },

    "full": {
        "graph": True,
        "dual": True,
        "cross_encoder": True,
    }

}


def configure_pipeline(exp):
    pipeline = RAGPipeline(
        generator_type="ollama",
        use_graph=exp["graph"],
        use_cross_encoder=exp["cross_encoder"],
        use_dual_embeddings=exp["dual"],
    )
    pipeline.initialize()
    return pipeline


def save_csv(results, experiment):

    output = RESULT_DIR / f"{experiment}_predictions.csv"

    with open(output, "w", newline="", encoding="utf8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()

        writer.writerows(results)


def save_summary(results, experiment):

    summary = {

        "exact_match":
        sum(x["em"] for x in results) / len(results),

        "token_f1":
        sum(x["f1"] for x in results) / len(results),

        "rouge_l":
        sum(x["rouge_l"] for x in results) / len(results),

        "latency_ms":
        sum(x["latency_ms"] for x in results) / len(results),

    }

    with open(
        RESULT_DIR / f"{experiment}_metrics.json",
        "w",
        encoding="utf8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    return summary


def main():

    dataset = load_dataset(DATASET)

    all_results = []

    for name, experiment in EXPERIMENTS.items():

        print()
        print("=" * 70)
        print(name.upper())
        print("=" * 70)

        experiment_start = time.time()

        pipeline = configure_pipeline(experiment)

        print(
            f"Running -> "
            f"Graph={experiment['graph']}, "
            f"Dual={experiment['dual']}, "
            f"CrossEncoder={experiment['cross_encoder']}"
        )

        results = evaluate(
            pipeline,
            dataset,
            name,
        )

        save_csv(results, name)

        metrics = save_summary(results, name)

        elapsed = time.time() - experiment_start

        print(
            f"\n{name.upper()} completed in "
            f"{elapsed/60:.2f} minutes.\n"
        )

        metrics["experiment"] = name

        all_results.append(metrics)

    print()
    print("=" * 70)

    for x in all_results:

        print(x)

    print("=" * 70)


if __name__ == "__main__":

    main()