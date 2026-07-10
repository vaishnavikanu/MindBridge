import csv
import json
from pathlib import Path

from tqdm import tqdm

from evaluation.metrics.exact_match import exact_match
from evaluation.metrics.token_f1 import token_f1
from evaluation.metrics.rouge_l import rouge_l
#from evaluation.config import EXPERIMENTS
from src.pipeline.rag_pipeline import RAGPipeline
# from evaluation.metrics.recall import (
#     recall_at_1,
#     recall_at_3,
#     recall_at_5,
# )

# from evaluation.metrics.mrr import reciprocal_rank

DATA_DIR = Path("evaluation/data")
RESULT_DIR = Path("evaluation/results")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_dataset(name):

    with open(
        DATA_DIR / f"{name}.json",
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)



def evaluate(
    pipeline,
    dataset,
    experiment,
):

    results = []

    
    print()
    print("=" * 70)
    print(f"Running Experiment : {experiment}")
    print(f"Total Queries      : {len(dataset)}")
    print("=" * 70)

    checkpoint_file = (
        RESULT_DIR /
        f"{experiment}_checkpoint.csv"
    )

    progress = tqdm(
    dataset,
    total=len(dataset),
    desc=f"{experiment}",
    unit="query",
    dynamic_ncols=True,
)

    for sample in progress:

        prediction = pipeline.query(

            query=sample["question"],

            role=sample["role"],

            user_id="evaluation",

        )
        retrieved = prediction.retrieved_texts
        candidates = [x.chunk.text for x in (prediction.retrieved_candidates or [])]

        em = exact_match(

            prediction.response,

            sample["answer"],

        )

        f1 = token_f1(

            prediction.response,

            sample["answer"],

        )

        rouge = rouge_l(

            prediction.response,

            sample["answer"],

        )

        # r1 = recall_at_1(

        #     retrieved,

        #     sample["answer"]

        # )

        # r3 = recall_at_3(

        #     retrieved,

        #     sample["answer"]

        # )

        # r5 = recall_at_5(

        #     retrieved,

        #     sample["answer"]

        # )

        # mrr = reciprocal_rank(

        #     retrieved,

        #     sample["answer"]

        # )

        avg_em = (
            sum(r["em"] for r in results) + em
        ) / (len(results) + 1)

        avg_f1 = (
            sum(r["f1"] for r in results) + f1
        ) / (len(results) + 1)

        progress.set_postfix({

            "Done":
                f"{len(results)+1}/{len(dataset)}",

            "Avg EM":
                f"{avg_em:.3f}",

            "Avg F1":
                f"{avg_f1:.3f}",

            "Latency":
                f"{prediction.latency_ms:.0f}ms",

        })

        results.append(

            {
                
                "experiment": experiment,

                "id": sample["id"],

                "role": sample["role"],

                "question": sample["question"],

                "ground_truth": sample["answer"],

                "prediction": prediction.response,

                "prompt": prediction.prompt or "",

                "em": em,

                "f1": f1,

                "rouge_l": rouge,

                "latency_ms": prediction.latency_ms,

                "retrieved_1": retrieved[0] if len(retrieved) > 0 else "",

                "retrieved_2": retrieved[1] if len(retrieved) > 1 else "",

                "retrieved_3": retrieved[2] if len(retrieved) > 2 else "",

                "retrieved_4": retrieved[3] if len(retrieved) > 3 else "",

                "retrieved_5": retrieved[4] if len(retrieved) > 4 else "",

                "retrieved_context":

                    "\n\n".join(retrieved),

                "retrieved_candidates":
                    "\n\n".join(candidates),

                #  "recall@1": r1,

                # "recall@3": r3,

                # "recall@5": r5,

                # "mrr": mrr,

            }

        )

        #
        # Autosave every 10 queries
        #
        if len(results) % 25 == 0:

            with open(
                checkpoint_file,
                "w",
                newline="",
                encoding="utf8",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=results[0].keys(),
                )

                writer.writeheader()

                writer.writerows(results)

            print(
                f"\n✓ Checkpoint saved "
                f"({len(results)}/{len(dataset)})"
            )

    return results


def save_predictions(
    results,
    experiment,
    ):

    output = RESULT_DIR / f"{experiment}_predictions.csv"

    with open(

        output,

        "w",

        newline="",

        encoding="utf-8",

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=results[0].keys(),

        )

        writer.writeheader()

        writer.writerows(results)

    print(f"Saved predictions -> {output}")


def save_summary(
    results,
    experiment,
    ):

    summary = {

        "exact_match":

            sum(r["em"] for r in results)

            / len(results),

        "token_f1":

            sum(r["f1"] for r in results)

            / len(results),

        "rouge_l":

            sum(r["rouge_l"] for r in results)

            / len(results),

        "latency_ms":

            sum(r["latency_ms"] for r in results)

            / len(results),

        # "recall@1":
        #     sum(r["recall@1"] for r in results)/len(results),

        # "recall@3":
        #     sum(r["recall@3"] for r in results)/len(results),

        # "recall@5":
        #     sum(r["recall@5"] for r in results)/len(results),

        # "mrr":
        #     sum(r["mrr"] for r in results)/len(results),


    }

    with open(

        RESULT_DIR / f"{experiment}_metrics.json",

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            summary,

            f,

            indent=4,

        )

    print()

    print(summary)

    print()


def main():

    experiment = "full"

    #settings = EXPERIMENTS[experiment]

    pipeline = RAGPipeline()

    pipeline.initialize()

    #
    # Change this when switching datasets
    #
    dataset_name = "counselchat_unique"

    dataset = load_dataset(

        dataset_name

    )

    results = evaluate(

        pipeline,

        dataset,

        experiment,
    )

    save_predictions(

        results

    )

    save_summary(

        results

    )


if __name__ == "__main__":

    main()