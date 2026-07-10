from typing import List


def reciprocal_rank(retrieved_chunks: List[str], ground_truth: str):

    gt = ground_truth.lower()

    for i, chunk in enumerate(retrieved_chunks):

        if gt[:100] in chunk.lower():
            return 1.0 / (i + 1)

    return 0.0