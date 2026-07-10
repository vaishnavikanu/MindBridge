from typing import List


def recall_at_k(retrieved_chunks: List[str], ground_truth: str, k: int = 5):
    """
    Computes Recall@K.

    retrieved_chunks : list of retrieved chunk texts
    ground_truth     : reference answer
    """

    gt = ground_truth.lower()

    retrieved = retrieved_chunks[:k]

    for chunk in retrieved:
        if gt[:100] in chunk.lower():
            return 1.0

    return 0.0


def recall_at_1(retrieved_chunks, ground_truth):
    return recall_at_k(retrieved_chunks, ground_truth, 1)


def recall_at_3(retrieved_chunks, ground_truth):
    return recall_at_k(retrieved_chunks, ground_truth, 3)


def recall_at_5(retrieved_chunks, ground_truth):
    return recall_at_k(retrieved_chunks, ground_truth, 5)