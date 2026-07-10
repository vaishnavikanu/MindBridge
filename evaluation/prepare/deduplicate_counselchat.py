import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.embeddings.embedder import get_embedder


INPUT = Path("evaluation/data/counselchat.json")
OUTPUT = Path("evaluation/data/counselchat_unique.json")

# Final dataset size
TARGET_SIZE = 250

# Semantic duplicate threshold
SIMILARITY_THRESHOLD = 0.92


def normalize(text: str):
    text = text.lower().strip()
    text = " ".join(text.split())
    return text


def cosine(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def main():

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Original samples : {len(data)}")

    #
    # ----------------------------------------------------
    # Step 1 : Remove exact duplicates
    # ----------------------------------------------------
    #
    seen = set()
    exact_unique = []

    for sample in data:

        q = normalize(sample["question"])

        if q in seen:
            continue

        seen.add(q)
        exact_unique.append(sample)

    print(f"After exact duplicate removal : {len(exact_unique)}")

    #
    # ----------------------------------------------------
    # Step 2 : Embed questions
    # ----------------------------------------------------
    #
    embedder = get_embedder("sbert")

    questions = [
        normalize(x["question"])
        for x in exact_unique
    ]

    embeddings = embedder.embed(questions)

    #
    # ----------------------------------------------------
    # Step 3 : Remove semantic duplicates
    # ----------------------------------------------------
    #
    selected = []
    selected_embeddings = []

    for sample, emb in tqdm(
        zip(exact_unique, embeddings),
        total=len(exact_unique),
    ):

        duplicate = False

        for prev in selected_embeddings:

            sim = cosine(emb, prev)

            if sim >= SIMILARITY_THRESHOLD:
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(sample)
        selected_embeddings.append(emb)

    print(f"After semantic deduplication : {len(selected)}")

    #
    # ----------------------------------------------------
    # Step 4 : Keep first TARGET_SIZE
    # ----------------------------------------------------
    #
    if len(selected) > TARGET_SIZE:
        selected = selected[:TARGET_SIZE]

    #
    # Reassign ids
    #
    for i, sample in enumerate(selected):
        sample["id"] = str(i)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(
            selected,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("==============================")
    print(f"Original Dataset : {len(data)}")
    print(f"Unique Dataset   : {len(selected)}")
    print("==============================")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()