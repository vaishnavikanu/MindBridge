import json
import re
from pathlib import Path

from tqdm import tqdm

from src.generation.generator import get_generator

INPUT = Path("evaluation/data/clinician_candidates.json")
OUTPUT = Path("evaluation/data/clinician_qa.json")

PROMPT = """
You are an experienced psychiatrist creating examination questions for psychiatry residents.

Below is a passage from an authoritative mental health guideline.

Generate EXACTLY ONE clinician-level question.

Rules:

1. The question MUST require clinical knowledge.

2. Ask ONLY about:
- diagnosis
- differential diagnosis
- symptoms
- assessment
- screening
- treatment
- medication
- psychotherapy
- prognosis
- risk factors
- management
- clinical decision making

3. NEVER ask about:
- article title
- page title
- website
- purpose of the article
- purpose of the passage
- navigation
- disclaimer
- organization names
- author
- journal
- URL

4. The answer MUST exist in the passage.

Return ONLY valid JSON.

{{
    "question":"..."
}}

PASSAGE

----------------

{context}

----------------
"""


def extract_reference_answer(text: str, max_sentences: int = 3):

    REMOVE = [

        "You Are Here",
        "Home",
        "Skip navigation",
        "Official websites",
        "Resources",
        "Find an Expert",
        "Clinical Trials",
        "Journal Articles",
        "Patient Handouts",
        "Related Issues",
        "Statistics",
        "Disclaimers",
        "Also in",
        "URL of this page",

    ]

    #
    # Remove navigation keywords
    #
    for item in REMOVE:
        text = text.replace(item, "")

    #
    # Remove extra blank lines
    #
    text = re.sub(r"\n\s*\n+", "\n", text)

    #
    # Split into sentences
    #
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    good = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 40:
            continue

        lower = sentence.lower()

        #
        # Skip navigation / metadata sentences
        #
        if any(x in lower for x in [

            "copyright",
            "disclaimer",
            "journal article",
            "clinical trial",
            "find an expert",
            "related issues",
            "patient handout",
            "statistics",
            "medlineplus",
            "national library of medicine",
            "official website",

        ]):
            continue

        good.append(sentence)

    return " ".join(good[:max_sentences])


def main():

    generator = get_generator("ollama")

    with open(INPUT, "r", encoding="utf-8") as f:

        chunks = json.load(f)

    dataset = []

    skipped = 0

    for sample in tqdm(chunks):

        text = sample["text"]

        if text.startswith("You Are Here"):
            continue

        if text.startswith("Skip navigation"):
            continue

        if text.startswith("Official websites"):
            continue


        prompt = PROMPT.format(

            context=sample["text"]

        )

        try:

            response = generator.generate(prompt)

            #
            # Sometimes Ollama adds markdown
            #
            response = response.replace("```json", "")
            response = response.replace("```", "")
            response = response.strip()

            qa = json.loads(response)

            question = qa["question"].lower()

            BAD = [

                "purpose",

                "article",

                "passage",

                "website",

                "organization",

                "title",

                "journal",

                "url",

            ]

            if any(word in question for word in BAD):
                continue

            answer = extract_reference_answer(

                sample["text"]

            )

            if len(answer) < 40:

                skipped += 1

                continue

            dataset.append(

                {

                    "id": str(len(dataset)),

                    "role": "clinician",

                    "question": qa["question"],

                    "answer": answer,

                    "evidence": sample["text"],

                    "document": sample["document"],

                    "chunk_id": sample["chunk_id"],

                }

            )

        except Exception:

            skipped += 1

            continue

    with open(

        OUTPUT,

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            dataset,

            f,

            indent=2,

            ensure_ascii=False,

        )

    print("\n===========================")
    print(f"Generated : {len(dataset)}")
    print(f"Skipped   : {skipped}")
    print("===========================\n")


if __name__ == "__main__":

    main()