import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.translation.translator import Translator

translator = Translator()

translator.load()

print("=" * 60)

english = "I have been feeling depressed for several weeks."

hindi = translator.translate(
    english,
    "en",
    "hi",
)

print("English")
print(english)

print()

print("Hindi")
print(hindi)

print("=" * 60)

english_back = translator.translate(
    hindi,
    "hi",
    "en",
)

print("Back to English")
print(english_back)

print("=" * 60)

telugu = translator.translate(
    english,
    "en",
    "te",
)

print("Telugu")
print(telugu)

print("=" * 60)