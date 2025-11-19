"""
Text preprocessing utilities for sarcasm detection project.

This module:
- downloads required NLTK resources (optional helper),
- defines stopwords configuration,
- provides functions for cleaning and lemmatizing headlines.
"""

import string
from typing import Iterable

import nltk
from nltk.data import find
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

REQUIRED_NLTK_PACKAGES = [
    "punkt",
    "punkt_tab",
    "wordnet",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]

def download_nltk_resources() -> None:
    resources = [
        "punkt",
        "punkt_tab",
        "wordnet",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for res in resources:
        try:
            nltk.download(res)
        except Exception as exc:
            print(f"Warning: could not download NLTK resource '{res}': {exc}")

def ensure_nltk_resources():
    """Download missing NLTK resources (only if not already installed)."""
    for pkg in REQUIRED_NLTK_PACKAGES:
        try:
            find(pkg)
        except LookupError:
            print(f"Downloading missing NLTK package: {pkg}")
            nltk.download(pkg)

ensure_nltk_resources()


_LEMMATIZER = WordNetLemmatizer()

_STOP_WORDS = set(stopwords.words("english"))
_STOP_WORDS -= {"no", "not"}


def clean_token(token: str) -> str:
    """Strip punctuation characters from the beginning and end of a token."""
    return token.strip(string.punctuation)


def get_wordnet_pos(treebank_tag: str):
    """Map POS tags from the Penn Treebank tagset to WordNet POS tags."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_text(text: str) -> str:
    """
    Clean, tokenize, filter stopwords and lemmatize a single headline.
    """
    # Optional simple normalization: replace hyphens with spaces
    text = text.replace("-", " ")

    tokens = word_tokenize(text.lower())
    cleaned_tokens = [clean_token(t) for t in tokens]
    filtered_tokens = [
        t for t in cleaned_tokens
        if t.isalpha() and t not in _STOP_WORDS
    ]

    pos_tags = pos_tag(filtered_tokens)
    lemmas = [
        _LEMMATIZER.lemmatize(token, pos=get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]

    return " ".join(lemmas)


def lemmatize_corpus(texts: Iterable[str]) -> list[str]:
    """Apply lemmatize_text to an iterable of texts."""
    return [lemmatize_text(t) for t in texts]


if __name__ == "__main__":
    download_nltk_resources()
    sample = "I can't believe this is actually true - this must be a joke!"
    print("Original:", sample)
    print("Lemmatized:", lemmatize_text(sample))
