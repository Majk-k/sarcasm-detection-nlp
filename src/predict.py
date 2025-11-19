"""
Inference utilities for the sarcasm detection model.

This module:
- loads the trained model and vectorizer from disk,
- exposes helper functions to predict sarcasm for new headlines,
- contains a simple CLI-style example in the __main__ block.
"""

import pickle
from pathlib import Path
from typing import Iterable

from src.preprocessing import lemmatize_text


def load_model_and_vectorizer(
    model_dir: str = "models",
    model_name: str = "multinomial_nb_tfidf",
):
    """
    Load trained model and vectorizer from the given directory.

    Args:
        model_dir: directory where .pkl files are stored.
        model_name: base name of model/vectorizer files.

    Returns:
        (model, vectorizer)
    """
    model_path = Path(model_dir) / f"{model_name}.pkl"
    vectorizer_path = Path(model_dir) / f"{model_name}_vectorizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_headlines(
    headlines: Iterable[str],
    model_dir: str = "models",
    model_name: str = "multinomial_nb_tfidf",
) -> list[int]:
    """
    Predict sarcasm labels for a list of raw headlines.

    Args:
        headlines: iterable of raw headline strings.
        model_dir: directory where model artifacts are stored.
        model_name: base name of model/vectorizer files.

    Returns:
        List of predicted labels (1 = sarcastic, 0 = not sarcastic).
    """
    model, vectorizer = load_model_and_vectorizer(
        model_dir=model_dir,
        model_name=model_name,
    )

    # Preprocess headlines using the same logic as during training
    clean_texts = [lemmatize_text(h) for h in headlines]
    X = vectorizer.transform(clean_texts)
    preds = model.predict(X)
    return list(int(p) for p in preds)


def predict_headlines_with_proba(
    headlines: Iterable[str],
    model_dir: str = "models",
    model_name: str = "multinomial_nb_tfidf",
):
    """
    Predict sarcasm labels and probabilities for a list of raw headlines.

    Returns:
        List of tuples: (predicted_label, prob_non_sarcastic, prob_sarcastic)
    """
    model, vectorizer = load_model_and_vectorizer(
        model_dir=model_dir,
        model_name=model_name,
    )

    clean_texts = [lemmatize_text(h) for h in headlines]
    X = vectorizer.transform(clean_texts)
    probas = model.predict_proba(X)
    preds = model.predict(X)

    results = []
    for label, proba in zip(preds, probas):
        # proba is [p_class0, p_class1] where 1 = sarcastic
        results.append((int(label), float(proba[0]), float(proba[1])))

    return results


if __name__ == "__main__":
    # Simple manual test of inference
    sample_headlines = [
        "Oh great, another Monday morning meeting at 7 AM",
        "Local school opens new library for children",
        "I just love waiting in traffic for hours every day",
    ]

    predictions = predict_headlines_with_proba(sample_headlines)

    for text, (label, p_non, p_sarc) in zip(sample_headlines, predictions):
        label_str = "sarcastic" if label == 1 else "non-sarcastic"
        print("=" * 80)
        print(f"Headline: {text}")
        print(f"Predicted label: {label_str}")
        print(f"P(non-sarcastic) = {p_non:.3f}, P(sarcastic) = {p_sarc:.3f}")
