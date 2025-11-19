"""
Training script for sarcasm detection model.

This script:
- loads the Kaggle dataset with news headlines,
- preprocesses headlines using src.preprocessing,
- vectorizes text with TF-IDF,
- trains a Multinomial Naive Bayes model with GridSearchCV,
- evaluates the model on a held-out test set,
- saves the trained model and vectorizer to the 'models/' directory.
"""

import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
)

from src.preprocessing import lemmatize_text


def load_data() -> pd.DataFrame:
    """
    Load the sarcasm dataset from a local JSON file.

    Expects file:
        data/Sarcasm_Headlines_Dataset.json

    Returns:
        DataFrame with at least ['headline', 'is_sarcastic'].
    """
    # Path to this file: src/train_model.py
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "Sarcasm_Headlines_Dataset.json"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {data_path}\n"
            f"Make sure you downloaded 'Sarcasm_Headlines_Dataset.json' "
            f"from Kaggle and placed it in the 'data/' folder."
        )

    # JSON Lines format
    df = pd.read_json(data_path, lines=True)

    # Drop 'article_link' as it is not used in this project
    if "article_link" in df.columns:
        df = df.drop(columns=["article_link"])

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'clean_headline' column using lemmatization-based preprocessing.

    Args:
        df: input DataFrame with 'headline' column.

    Returns:
        DataFrame with additional 'clean_headline' column.
    """
    df = df.copy()
    df["clean_headline"] = df["headline"].apply(lemmatize_text)
    return df


def train_and_evaluate(
    df: pd.DataFrame,
    max_features: int = 5000,
    test_size: float = 0.2,
    random_state: int = 1,
):
    """
    Train a Multinomial Naive Bayes model with TF-IDF features and evaluate it.

    Args:
        df: preprocessed DataFrame with 'clean_headline' and 'is_sarcastic'.
        max_features: maximum vocabulary size for TF-IDF.
        test_size: fraction of data used for test split.
        random_state: random seed for reproducibility.

    Returns:
        (best_model, vectorizer, metrics_dict)
    """
    X_text = df["clean_headline"]
    y = df["is_sarcastic"]

    # Train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Multinomial Naive Bayes with hyperparameter tuning
    base_model = MultinomialNB()
    param_grid = {
        "alpha": [0.01, 0.1, 0.5, 1.0, 5.0],
        "fit_prior": [True, False],
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best macro F1 (CV):", grid_search.best_score_)

    best_model = MultinomialNB(**grid_search.best_params_)
    best_model.fit(X_train, y_train)

    # Evaluation on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=4)

    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
    }

    print("\n=== Evaluation on test set ===")
    print(report)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    return best_model, vectorizer, metrics


def save_artifacts(
    model,
    vectorizer,
    output_dir: str = "models",
    model_name: str = "multinomial_nb_tfidf",
) -> None:
    """
    Save trained model and vectorizer to the given directory using pickle.

    Args:
        model: fitted sklearn model.
        vectorizer: fitted vectorizer.
        output_dir: directory path where artifacts will be saved.
        model_name: base name used for output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = Path(output_dir) / f"{model_name}.pkl"
    vectorizer_path = Path(output_dir) / f"{model_name}_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved to:      {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")


def main() -> None:
    """
    End-to-end training entry point:

    - load data,
    - preprocess text,
    - train and evaluate model,
    - save trained artifacts.
    """
    print("Loading data from local JSON file...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    print("Preprocessing headlines...")
    df = preprocess_dataframe(df)

    print("Training model...")
    model, vectorizer, metrics = train_and_evaluate(df)

    print("Saving artifacts to 'models/'...")
    save_artifacts(model, vectorizer, output_dir="models")


if __name__ == "__main__":
    main()
