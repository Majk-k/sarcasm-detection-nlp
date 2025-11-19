# Sarcasm Detection in News Headlines

This project focuses on classifying news headlines as **sarcastic** or **not sarcastic** using classical NLP preprocessing and machine-learning models.  
The pipeline includes text cleaning, lemmatization, feature extraction, model training, and evaluation.

---

## Dataset

- **Source:** Kaggle  
  *News Headlines Dataset for Sarcasm Detection*  
- **Link:** https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection  
- The dataset is stored **locally** in the `data/` folder.

You must place the following file in:
```
data/Sarcasm_Headlines_Dataset.json
```
Each record contains:
- `headline` — news headline  
- `is_sarcastic` — label (`1` = sarcastic, `0` = not sarcastic)

---

## Problem Definition

Binary classification task:
- Input: English news headline  
- Output: 1 = sarcastic, 0 = not sarcastic

---

## Approach

### 1. Preprocessing
- Expand contractions (e.g., *don’t → do not*)
- Tokenize text
- Lowercase transformation
- Strip punctuation
- Remove stopwords, while keeping **"no"** and **"not"**
- POS tagging
- Lemmatization using WordNet
- Automatic download of missing NLTK resources (first run only)

### 2. Feature Extraction
- TF-IDF vectorizer  
- Maximum vocabulary size: **5000 words**

### 3. Models Evaluated
- **Multinomial Naive Bayes**
- Hyperparameter tuning with `GridSearchCV` (macro F1)

Hyperparameter tuning was performed using `GridSearchCV` and macro F1-score.

### 4. Evaluation Metrics
- Accuracy  
- Macro F1-score  
- Classification report  

---

## Results

The final model is a Multinomial Naive Bayes classifier trained on TF-IDF features.

- Best model (GridSearchCV): `MultinomialNB(alpha=1.0, fit_prior=False)`
- Best macro F1 (cross-validation): **0.7720**
- Test accuracy: **0.7729**
- Test macro F1: **0.7708**

Example classification report (test set):

```
              precision    recall  f1-score   support

           0     0.8121    0.7744    0.7928      2997
           1     0.7279    0.7710    0.7488      2345

    accuracy                         0.7729      5342
   macro avg     0.7700    0.7727    0.7708      5342
weighted avg     0.7751    0.7729    0.7735      5342
```
---

## Project Structure

```
sarcasm-detector/
├─ data/
│  └─ Sarcasm_Headlines_Dataset.json
├─ src/
│  ├─ __init__.py
│  ├─ preprocessing.py
│  ├─ train_model.py
│  └─ predict.py
├─ models/              # saved model artifacts (created at runtime)
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## How to Run

### 1. (Optional) Create and activate a virtual environment
```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Train the model
```
python -m src.train_model
```

The script will:
- load the local dataset from `data/`
- preprocess all headlines
- train the model with GridSearchCV
- evaluate performance
- save the final model and vectorizer under `models/`

### 4. Run predictions on sample headlines
```
python -m src.predict
```

This script loads the saved model and prints predictions and probabilities for a few example headlines.

---

## Requirements

Main packages used (full list in `requirements.txt`):

- pandas  
- scikit-learn  
- nltk  
- matplotlib / seaborn (exploration)
- prettytable

---

## Notes

- The dataset is **not included** in the repository.  
  You need to download Sarcasm_Headlines_Dataset.json from Kaggle and place it in the data/ folder.
- NLTK resources are downloaded automatically if missing.
- The `models/` directory is created automatically and can be ignored in version control.
