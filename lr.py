import os
import numpy as np
import joblib

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def train_lr(df, save_path=None):
    # ----------------------------
    # Validation
    # ----------------------------
    required = {"text", "flagged"}
    if not required.issubset(df.columns):
        raise ValueError("DataFrame must contain 'text' and 'flagged' columns")

    df = df.dropna(subset=["text", "flagged"]).copy()
    df["flagged"] = df["flagged"].astype(int)

    X = df["text"].astype(str)
    y = df["flagged"].values

    print("Label distribution:", Counter(y))

    # ----------------------------
    # Train / test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ----------------------------
    # Pipeline
    # ----------------------------
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=40000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode"
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear"
            )
        )
    ])

    print("Training Logistic Regression (TF-IDF)...")
    model.fit(X_train, y_train)

    # ----------------------------
    # Evaluation
    # ----------------------------
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds, digits=4))

    auc = roc_auc_score(y_test, probs)
    print(f"\nROC AUC: {auc:.4f}")

    # ----------------------------
    # Save
    # ----------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(model, save_path)
        print(f"\nModel saved to {save_path}")

    return model
