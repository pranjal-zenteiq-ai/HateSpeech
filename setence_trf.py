import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# -----------------------------
# Config
# -----------------------------
DEFAULT_SUBSET_SIZE = 80000
RANDOM_STATE = 42


# -----------------------------
# Utilities
# -----------------------------
def clean_dataset(df):
    if "Content" not in df.columns or "Label" not in df.columns:
        raise ValueError("Dataset must contain 'Content' and 'Label' columns")

    df = df.dropna(subset=["Content", "Label"]).copy()
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
    )

    df = df[df["Label"].isin(["0", "1"])]
    df["Label"] = df["Label"].astype(int)

    if "Content_int" in df.columns:
        df = df.drop(columns=["Content_int"])

    return df


def stratified_subset(df, n_samples):
    if n_samples >= len(df):
        return df

    return (
        df.groupby("Label", group_keys=False)
        .apply(lambda x: x.sample(
            int(n_samples * len(x) / len(df)),
            random_state=RANDOM_STATE
        ))
        .reset_index(drop=True)
    )


# -----------------------------
# Main training logic
# -----------------------------
def main(args):
    print("Loading dataset:", args.data)
    df = pd.read_csv(args.data)

    df = clean_dataset(df)

    print("\nLabel distribution (full dataset):")
    print(df["Label"].value_counts())

    if args.subset:
        print(f"\nUsing stratified subset of size {args.subset}")
        df = stratified_subset(df, args.subset)

    print("\nFinal dataset size:", len(df))

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Content"].astype(str).tolist(),
        df["Label"].values,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["Label"].values
    )

    print("\nTrain size:", len(X_train_text))
    print("Test size:", len(X_test_text))

    # -----------------------------
    # Sentence Transformer
    # -----------------------------
    print("\nLoading sentence transformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nEncoding training data...")
    X_train_emb = encoder.encode(
        X_train_text,
        batch_size=args.batch_size,
        show_progress_bar=True
    )

    print("\nEncoding test data...")
    X_test_emb = encoder.encode(
        X_test_text,
        batch_size=args.batch_size,
        show_progress_bar=True
    )

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    base_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1
    )

    if args.grid_search:
        print("\nRunning GridSearchCV (small & safe)...")

        param_grid = {
            "C": [0.1, 0.5, 1.0, 2.0]
        }

        gs = GridSearchCV(
            base_model,
            param_grid,
            scoring="f1",
            cv=3,
            n_jobs=1,
            verbose=2
        )

        # Grid search on a smaller slice
        gs.fit(X_train_emb[:args.grid_subset], y_train[:args.grid_subset])
        model = gs.best_estimator_

        print("\nBest parameters:", gs.best_params_)

    else:
        model = base_model
        model.fit(X_train_emb, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    preds = model.predict(X_test_emb)
    probs = model.predict_proba(X_test_emb)[:, 1]

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds, digits=4))

    auc = roc_auc_score(y_test, probs)
    print("\nROC AUC:", round(auc, 4))

    # -----------------------------
    # Save
    # -----------------------------
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        joblib.dump((model, encoder), args.save_model)
        print("\nModel saved to:", args.save_model)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic Hate Speech Detection using Sentence Transformers"
    )

    parser.add_argument(
        "data",
        help="Path to HateSpeechDatasetBalanced.csv"
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=DEFAULT_SUBSET_SIZE,
        help="Stratified subset size for semantic training"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sentence encoding"
    )

    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run small GridSearchCV for Logistic Regression"
    )

    parser.add_argument(
        "--grid-subset",
        type=int,
        default=50000,
        help="Subset size used during GridSearchCV"
    )

    parser.add_argument(
        "--save-model",
        default="semantic_lr_model.joblib",
        help="Path to save trained model"
    )

    args = parser.parse_args()
    main(args)
