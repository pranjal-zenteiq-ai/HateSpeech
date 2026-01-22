import os
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score
)


def train_xgb_tfidf(df, save_path=None):
    # ----------------------------
    # Validation
    # ----------------------------
    required = {"text", "flagged"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'flagged' columns")

    df = df.dropna(subset=["text", "flagged"]).copy()
    df["flagged"] = df["flagged"].astype(int)

    X_text = df["text"].astype(str)
    y = df["flagged"].values

    # ----------------------------
    # Train / test split
    # ----------------------------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ----------------------------
    # TF-IDF
    # ----------------------------
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    # ----------------------------
    # Class imbalance
    # ----------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    # ----------------------------
    # XGBoost (stable params)
    # ----------------------------
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBoost (TF-IDF)...")
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

    roc_auc = roc_auc_score(y_test, probs)
    print(f"\nROC AUC: {roc_auc:.4f}")

    print("\nThreshold analysis:")
    for th in [0.3, 0.4, 0.5]:
        p = precision_score(y_test, probs >= th)
        r = recall_score(y_test, probs >= th)
        print(f"threshold={th:.2f} → precision={p:.4f}, recall={r:.4f}")

    # ----------------------------
    # Save
    # ----------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump((model, tfidf), save_path)
        print(f"\nModel saved to {save_path}")

    return model
