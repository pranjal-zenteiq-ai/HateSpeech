import os
import xgboost as xgb
import torch
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score
)


def train_xgb_semantic_v2(df, save_path=None, return_model=False):
    # ----------------------------
    # Sanity checks
    # ----------------------------
    required_cols = {"text", "flagged", "has_error"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    # ----------------------------
    # Filter usable rows
    # ----------------------------
    df = df.copy()
    df = df[df["has_error"] == False]
    df = df.dropna(subset=["flagged", "text"])

    y = df["flagged"].astype(int).values
    texts = df["text"].astype(str).tolist()

    print(f"Training samples after filtering: {len(df)}")
    print("Label distribution:")
    print(df["flagged"].value_counts())

    # ----------------------------
    # Sentence embeddings (GPU)
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("Encoding texts...")
    X = encoder.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # ----------------------------
    # Train / Val / Test split
    # ----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    # ----------------------------
    # DMatrix
    # ----------------------------
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # ----------------------------
    # XGBoost params (GPU-safe)
    # ----------------------------
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0.5,
        "lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42
    }

    print("Training XGBoost (semantic model)...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=800,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    probs = model.predict(dtest)
    preds = (probs >= 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=4))

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")

    print("\nThreshold analysis:")
    for th in [0.3, 0.4, 0.5]:
        th_preds = (probs >= th).astype(int)
        p = precision_score(y_test, th_preds)
        r = recall_score(y_test, th_preds)
        print(f"threshold={th:.2f} → precision={p:.4f}, recall={r:.4f}")

    # ----------------------------
    # Save
    # ----------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump((model, encoder), save_path)
        print(f"\nModel saved to {save_path}")

    if return_model:
        return model, encoder

    return model
