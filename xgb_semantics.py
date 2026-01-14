import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


def train_xgb_semantic(df, save_path=None):
    texts = df["Content"].astype(str).tolist()
    labels = df["Label"].astype(int).values

    print("Loading sentence transformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding sentences...")
    X = encoder.encode(texts, batch_size=64, show_progress_bar=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1
    )

    print("Training XGBoost on embeddings...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds, digits=4))

    print("\nROC AUC:", roc_auc_score(y_test, probs))

    if save_path:
        joblib.dump((model, encoder), save_path)
        print("Saved model to", save_path)

    return model
