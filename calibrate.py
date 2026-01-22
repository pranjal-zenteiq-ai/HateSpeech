# hatespeech/calibrate.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from ensemble2 import HateSpeechEnsemble

def fit_calibrator(ensemble, X_texts, y_true, save_path):
    # compute ensemble final_prob WITHOUT calibrator
    details = ensemble.predict_debug(X_texts)
    final_probs = np.array([d["final_prob"] for d in details])

    # split into train/val for calibrator fitting
    X_train, X_val, y_train, y_val = train_test_split(
        final_probs.reshape(-1, 1), y_true, test_size=0.33, random_state=42, stratify=y_true
    )

    # fit logistic regression (Platt scaling)
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(X_train, y_train)

    # evaluate before/after on val
    before = brier_score_loss(y_val, X_val.ravel())
    after = brier_score_loss(y_val, lr.predict_proba(X_val)[:, 1])

    print(f"Brier score BEFORE calibrator: {before:.6f}")
    print(f"Brier score AFTER  calibrator: {after:.6f}")

    # save calibrator
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(lr, save_path)
    print(f"Calibrator saved to {save_path}")
    return lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file containing at least columns: text, flagged")
    parser.add_argument("--models-dir", default="models", help="models directory")
    parser.add_argument("--save", default=None, help="path to save calibrator (default models/calibrator.joblib)")
    parser.add_argument("--sample", type=int, default=None, help="optional sample size to speed debug")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "text" not in df.columns or "flagged" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'flagged' columns")

    df = df.dropna(subset=["text", "flagged"]).copy()
    df["flagged"] = df["flagged"].astype(int)

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)

    texts = df["text"].astype(str).tolist()
    labels = df["flagged"].astype(int).values

    ensemble = HateSpeechEnsemble(model_dir=args.models_dir)

    save_path = args.save or os.path.join(args.models_dir, "calibrator.joblib")
    calibrator = fit_calibrator(ensemble, texts, labels, save_path)

    # Quick check: attach calibrator to ensemble and print a few examples
    ensemble.calibrator = calibrator
    ensemble.has_calibrator = True
    print("Sample calibrated outputs (first 5):")
    for d in ensemble.predict_debug(texts[:5]):
        print({
            "text": d["text"][:120],
            "raw_final": d["final_prob"],
            "calibrated": ensemble._apply_calibrator([d["final_prob"]])[0]
        })

if __name__ == "__main__":
    main()
