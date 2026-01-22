import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
from BC import BinaryClassifier


def main():
    print("Loading dataset...")
    df = pd.read_csv("HateSpeech__Balanced.csv")
    
    # Rename columns to match expected format
    df = df.rename(columns={"text": "Content", "label": "Label"})
    
    # Clean data
    df = df.dropna(subset=["Content", "Label"])
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
    )
    df = df[df["Label"].isin(["0", "1"])]
    df["Label"] = df["Label"].astype(int)
    
    # Use 50k stratified sample for evaluation
    df_test = (
        df.groupby("Label", group_keys=False)
        .apply(lambda x: x.sample(n=min(25000, len(x)), random_state=42))
        .reset_index(drop=True)
    )
    
    print(f"\nTest set size: {len(df_test)} (50K STRATIFIED SAMPLE)")
    print(f"Label distribution:\n{df_test['Label'].value_counts()}\n")
    
    # Load ensemble
    print("Loading ensemble models...")
    bc = BinaryClassifier(
        lr_model_path="models/lr_model_balanced.joblib",
        xgb_model_path="models/xgb_semantic_balanced.joblib",
        semantic_weight=0.6,
        lexical_weight=0.4,
        low_threshold=0.4,
        high_threshold=0.5
    )
    joblib.dump(bc, "models/binary_classifier_ensemble.joblib")
    
    # Get predictions and probabilities
    print("Running predictions...")
    texts = df_test["Content"].astype(str).tolist()
    y_true = df_test["Label"].values
    y_proba = bc.predict_proba(texts)

    # Tri-level predictions from probabilities
    y_tri = np.zeros(len(y_proba), dtype=int)
    y_tri[(y_proba > bc.low) & (y_proba <= bc.high)] = 1   # MAYBE_HATE
    y_tri[y_proba > bc.high] = 2                           # HATE
    # Binary predictions for evaluation:
    # Only HIGH confidence hate counts as HATE
    y_pred = (y_tri == 2).astype(int)

    
    # ===========================
    # COMPREHENSIVE METRICS
    # ===========================
    print("\n" + "="*70)
    print("ENSEMBLE CLASSIFIER EVALUATION (Binary: SAFE vs HATE)")
    print("="*70)
    
    print("\n[1] CONFUSION MATRIX")
    print("-" * 70)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"\nBreakdown:")
    print(f"  True Negatives (TN):  {cm[0,0]:,}")
    print(f"  False Positives (FP): {cm[0,1]:,}")
    print(f"  False Negatives (FN): {cm[1,0]:,}")
    print(f"  True Positives (TP):  {cm[1,1]:,}")
    
    print("\n[2] CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(y_true, y_pred, digits=4, target_names=["SAFE", "HATE"]))
    
    print("\n[3] OVERALL METRICS")
    print("-" * 70)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    print("\n[4] THRESHOLD ANALYSIS")
    print("-" * 70)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred_thresh)
        prec = precision_score(y_true, y_pred_thresh)
        rec = recall_score(y_true, y_pred_thresh)
        f1_thresh = f1_score(y_true, y_pred_thresh)
        print(f"Threshold {threshold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1_thresh:.4f}")
    
    print("\n[5] PROBABILITY DISTRIBUTION")
    print("-" * 70)
    print(f"Min probability: {y_proba.min():.4f}")
    print(f"Max probability: {y_proba.max():.4f}")
    print(f"Mean probability: {y_proba.mean():.4f}")
    print(f"Median probability: {np.median(y_proba):.4f}")
    print(f"Std Dev: {y_proba.std():.4f}")
    
    print("\n[6] SAMPLE PREDICTIONS (First 10 Examples)")
    print("-" * 70)
    for i in range(min(10, len(df_test))):
        text = texts[i][:80]
        actual = "HATE" if y_true[i] == 1 else "SAFE"
        predicted = "HATE" if y_pred[i] == 1 else "SAFE"
        prob = y_proba[i]
        match = "✓" if y_pred[i] == y_true[i] else "✗"
        print(f"{match} [{prob:.4f}] Pred={predicted:5s} | True={actual:5s} | {text}...")
    
    print("\n[7] MISCLASSIFICATION ANALYSIS")
    print("-" * 70)
    mismatches = np.where(y_pred != y_true)[0]
    print(f"Total misclassifications: {len(mismatches):,} / {len(y_true):,} ({len(mismatches)/len(y_true)*100:.2f}%)")
    
    false_positives = np.where((y_pred == 1) & (y_true == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y_true == 1))[0]
    
    print(f"False Positives (predicted HATE, actual SAFE): {len(false_positives):,}")
    print(f"False Negatives (predicted SAFE, actual HATE): {len(false_negatives):,}")
    
    if len(false_positives) > 0:
        print("\n  Sample False Positives (Top 3 by probability):")
        fp_indices = false_positives[np.argsort(y_proba[false_positives])[-3:]]
        for idx in fp_indices:
            print(f"    Prob={y_proba[idx]:.4f}: {texts[idx][:100]}...")
    
    if len(false_negatives) > 0:
        print("\n  Sample False Negatives (Top 3 by probability):")
        fn_indices = false_negatives[np.argsort(y_proba[false_negatives])[:3]]
        for idx in fn_indices:
            print(f"    Prob={y_proba[idx]:.4f}: {texts[idx][:100]}...")
    
    print("\n[8] ENSEMBLE WEIGHTS")
    print("-" * 70)
    print(f"Semantic (XGBoost): {bc.semantic_weight*100:.1f}%")
    print(f"Lexical (TF-IDF LR): {bc.lexical_weight*100:.1f}%")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
