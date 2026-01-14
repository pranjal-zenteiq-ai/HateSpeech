import os
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt


def train_lr(df, save_path=None):
    # Basic validation
    if "Content" not in df.columns or "Label" not in df.columns:
        raise ValueError("DataFrame must contain 'Content' and 'Label' columns")

    df = df.dropna(subset=["Content", "Label"]).copy()
    df = df[df["Label"].isin([0, 1])]
    df["Label"] = df["Label"].astype(int)

    X = df["Content"].astype(str)
    y = df["Label"].astype(int)

    label_counts = Counter(y)
    print("Label counts:", label_counts)

    if len(label_counts) < 2:
        raise ValueError("Need at least two classes (0 and 1) to train a classifier")

    # Stratify only if both classes have enough samples for the split
    min_class = min(label_counts.values())
    if min_class < 2:
        raise ValueError("Each class needs at least 2 samples for a train/test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))
    ])

    # Keep grid reasonably small for faster runs; adjust as needed
    param_grid = {
        "tfidf__max_features": [10000, 30000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 5.0],
        "clf__penalty": ["l2"]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise"
    )

    try:
        gs.fit(X_train, y_train)
    except Exception as e:
        print("Error during GridSearchCV fit:", e)
        raise

    print("\nBest Params:", gs.best_params_)

    model = gs.best_estimator_

    preds = model.predict(X_test)

    # Get probabilities (fall back to decision_function if needed)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # Use a sigmoid on decision function for a probability-like score
        df_scores = model.decision_function(X_test)
        probs = 1 / (1 + np.exp(-df_scores))

    print("\nConfusion Matrix")
    cm = confusion_matrix(y_test, preds)
    print(cm)

    print("\nClassification Report")
    print(classification_report(y_test, preds, digits=4))

    # ROC AUC
    try:
        auc = roc_auc_score(y_test, probs)
        print(f"\nROC AUC: {auc:.4f}")

        # Plot ROC curve and save
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plot_path = "roc_curve.png"
        plt.savefig(plot_path)
        print(f"ROC curve saved to {plot_path}")
    except Exception as e:
        print("Could not compute ROC AUC:", e)

    # Show some predictions and misclassified examples
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\nSample predictions (first 5):")
    for i in range(min(5, len(X_test))):
        print(f"{X_test.iloc[i][:120]} -> pred={preds[i]}, prob={probs[i]:.4f}, true={y_test.iloc[i]}")

    # Show a few false positives / false negatives
    mismatches = np.where(preds != y_test)[0]
    if len(mismatches) > 0:
        print(f"\nShowing up to 3 mismatches (total {len(mismatches)}):")
        for i in mismatches[:3]:
            print(f"- idx={i}, text={X_test.iloc[i][:120]}, pred={preds[i]}, prob={probs[i]:.4f}, true={y_test.iloc[i]}")
    else:
        print("\nNo mismatches on the test set.")

    # Save the model if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

    return model
