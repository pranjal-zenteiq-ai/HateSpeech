import argparse
import sys
import pandas as pd

# Import LR training
try:
    from lr import train_lr
    has_lr = True
except Exception as e:
    print("Warning: LR not available:", e)
    has_lr = False

# Import semantic XGBoost training
try:
    from xgb_semantics import train_xgb_semantic
    has_xgb = True
except Exception as e:
    print("Warning: XGB not available:", e)
    has_xgb = False

# Import final binary classifier (ensemble)
try:
    from BC import BinaryClassifier
    has_bc = True
except Exception as e:
    print("Warning: BC not available:", e)
    has_bc = False


def clean_labels(df):
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
    )
    df = df[df["Label"].isin(["0", "1"])]
    df["Label"] = df["Label"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Hate Speech Classification")

    parser.add_argument(
        "data",
        help="CSV file (training/inference) or raw text (bc mode)"
    )

    parser.add_argument(
        "--model", "-m",
        choices=["lr", "xgb", "bc"],
        required=True,
        help="lr=train TF-IDF LR | xgb=train semantic XGB | bc=final ensemble inference"
    )

    parser.add_argument("--save-model", default=None)
    parser.add_argument("--sample", type=int, default=None)

    # BC-specific arguments
    parser.add_argument("--lr-model", default="models/lr_tfidf.joblib")
    parser.add_argument("--xgb-model", default="models/xgb_semantic.joblib")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    # ----------------------------
    # Training modes
    # ----------------------------
    if args.model in ["lr", "xgb"]:
        print("Loading:", args.data)
        df = pd.read_csv(args.data)

        if "Content" not in df.columns or "Label" not in df.columns:
            print("CSV must contain 'Content' and 'Label'")
            print("Found:", df.columns.tolist())
            sys.exit(1)

        df = df.dropna(subset=["Content", "Label"])
        df = clean_labels(df)

        if args.sample:
            df = df.sample(n=min(args.sample, len(df)), random_state=42)
            print("Using sample:", len(df))

        if "Content_int" in df.columns:
            df = df.drop(columns=["Content_int"])

        print("\n=== Label Distribution ===")
        print(df["Label"].value_counts())
        print("Total samples:", len(df))

        if args.model == "lr":
            if not has_lr:
                print("LR module not found")
                sys.exit(1)
            train_lr(df, save_path=args.save_model)

        else:
            if not has_xgb:
                print("XGB module not found")
                sys.exit(1)
            train_xgb_semantic(df, save_path=args.save_model)

    # ----------------------------
    # Final binary classifier (ensemble)
    # ----------------------------
    else:
        if not has_bc:
            print("BC module not found")
            sys.exit(1)

        # Determine input texts
        if args.data.endswith(".csv"):
            df = pd.read_csv(args.data)
            if "Content" not in df.columns:
                print("CSV must contain 'Content' column for bc mode")
                sys.exit(1)
            texts = df["Content"].astype(str).tolist()
        else:
            # Single raw text
            texts = [args.data]

        bc = BinaryClassifier(
            lr_model_path=args.lr_model,
            xgb_model_path=args.xgb_model
        )

        probs = bc.predict_proba(texts)
        preds = bc.predict(texts, low=args.threshold)

        print("\n=== Binary Classification Output ===")
        for text, pred, prob in zip(texts, preds, probs):
            print(f"[{pred}] prob={prob:.4f} → {text[:120]}...")
        print("\n[TRI-LEVEL DISTRIBUTION]")
        print("-" * 70)
        print(f"SAFE:        {(y_tri == 0).sum():,}")
        print(f"MAYBE_HATE:  {(y_tri == 1).sum():,}")
        print(f"HATE:        {(y_tri == 2).sum():,}")



if __name__ == "__main__":
    main()
