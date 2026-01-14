import argparse
import sys
import pandas as pd

# Import LR
try:
    from lr import train_lr
    has_lr = True
except Exception as e:
    print("Warning: LR not available:", e)
    has_lr = False

# Import semantic XGBoost
try:
    from xgb_semantics import train_xgb_semantic
    has_xgb = True
except Exception as e:
    print("Warning: XGB not available:", e)
    has_xgb = False


def clean_labels(df):
    df["Label"] = df["Label"].astype(str).str.strip().str.replace(".0", "", regex=False)
    df = df[df["Label"].isin(["0", "1"])]
    df["Label"] = df["Label"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train hate-speech classifier")
    parser.add_argument("data", nargs="?", default="HateSpeechDataset.csv")
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--model", "-m", choices=["lr", "xgb"], default="xgb")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
