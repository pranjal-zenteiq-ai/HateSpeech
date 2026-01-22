# import argparse
# import sys
# import pandas as pd

# from lr import train_lr
# from xgb_semantics import train_xgb_semantic   


# def normalize_labels(df):
#     df["moderation_detected"] = (
#         df["moderation_detected"]
#         .astype(str)
#         .str.strip()
#         .str.lower()
#         .map({
#             "true": 1,
#             "false": 0,
#             "1": 1,
#             "0": 0,
#             "yes": 1,
#             "no": 0
#         })
#     )
#     df = df.dropna(subset=["moderation_detected"])
#     df["moderation_detected"] = df["moderation_detected"].astype(int)
#     return df


# def main():
#     parser = argparse.ArgumentParser(description="Hate Speech Model Training")

#     parser.add_argument("data", help="CSV with columns: text, moderation_detected")
#     parser.add_argument("--model", choices=["lr", "xgb"], required=True)
#     parser.add_argument("--save-model", default=None)
#     parser.add_argument("--sample", type=int, default=None)

#     args = parser.parse_args()

#     df = pd.read_csv(args.data)

#     if not {"text", "moderation_detected"}.issubset(df.columns):
#         print("CSV must contain: text, moderation_detected")
#         sys.exit(1)

#     df = df.dropna(subset=["text", "moderation_detected"])
#     df = normalize_labels(df)

#     if args.sample:
#         df = df.sample(n=min(args.sample, len(df)), random_state=42)
#         print("Using sample:", len(df))

#     print("\nLabel distribution:")
#     print(df["moderation_detected"].value_counts())

#     if args.model == "lr":
#         train_lr(df, save_path=args.save_model)
#     else:
#         train_xgb_semantic(df, save_path=args.save_model)


# if __name__ == "__main__":
#     main()
import argparse
import sys
import pandas as pd

from lr import train_lr
from xgb_semantics2 import train_xgb_semantic_v2
from xgb_tfidf import train_xgb_tfidf



def main():
    parser = argparse.ArgumentParser(description="Hate Speech Model Training")

    parser.add_argument(
        "data",
        help="Merged moderation CSV (must contain text, flagged, has_error)"
    )
    parser.add_argument(
        "--model",
        choices=["lr", "xgb","xgb_tfidf"],
        required=True
    )
    parser.add_argument(
        "--save-model",
        default=None
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    required_cols = {"text", "flagged", "has_error"}
    if not required_cols.issubset(df.columns):
        print("Dataset must contain:", required_cols)
        print("Found:", df.columns.tolist())
        sys.exit(1)

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print("Using sample size:", len(df))

    print("\nDataset overview:")
    print("Total rows:", len(df))
    print("Flagged distribution (raw):")
    print(df["flagged"].value_counts(dropna=False))
    print("Error rows:", df["has_error"].sum())

    if args.model == "lr":
        train_lr(df, save_path=args.save_model)
    elif args.model == "xgb_tfidf":
        train_xgb_tfidf(df, save_path=args.save_model)
    else:
        train_xgb_semantic_v2(df, save_path=args.save_model)


if __name__ == "__main__":
    main()
