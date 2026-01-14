import pandas as pd
from XGB import train_xgb

# Load and clean
df = pd.read_csv("HateSpeechDataset.csv")
df = df.dropna(subset=["Content", "Label"]) 
# normalize labels
if "Label" in df.columns:
    df["Label"] = df["Label"].astype(str)
    df["Label"] = df["Label"].str.strip()
    df["Label"] = df["Label"].str.replace(".0", "", regex=False)
    df = df[df["Label"].isin(["0","1"])].copy()
    df["Label"] = df["Label"].astype(int)
else:
    raise RuntimeError("Label column missing in CSV")

# sample for quick run
sample_n = min(200, len(df))
print(f"Using sample of {sample_n} rows for a quick run")
df = df.sample(n=sample_n, random_state=42)

# Run training (will print confusion matrix, classification report, ROC AUC)
train_xgb(df, save_path="models/xgb_emb_sample.joblib")
