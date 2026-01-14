import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load model
model_path = "models/xgb_model.joblib"
model, tfidf = joblib.load(model_path)

# Load and clean data
df = pd.read_csv("HateSpeechDataset.csv")
df = df.dropna(subset=["Content", "Label"]) 
# normalize labels similar to main.clean_labels
if "Label" in df.columns:
    df["Label"] = df["Label"].astype(str)
    df["Label"] = df["Label"].str.strip()
    df["Label"] = df["Label"].str.replace(".0", "", regex=False)
    df = df[df["Label"].isin(["0", "1"])].copy()
    df["Label"] = df["Label"].astype(int)
else:
    raise RuntimeError("Label column missing in CSV")

X = df["Content"].astype(str)
y = df["Label"].astype(int)

# Recreate the same train/test split used in training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_vec = tfidf.transform(X_test)

preds = model.predict(X_test_vec)
probs = model.predict_proba(X_test_vec)[:, 1]

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, preds))

print("\n=== Classification Report ===")
print(classification_report(y_test, preds, digits=4))

try:
    auc = roc_auc_score(y_test, probs)
    print("\nROC AUC:", auc)
except Exception as e:
    print("\nROC AUC could not be computed:", e)
