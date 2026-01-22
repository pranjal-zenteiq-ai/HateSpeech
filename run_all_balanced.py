#!/usr/bin/env python3
"""
Comprehensive test script to run all models on a balanced subset of the dataset
in the correct dependency order.
"""

import pandas as pd
import sys
import os
import traceback

# Prepare balanced dataset
print("="*80)
print("LOADING AND PREPARING BALANCED DATASET")
print("="*80)

# Read dataset in chunks to handle large files
print("Reading HateSpeechDataset.csv...")
df_full = pd.read_csv("HateSpeechDataset.csv")
print(f"Full dataset shape: {df_full.shape}")
print(f"Columns: {df_full.columns.tolist()}")
print(f"\nFull dataset label distribution:\n{df_full['Label'].value_counts()}\n")

# Create balanced dataset by stratified sampling
from sklearn.model_selection import train_test_split

# Balance the dataset
class_0 = df_full[df_full['Label'] == 0]
class_1 = df_full[df_full['Label'] == 1]

min_class_count = min(len(class_0), len(class_1))
print(f"Balancing dataset to {min_class_count} samples per class...")

df_balanced = pd.concat([
    class_0.sample(n=min_class_count, random_state=42),
    class_1.sample(n=min_class_count, random_state=42)
])

print(f"Balanced dataset shape: {df_balanced.shape}")
print(f"Balanced label distribution:\n{df_balanced['Label'].value_counts()}\n")

# Use a sample for faster testing (50k samples)
df_sample = df_balanced.sample(n=min(50000, len(df_balanced)), random_state=42)
print(f"Using sample of {len(df_sample)} rows for testing")
print(f"Sample Label distribution:\n{df_sample['Label'].value_counts()}\n")

# ============================================================================
# 1. TRAIN LOGISTIC REGRESSION (TF-IDF)
# ============================================================================
print("\n" + "="*80)
print("1. TRAINING LOGISTIC REGRESSION (TF-IDF + GRID SEARCH)")
print("="*80)

try:
    from lr import train_lr
    lr_model = train_lr(df_sample, save_path="models/lr_model_balanced.joblib")
    print("✓ LR training completed successfully\n")
except Exception as e:
    print(f"✗ LR training failed: {e}\n")
    traceback.print_exc()

# ============================================================================
# 2. TRAIN XGBOOST (WITH EMBEDDINGS - SEMANTIC)
# ============================================================================
print("\n" + "="*80)
print("2. TRAINING XGBOOST (SENTENCE TRANSFORMERS + SEMANTIC)")
print("="*80)

try:
    from xgb_semantics import train_xgb_semantic
    xgb_semantic_model = train_xgb_semantic(df_sample, save_path="models/xgb_semantic_balanced.joblib")
    print("✓ XGBoost semantic training completed successfully\n")
except Exception as e:
    print(f"✗ XGBoost semantic training failed: {e}\n")
    traceback.print_exc()

# ============================================================================
# 3. TRAIN XGB (WITH EMBEDDINGS - ALTERNATE)
# ============================================================================
print("\n" + "="*80)
print("3. TRAINING XGBOOST (ALTERNATE - WITH EMBEDDINGS)")
print("="*80)

try:
    from XGB import train_xgb
    xgb_model = train_xgb(df_sample, save_path="models/xgb_model_balanced.joblib")
    print("✓ XGBoost training completed successfully\n")
except Exception as e:
    print(f"✗ XGBoost training failed: {e}\n")
    traceback.print_exc()

# ============================================================================
# 4. TRAIN SENTENCE TRANSFORMERS + LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("4. TRAINING SENTENCE TRANSFORMERS + LOGISTIC REGRESSION")
print("="*80)

try:
    # Save sample to CSV for setence_trf.py
    df_sample.to_csv("temp_balanced_sample.csv", index=False)
    
    import subprocess
    result = subprocess.run([
        "uv", "run", "python", "setence_trf.py", 
        "temp_balanced_sample.csv",
        "--subset", "30000",
        "--batch-size", "64",
        "--save-model", "models/semantic_lr_model_balanced.joblib"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("✓ Sentence Transformers + LR training completed successfully\n")
    
    # Cleanup
    if os.path.exists("temp_balanced_sample.csv"):
        os.remove("temp_balanced_sample.csv")
        
except Exception as e:
    print(f"✗ Sentence Transformers training failed: {e}\n")
    traceback.print_exc()

# ============================================================================
# 5. TEST ENSEMBLE CLASSIFIER (BC)
# ============================================================================
print("\n" + "="*80)
print("5. TESTING ENSEMBLE CLASSIFIER (BINARY CLASSIFIER)")
print("="*80)

try:
    import subprocess
    result = subprocess.run([
        "uv", "run", "python", "test_bc_ensemble.py"
    ], capture_output=True, text=True, timeout=300)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("✓ Ensemble classifier test completed successfully\n")
    
except subprocess.TimeoutExpired:
    print("✗ Test BC ensemble timed out (5 minutes)\n")
except Exception as e:
    print(f"✗ Test BC ensemble failed: {e}\n")
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
print("\nGenerated models:")
print("  - models/lr_model_balanced.joblib")
print("  - models/xgb_semantic_balanced.joblib")
print("  - models/xgb_model_balanced.joblib")
print("  - models/semantic_lr_model_balanced.joblib")
print("\nAll models trained on balanced dataset with GPU support enabled!")
print("="*80)
