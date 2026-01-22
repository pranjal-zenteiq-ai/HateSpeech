#!/usr/bin/env python3
"""Test script to run xgb_semantics.py training"""

import pandas as pd
from xgb_semantics import train_xgb_semantic

# Load dataset
print("Loading dataset...")
df = pd.read_csv("HateSpeechDataset.csv")

# Use a sample for faster testing (1500 samples)
df_sample = df.sample(n=min(1500, len(df)), random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Sample shape: {df_sample.shape}")
print(f"\nLabel distribution:\n{df_sample['Label'].value_counts()}")

# Train model
print("\n" + "="*70)
print("TRAINING XGBOOST WITH SEMANTIC EMBEDDINGS")
print("="*70)
train_xgb_semantic(df_sample, save_path="models/xgb_semantic_test.joblib")
print("\nTraining complete!")
