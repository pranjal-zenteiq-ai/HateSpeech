#!/usr/bin/env python3
"""Test script to run XGB.py training"""

import pandas as pd
from XGB import train_xgb

# Load dataset
print("Loading dataset...")
df = pd.read_csv("HateSpeech__Unbalanced.csv")

# Rename columns to match expected format
df = df.rename(columns={'text': 'Content', 'label': 'Label'})

# Use a sample for faster testing (2000 samples)
df_sample = df.sample(n=min(2000, len(df)), random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Sample shape: {df_sample.shape}")
print(f"\nLabel distribution:\n{df_sample['Label'].value_counts()}")

# Train model
print("\n" + "="*70)
print("TRAINING XGBOOST WITH SENTENCE EMBEDDINGS")
print("="*70)
train_xgb(df_sample, save_path="models/xgb_model_test.joblib")
print("\nTraining complete!")
