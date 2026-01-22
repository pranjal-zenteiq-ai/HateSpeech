#!/usr/bin/env python3
"""Test script to run lr.py training"""

import pandas as pd
from lr import train_lr

# Load dataset
print("Loading dataset...")
df = pd.read_csv("HateSpeech__Unbalanced.csv")

# Rename columns to match expected format
df = df.rename(columns={'text': 'Content', 'label': 'Label'})

# Use a sample for faster testing (5000 samples)
df_sample = df.sample(n=min(5000, len(df)), random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Sample shape: {df_sample.shape}")
print(f"Columns: {df_sample.columns.tolist()}")
print(f"\nLabel distribution:\n{df_sample['Label'].value_counts()}")

# Train model
print("\n" + "="*70)
print("TRAINING LOGISTIC REGRESSION MODEL (TF-IDF)")
print("="*70)
train_lr(df_sample, save_path="models/lr_model_test.joblib")
print("\nTraining complete!")
