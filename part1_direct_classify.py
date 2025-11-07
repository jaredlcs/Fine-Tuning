"""
Part 1: Direct Prompt Injection Classification
Dataset: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
Model: https://huggingface.co/protectai/deberta-v3-base-prompt-injection
"""

from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load the model
print("Loading model...")
# TODO: Load the deberta-v3-base-prompt-injection model
classifier = None  # pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection")

# Load the dataset
print("Loading dataset...")
# TODO: Load the safe-guard-prompt-injection dataset
dataset = None  # load_dataset("xTRam1/safe-guard-prompt-injection")

# Prepare test data
# TODO: Extract text and labels from dataset
texts = []
true_labels = []

# Run predictions
print("Running predictions...")
predictions = []
# TODO: Run classifier on texts

# Calculate metrics
# TODO: Calculate accuracy, precision, recall, F1-score
print("\n=== Part 1 Results ===")
print(f"Total samples: {len(texts)}")
print(f"Accuracy: TODO")
print(f"Precision: TODO")
print(f"Recall: TODO")
print(f"F1-Score: TODO")

# Show confusion matrix
# TODO: Display confusion matrix

# Save results
# TODO: Save results to CSV or JSON
