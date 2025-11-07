"""
Part 2: Harder Prompt Injection Classification
Dataset: https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection
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
# TODO: Load the SPML_Chatbot_Prompt_Injection dataset
dataset = None  # load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")

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
print("\n=== Part 2 Results ===")
print(f"Total samples: {len(texts)}")
print(f"Accuracy: TODO")
print(f"Precision: TODO")
print(f"Recall: TODO")
print(f"F1-Score: TODO")

# Show confusion matrix
# TODO: Display confusion matrix

# Compare with Part 1
print("\n=== Comparison ===")
print("Part 2 is harder because: TODO - analyze why")

# Save results
# TODO: Save results to CSV or JSON
