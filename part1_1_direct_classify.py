"""
Part 1_1: Direct Prompt Injection Classification
Dataset: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
Model: https://huggingface.co/protectai/deberta-v3-base-prompt-injection

first iteration, default hyperparameters
"""
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


SEED=42
set_seed(SEED)
print(f"Using seed: {SEED}")

def tokenize_input(data):
    tokens = tokenizer(data["text"], truncation=True, padding=True, max_length=128)
    return tokens

# Load the model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection")
model = AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection")
# Load the dataset
print("Loading dataset...")
dataset = load_dataset("xTRam1/safe-guard-prompt-injection")

# Prepare test data
tokenized_data = dataset.map(tokenize_input, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
g = torch.Generator()
g.manual_seed(SEED)

training_data = DataLoader(
    tokenized_data["train"],
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator,
    generator=g,
)
test_data = DataLoader(
    tokenized_data["test"],
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Training on device: {device}")
model.train()

loss_history = []
final_loss = None
for epoch in range(num_epochs):
    for idx, batch in enumerate(training_data):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        final_loss = loss.item()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item()}")
        loss_history.append(loss.item())

print("Evaluating model...")
model.eval()

all_preds = []
all_labels = []

for batch in test_data:
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch["labels"].cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="binary")
recall = recall_score(all_labels, all_preds, average="binary")
f1 = f1_score(all_labels, all_preds, average="binary")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

total_errors = fp + fn
error_rate = total_errors / len(all_labels)
print(f"True negatives: {tn:4d}")
print(f"False positives: {fp:4d}")
print(f"False negatives: {fn:4d}")
print(f"True positives: {tp:4d}")

metrics = {
    "loss": [final_loss],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "true_negatives": [tn],
    "false_positives": [fp],
    "false_negatives": [fn],
    "true_positives": [tp],
    "total_samples": [len(all_labels)],
    "error rate": [error_rate],
}

df_metrics = pd.DataFrame(metrics)

name = "part1_1_deberta_prompt_injections_metrics.csv"
df_metrics.to_csv(name, index=False)
print(f"Metrics saved to {name}")

print(df_metrics)

model_save_path = "part1_1_deberta_prompt_injection_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")
