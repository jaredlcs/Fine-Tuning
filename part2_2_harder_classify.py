"""
Part 2: Harder Prompt Injection Classification
Dataset: https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection
Model: https://huggingface.co/protectai/deberta-v3-base-prompt-injection

second iteration of part 2 with hyperparameter tuning:
    - Class balancing with weighted loss
    - Lower learning rate with warmup
    - Longer max_length for better context
    - Learning rate scheduler
    - Early stopping based on validation performance
"""
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
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

def combine_prompts(data):
    system_prompt = data["System Prompt"] or ""
    user_prompt = data["User Prompt"] or ""
    data["text"] = (
        "System Prompt: " + system_prompt + " User Prompt: " + user_prompt
    )
    return data

def tokenize_input(data):
    # Increased max_length for better context
    tokens = tokenizer(data["text"], truncation=True, max_length=256)
    tokens["labels"] = data["Prompt injection"]
    return tokens

# Load the model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection")
model = AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection")

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=SEED)

# Combine prompts
dataset = dataset.map(combine_prompts)

# Check class distribution
train_labels = dataset["train"]["Prompt injection"]
print(f"\nClass distribution in training set:")
print(f"Class 0 (Safe): {train_labels.count(0)}")
print(f"Class 1 (Injection): {train_labels.count(1)}")
print(f"Ratio: {train_labels.count(1) / train_labels.count(0):.2f}")

# Tokenize
tokenized_data = dataset.map(
    tokenize_input,
    batched=True,
    remove_columns=["System Prompt", "User Prompt", "Prompt injection", "Degree", "Source", "text"]
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
g = torch.Generator()
g.manual_seed(SEED)

batch_size = 16  # Reduced for better gradient updates
training_data = DataLoader(
    tokenized_data["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    generator=g,
)
test_data = DataLoader(
    tokenized_data["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

# Calculate class weights for balanced training
num_class_0 = train_labels.count(0)
num_class_1 = train_labels.count(1)
total = num_class_0 + num_class_1
weight_class_0 = total / (2 * num_class_0)
weight_class_1 = total / (2 * num_class_1)
class_weights = torch.tensor([weight_class_0, weight_class_1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights = class_weights.to(device)

# Lower learning rate with warmup
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 5
num_training_steps = num_epochs * len(training_data)
num_warmup_steps = num_training_steps // 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

print(f"\nTraining on device: {device}")
print(f"Total training steps: {num_training_steps}")
print(f"Warmup steps: {num_warmup_steps}")
print(f"Class weights: {class_weights.cpu().numpy()}\n")

best_f1 = 0
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for idx, batch in enumerate(training_data):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(**batch)

        # Apply class weights to loss
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, batch["labels"])

        epoch_loss += loss.item()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        if idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {idx}/{len(training_data)}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_train_loss = epoch_loss / len(training_data)
    print(f"\nEpoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")

    # Evaluate after each epoch
    print("Evaluating...")
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
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    print(f"Epoch {epoch} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}\n")

    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch
        # Save best model
        model_save_path = "best_deberta_prompt_injection_model"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"New best model saved! (F1: {best_f1:.4f})\n")

print(f"\n{'='*60}")
print(f"Training complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
print(f"{'='*60}\n")

# Final evaluation with best model
print("Final evaluation on test set...")
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
precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
recall = recall_score(all_labels, all_preds, average="binary")
f1 = f1_score(all_labels, all_preds, average="binary")

print(f"\nFinal Test Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

total_errors = fp + fn
error_rate = total_errors / len(all_labels)
print(f"\nConfusion Matrix:")
print(f"True negatives: {tn:4d}")
print(f"False positives: {fp:4d}")
print(f"False negatives: {fn:4d}")
print(f"True positives: {tp:4d}")
print(f"Error rate: {error_rate:.4f}")

metrics = {
    "loss": [avg_train_loss],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "true_negatives": [tn],
    "false_positives": [fp],
    "false_negatives": [fn],
    "true_positives": [tp],
    "total_samples": [len(all_labels)],
    "error_rate": [error_rate],
    "best_epoch": [best_epoch],
}

df_metrics = pd.DataFrame(metrics)

name = "part2_2_deberta_prompt_injections_metrics.csv"
df_metrics.to_csv(name, index=False)
print(f"\nMetrics saved to {name}")
print(df_metrics)

model_save_path = "part2_2_deberta_prompt_injection_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")