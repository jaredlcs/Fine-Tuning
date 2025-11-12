"""
Part 1: Fine-tune and Classify Prompt Injection
Dataset: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
Model: https://huggingface.co/protectai/deberta-v3-base-prompt-injection
"""

from datasets import load_dataset
import numpy as np
from utils import (
    print_header, 
    load_model_and_tokenizer, 
    prepare_datasets, 
    create_trainer,
    display_results,
    save_model_and_results
)

print_header("PART 1: Fine-tuning on Safe-Guard Dataset")

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset("xTRam1/safe-guard-prompt-injection")
print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

# Load model & tokenizer
model_name = "protectai/deberta-v3-base-prompt-injection"
model, tokenizer = load_model_and_tokenizer(model_name, num_labels=2)

# Prepare datasets
train_data, test_data = prepare_datasets(dataset, tokenizer, text_column='text', label_column='label')

# Create trainer
trainer = create_trainer(
    model=model,
    train_data=train_data,
    test_data=test_data,
    output_dir="./results/part1",
    batch_size=32,
    epochs=3,
    learning_rate=2e-5
)

# Train
print_header("TRAINING", device_info=False)
trainer.train()

# Evaluate
print_header("FINAL EVALUATION", device_info=False)
preds = trainer.predict(test_data)
pred_labels = np.argmax(preds.predictions, axis=1)
true_labels = preds.label_ids

# Display and save results
metrics = display_results(true_labels, pred_labels, part_name="Part 1")
save_model_and_results(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset['test'],
    true_labels=true_labels,
    pred_labels=pred_labels,
    metrics=metrics,
    model_dir="./models/part1_finetuned",
    results_prefix="part1",
    text_column='text',
    cleanup_dir="./results/part1"
)

print_header("PART 1 COMPLETE!", device_info=False)

