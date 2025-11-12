"""
Part 2: Fine-tune and Classify Harder Prompt Injection
Dataset: https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection
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

print_header("PART 2: Fine-tuning on SPML Chatbot Dataset (HARDER)")

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")
if 'test' not in dataset:
    print("Splitting into train/test (80/20)...")
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

# Load model & tokenizer
model_name = "protectai/deberta-v3-base-prompt-injection"
model, tokenizer = load_model_and_tokenizer(model_name, num_labels=2)

# Prepare datasets
train_data, test_data = prepare_datasets(dataset, tokenizer, text_column='prompt', label_column='label')

# Create trainer
trainer = create_trainer(
    model=model,
    train_data=train_data,
    test_data=test_data,
    output_dir="./results/part2",
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
metrics = display_results(true_labels, pred_labels, part_name="Part 2")
save_model_and_results(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset['test'],
    true_labels=true_labels,
    pred_labels=pred_labels,
    metrics=metrics,
    model_dir="./models/part2_finetuned",
    results_prefix="part2",
    text_column='prompt',
    cleanup_dir="./results/part2"
)

# Additional context
print("\n=== Why Part 2 is Harder ===")
print("- More sophisticated adversarial attacks")
print("- Conversational context makes detection harder")
print("- Subtle manipulation vs obvious injections")

print_header("PART 2 COMPLETE!", device_info=False)

