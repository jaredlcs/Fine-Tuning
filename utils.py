"""
Utility functions for the classification projects
"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def print_header(text, device_info=True):
    """Print formatted section header"""
    print("=" * 60)
    print(text)
    print("=" * 60)
    if device_info:
        print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")


def compute_metrics(pred):
    """
    Compute classification metrics for Trainer
    
    Args:
        pred: Predictions object from Trainer
    
    Returns:
        Dictionary of metrics
    """
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(pred.label_ids, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average='binary')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def load_model_and_tokenizer(model_name, num_labels=2):
    """
    Load model and tokenizer
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels
    
    Returns:
        model, tokenizer
    """
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


def create_tokenize_function(tokenizer, text_column='text', max_length=128):
    """
    Create a tokenization function for dataset mapping
    
    Args:
        tokenizer: HuggingFace tokenizer
        text_column: Name of the text column in dataset
        max_length: Maximum sequence length
    
    Returns:
        Tokenization function
    """
    def tokenize(examples):
        return tokenizer(examples[text_column], padding='max_length', truncation=True, max_length=max_length)
    return tokenize


def prepare_datasets(dataset, tokenizer, text_column='text', label_column='label', max_length=128):
    """
    Tokenize and prepare train/test datasets
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: HuggingFace tokenizer
        text_column: Name of text column
        label_column: Name of label column
        max_length: Maximum sequence length
    
    Returns:
        train_data, test_data
    """
    print("Tokenizing...")
    tokenize_fn = create_tokenize_function(tokenizer, text_column, max_length)
    
    train_data = dataset['train'].map(tokenize_fn, batched=True, remove_columns=[text_column])
    test_data = dataset['test'].map(tokenize_fn, batched=True, remove_columns=[text_column])
    
    # Rename label column to 'labels' for Trainer
    if label_column != 'labels':
        train_data = train_data.rename_column(label_column, "labels")
        test_data = test_data.rename_column(label_column, "labels")
    
    return train_data, test_data


def create_trainer(model, train_data, test_data, output_dir, batch_size=32, epochs=3, learning_rate=2e-5):
    """
    Create a Trainer with standard configuration
    
    Args:
        model: Model to train
        train_data: Training dataset
        test_data: Test dataset
        output_dir: Output directory for results
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Trainer instance
    """
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=epochs,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            fp16=torch.cuda.is_available(),
            report_to="none",
        ),
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics
    )
    return trainer


def display_results(true_labels, pred_labels, part_name="Part"):
    """
    Display evaluation results in a formatted way
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        part_name: Name of the part/experiment
    
    Returns:
        Dictionary of metrics
    """
    acc = accuracy_score(true_labels, pred_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    cm = confusion_matrix(true_labels, pred_labels)
    
    print(f"\n=== {part_name} Results ===")
    print(f"Total samples: {len(true_labels)}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall: {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"Error Rate: {(1-acc):.4f} ({(1-acc)*100:.2f}%)")
    
    print("\n=== Confusion Matrix ===")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'error_rate': 1-acc,
        'confusion_matrix': cm
    }


def save_model_and_results(model, tokenizer, dataset, true_labels, pred_labels, metrics, 
                           model_dir, results_prefix, text_column='text', cleanup_dir=None):
    """
    Save model, tokenizer, and results to files
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        dataset: Test dataset
        true_labels: True labels
        pred_labels: Predicted labels
        metrics: Dictionary of metrics
        model_dir: Directory to save model
        results_prefix: Prefix for result files (e.g., 'part1')
        text_column: Name of text column in dataset
        cleanup_dir: Directory to clean up (optional)
    """
    # Save model
    print(f"\nSaving model to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save predictions
    pd.DataFrame({
        'text': [s[text_column] for s in dataset],
        'true_label': true_labels,
        'predicted_label': pred_labels,
        'correct': true_labels == pred_labels
    }).to_csv(f'results/{results_prefix}_results.csv', index=False)
    
    # Save metrics
    cm = metrics['confusion_matrix']
    pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'error_rate', 'tn', 'fp', 'fn', 'tp'],
        'value': [
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1'], 
            metrics['error_rate'], 
            cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        ]
    }).to_csv(f'results/{results_prefix}_metrics.csv', index=False)
    
    print(f"Results saved to results/{results_prefix}_results.csv and results/{results_prefix}_metrics.csv")
    
    # Clean up checkpoints
    if cleanup_dir:
        print(f"\nCleaning up training checkpoints...")
        shutil.rmtree(cleanup_dir, ignore_errors=True)
        print(f"Checkpoints removed (final model saved in {model_dir}/)")

