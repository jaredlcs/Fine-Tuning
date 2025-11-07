"""
Utility functions for the classification projects
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import json

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """
    Plot and optionally save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_metrics(y_true, y_pred, labels):
    """
    Print detailed classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
    """
    print(classification_report(y_true, y_pred, target_names=labels))

def save_results(results_dict, filepath):
    """
    Save results to JSON file
    
    Args:
        results_dict: Dictionary containing results
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {filepath}")

def load_and_prepare_dataset(dataset_name, text_column, label_column, split='test', max_samples=None):
    """
    Load and prepare dataset for classification
    
    Args:
        dataset_name: HuggingFace dataset name
        text_column: Name of the text column
        label_column: Name of the label column
        split: Dataset split to use
        max_samples: Maximum number of samples (None for all)
    
    Returns:
        texts: List of text samples
        labels: List of labels
    """
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    texts = dataset[text_column]
    labels = dataset[label_column]
    
    return texts, labels

def batch_predict(classifier, texts, batch_size=32):
    """
    Run predictions in batches
    
    Args:
        classifier: HuggingFace pipeline
        texts: List of texts to classify
        batch_size: Batch size for predictions
    
    Returns:
        predictions: List of prediction results
    """
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_preds = classifier(batch)
        predictions.extend(batch_preds)
    return predictions
