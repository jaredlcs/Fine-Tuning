"""
Mini Toy Project: Find Good Classifier
Requirements:
- Hugging Face model with less than 3B parameters
- Simple code with stats and accuracy
- Research and findings
"""

from transformers import pipeline
from datasets import load_dataset

# Model candidates (< 3B parameters):
# 1. distilbert-base-uncased-finetuned-sst-2-english (~67M params)
# 2. bert-base-uncased (~110M params)
# 3. roberta-base (~125M params)
# 4. deberta-v3-base (~184M params)
# 5. TODO: Add more candidates

def test_classifier(model_name, dataset_name="imdb", num_samples=100):
    """
    Test a classifier model on a dataset
    
    Args:
        model_name: HuggingFace model identifier
        dataset_name: Dataset to test on
        num_samples: Number of samples to test
    """
    print(f"\n=== Testing {model_name} ===")
    
    # Load model
    # TODO: Load classifier
    classifier = None  # pipeline("text-classification", model=model_name)
    
    # Load dataset
    # TODO: Load and prepare dataset
    dataset = None  # load_dataset(dataset_name)
    
    # Run predictions
    # TODO: Run predictions and calculate metrics
    
    print(f"Model: {model_name}")
    print(f"Parameters: TODO")
    print(f"Accuracy: TODO")
    print(f"Speed: TODO")
    print(f"Memory: TODO")
    
    return {
        "model": model_name,
        "accuracy": 0.0,
        "speed": 0.0,
        "memory": 0.0
    }

# Test multiple models
results = []
models_to_test = [
    # TODO: Add model names to test
    # "distilbert-base-uncased-finetuned-sst-2-english",
    # "bert-base-uncased",
]

for model_name in models_to_test:
    result = test_classifier(model_name)
    results.append(result)

# Summary
print("\n=== SUMMARY ===")
print("Best model: TODO")
print("Findings: TODO - Write your research findings")

# Save results
# TODO: Save comparison table
