# Prompt Injection Detection Project

This project implements fine-tuning and evaluation of DeBERTa models for prompt injection detection, comparing performance across different datasets.

## 📁 Project Structure

```
prompt-injection-detection/
├── README.md                    # This file
├── part1_training_evaluation.ipynb    # Fine-tuning on primary dataset
├── part2_generalization_test.ipynb    # Testing on secondary/professor dataset
└── models/                     # Generated fine-tuned models (created during training)
    └── hf_deberta_finetuned/   # Fine-tuned model directory
```

## 🎯 Project Overview

**Objective**: Fine-tune a DeBERTa model for prompt injection detection and evaluate its generalization capabilities across different datasets.

**Base Model**: `protectai/deberta-v3-base-prompt-injection`

**Datasets**:
- **Part 1**: `xTRam1/safe-guard-prompt-injection` (Training dataset)
- **Part 2**: Professor-provided dataset or `reshabhs/SPML_Chatbot_Prompt_Injection` (Generalization test)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GPU recommended for training (CPU works but is slow)
- Jupyter Notebook or VS Code with notebook support

### Installation
```bash
# Dependencies are installed automatically in notebooks
# Main packages: transformers, datasets, evaluate, scikit-learn, torch
```

### Usage

#### Step 1: Training and Part 1 Evaluation
```bash
# Open part1_training_evaluation.ipynb
# 1. Run cells 1-5 to set up and evaluate pretrained model
# 2. Set DO_TRAIN = True in cell 3 to enable training
# 3. Run cell 7 to fine-tune the model (requires GPU for reasonable speed)
# 4. Run cell 6 for comprehensive Part 1 evaluation
```

#### Step 2: Generalization Testing (Part 2)
```bash
# Open part2_generalization_test.ipynb
# 1. Update DATASET2_ID with professor's dataset name
# 2. Run all cells to evaluate both pretrained and fine-tuned models
```

## 📊 What Each Notebook Does

### Part 1: Training & Evaluation (`part1_training_evaluation.ipynb`)
- **Loads** the primary training dataset
- **Evaluates** pretrained model baseline
- **Fine-tunes** DeBERTa with advanced features:
  - Class weight balancing
  - Early stopping
  - Warmup learning rate schedule
  - Mixed precision training (FP16)
- **Comprehensive evaluation** with:
  - Confusion matrices
  - ROC/PR curves  
  - Classification reports
  - JSON summaries
- **Saves** fine-tuned model to `./models/hf_deberta_finetuned/`

### Part 2: Generalization Test (`part2_generalization_test.ipynb`)
- **Loads** secondary/professor dataset
- **Auto-detects** column names (handles various dataset formats)
- **Evaluates both**:
  - Pretrained model (baseline)
  - Fine-tuned model (from Part 1)
- **Same comprehensive metrics** as Part 1
- **Tests generalization** capability of fine-tuning

## 🔧 Configuration Options

### Key Parameters (modify in notebooks):
```python
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection"
EVAL_SAMPLE_SIZE = 200  # Set to None for full evaluation
DO_TRAIN = False        # Set to True to enable training
MAX_LENGTH = 256        # Token sequence length
```

### Training Parameters:
```python
num_train_epochs = 3
learning_rate = 2e-5
per_device_train_batch_size = 8  # Reduce if GPU memory limited
weight_decay = 0.01
warmup_ratio = 0.1
```

## 📈 Expected Results

### Part 1 (Training Dataset)
- **Baseline (Pretrained)**: ~85-90% accuracy
- **Fine-tuned**: Expected improvement of 3-8% accuracy
- **Detailed Analysis**: Confusion matrix, ROC/PR curves saved

### Part 2 (Generalization Test)
- **Tests model robustness** on unseen data distribution
- **Compares** pretrained vs fine-tuned performance
- **Reveals** if fine-tuning improves generalization

## 🗂️ Output Files

After running both notebooks, you'll have:

```
models/hf_deberta_finetuned/
├── config.json                 # Model configuration
├── pytorch_model.bin           # Fine-tuned weights
├── tokenizer.json              # Tokenizer files
├── train_log.json             # Training history
├── detailed_eval_part_1_dataset.json    # Part 1 metrics
└── detailed_eval_part_2_dataset.json    # Part 2 metrics
```

## 🎓 For Professors/Evaluators

### To Test With Your Own Dataset:
1. Open `part2_generalization_test.ipynb`
2. Change line: `DATASET2_ID = 'your-dataset-name-here'`
3. Run all cells - the code auto-detects column names

### Evaluation Metrics Provided:
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrix** with True/False Positives/Negatives
- **ROC Curve** with AUC score
- **Precision-Recall Curve** with Average Precision
- **Classification Report** by class
- **JSON Summaries** for programmatic analysis

## 🛠️ Troubleshooting

### Common Issues:

**Memory Errors**: 
- Reduce `per_device_train_batch_size` from 8 to 4 or 2
- Use gradient accumulation: `gradient_accumulation_steps = 2`

**Column Name Errors**:
- Check the debug output showing available columns
- Manually specify column names if auto-detection fails

**Slow Training**:
- Use GPU if available
- Reduce `EVAL_SAMPLE_SIZE` for quick testing

## 📝 Notes

- **GPU Recommended**: Training takes ~2-10 minutes on GPU vs hours on CPU
- **Automatic Column Detection**: Code handles various dataset formats
- **Reproducible**: All results saved as JSON for analysis
- **Extensible**: Easy to add new datasets or models

## 🤝 Academic Use

This project demonstrates:
- ✅ **Transfer Learning** with transformer models
- ✅ **Fine-tuning** best practices
- ✅ **Model Evaluation** methodologies
- ✅ **Generalization Testing** across datasets
- ✅ **Class Imbalance** handling
- ✅ **Comprehensive Metrics** reporting

Perfect for machine learning courses focusing on NLP and model evaluation.