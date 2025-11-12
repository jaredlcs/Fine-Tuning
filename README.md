# Fine-Tuning DeBERTa for Prompt Injection Detection

This repository contains fine-tuning pipelines for the DeBERTa model to detect prompt injection attacks using two different datasets.

## 📁 Project Structure

```
├── finetune_deberta_safeguard.py    # Fine-tune on Safe-Guard dataset
├── finetune_deberta_chatbot.py      # Fine-tune on SPML Chatbot dataset
├── utils.py                          # Shared utility functions
├── requirements.txt                  # Python dependencies
├── models/                           # Saved fine-tuned models
│   ├── part1_finetuned/             # Safe-Guard model
│   └── part2_finetuned/             # Chatbot model
└── results/                          # Training results and metrics
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

**Safe-Guard Dataset (Basic):**
```bash
python finetune_deberta_safeguard.py
```

**SPML Chatbot Dataset (Advanced):**
```bash
python finetune_deberta_chatbot.py
```

## 📊 Datasets

### 1. Safe-Guard Dataset
- **Source:** [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection)
- **Description:** Basic prompt injection detection
- **Model Output:** `models/part1_finetuned/`

### 2. SPML Chatbot Dataset
- **Source:** [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection)
- **Description:** More sophisticated adversarial attacks in conversational context
- **Model Output:** `models/part2_finetuned/`

## 🏗️ Model Architecture

- **Base Model:** [protectai/deberta-v3-base-prompt-injection](https://huggingface.co/protectai/deberta-v3-base-prompt-injection)
- **Task:** Binary classification (injection vs. safe)
- **Max Sequence Length:** 128 tokens

## ⚙️ Training Configuration

- **Learning Rate:** 2e-5
- **Batch Size:** 32 (training), 64 (evaluation)
- **Epochs:** 3
- **Optimization:** AdamW with fp16 (if GPU available)
- **Metric:** F1-score (for best model selection)

## 📈 Results

Results are automatically saved to `results/`:
- `part1_results.csv` / `part2_results.csv` - Per-sample predictions
- `part1_metrics.csv` / `part2_metrics.csv` - Aggregate metrics (accuracy, precision, recall, F1)

## 🛠️ Utilities

The `utils.py` module provides:
- `load_model_and_tokenizer()` - Load pretrained models
- `prepare_datasets()` - Tokenize and prepare data
- `create_trainer()` - Configure Trainer with standard settings
- `display_results()` - Format and display metrics
- `save_model_and_results()` - Save models and CSV outputs

## 💻 Hardware Requirements

- **GPU:** Recommended (CUDA-enabled)
- **RAM:** 8GB minimum
- **VRAM:** 4GB+ recommended
- **Training Time:** ~5-15 minutes per dataset (with GPU)

## 📝 Notes

- Training checkpoints are automatically cleaned up after completion
- Models are saved with both weights and tokenizer
- GPU usage of 90-100% at 60-80°C is normal and expected during training