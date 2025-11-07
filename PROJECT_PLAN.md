# Module 2 Project Plan

## Overview
This project focuses on prompt injection classification using transformer models.

## Mini Toy Project
**Goal**: Find the best classifier on HuggingFace with < 3B parameters

**Tasks**:
- [ ] Research classifiers on HuggingFace (< 3B parameters)
- [ ] Test at least 3-5 different models
- [ ] Compare accuracy, speed, and memory usage
- [ ] Document findings in `results/findings.md`

**File**: `mini_toy_project.py`

---

## Part 1: Direct Prompt Injection Classification
**Dataset**: [safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection)
**Model**: [deberta-v3-base-prompt-injection](https://huggingface.co/protectai/deberta-v3-base-prompt-injection)

**Tasks**:
- [ ] Load the dataset and explore its structure
- [ ] Load the pre-trained model
- [ ] Run classification on test set
- [ ] Calculate metrics (accuracy, precision, recall, F1)
- [ ] Generate confusion matrix
- [ ] Save results to `results/part1_results.csv`

**File**: `part1_direct_classify.py`

---

## Part 2: Harder Prompt Injection Classification
**Dataset**: [SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection)
**Model**: Same as Part 1

**Tasks**:
- [ ] Load the harder dataset
- [ ] Run classification using same model
- [ ] Calculate metrics
- [ ] Compare results with Part 1
- [ ] Analyze why Part 2 is harder
- [ ] Save results to `results/part2_results.csv`

**File**: `part2_harder_classify.py`

---

## Part 3: NVIDIA Content Safety (Advanced - Due by Thanksgiving)
**Dataset**: [Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
**Model**: [llama-3.1-nemoguard-8b-content-safety](https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-content-safety)

**Note**: This is the advanced level - work on Parts 1 & 2 first!

---

## Utilities
- `utils.py` - Helper functions for metrics, visualization, and data processing
- `requirements.txt` - Python dependencies

## Results
All results will be saved in the `results/` directory.

---

## Getting Started

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Start with the Mini Toy Project to get familiar with the workflow

3. Move to Part 1 (easier dataset)

4. Complete Part 2 (harder dataset)

5. Compare and analyze results

---

## Tips
- Start with small samples to test your code
- Use batch processing for efficiency
- Document your findings as you go
- Keep code simple and focused on results
