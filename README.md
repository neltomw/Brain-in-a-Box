# Brain-in-a-Box
# Finetuning an AI Model for Counseling Conversations

This repository provides a guide to fine-tuning a machine learning model using the Unsloth framework and related tools. The objective is to adapt a pretrained model to deliver counseling-style responses by integrating two datasets: a mental health counseling dataset and the PubMedQA dataset.

---

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
  - [Package Installation](#package-installation)
  - [Memory Optimization](#memory-optimization)
- [Data Preparation](#data-preparation)
  - [Loading Data](#loading-data)
  - [Transformation](#transformation)
  - [Filtering](#filtering)
  - [Visualization](#visualization)
- [Model Configuration](#model-configuration)
  - [Pretrained Model](#pretrained-model)
  - [LoRA Configuration](#lora-configuration)
  - [Prompt Formatting](#prompt-formatting)
- [Training](#training)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [Known Issues](#known-issues)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This script:
- Installs and configures required Python packages.
- Loads and preprocesses two datasets for format alignment.
- Fine-tunes a pretrained LLaMA 3.2 model using Unsloth with LoRA (Low-Rank Adaptation) for efficient training.
- Visualizes key dataset statistics, such as context and response lengths.
- Saves the fine-tuned model for use in conversational AI tasks.

---

## Setup

### Package Installation
Install the required dependencies:
```bash
pip install unsloth seaborn
pip install --upgrade --no-cache-dir unsloth \
    git+https://github.com/huggingface/transformers.git \
    git+https://github.com/huggingface/trl.git
```

### Memory Optimization
Optimize CUDA memory to prevent memory fragmentation:
```python
import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()
```

---

## Data Preparation

### Loading Data
- **Mental Health Counseling Dataset:** JSON file containing counseling conversations.
- **PubMedQA Dataset:** Parquet file with medical Q&A pairs.

### Transformation
Align PubMedQA dataset with the counseling dataset format:
- `Question` → `Context`
- `Long Answer` → `Response`

### Filtering
Filter data to ensure compatibility with the model:
- Context length ≤ 1500 characters.
- Response length ≤ 4000 characters.

### Visualization
Visualize context and response length distributions with histograms using Seaborn.

---

## Model Configuration

### Pretrained Model
- **Model:** LLaMA 3.2
- **Maximum Sequence Length:** 5020 tokens
- **Quantization:** 4-bit for memory efficiency.

### LoRA Configuration
- **Rank:** 16
- **Alpha:** 16
- **Target Modules:** Includes `q_proj`, `k_proj`, etc.

### Prompt Formatting
Training prompts follow this structure:
```plaintext
### Input:
{Context}

### Response:
{Response}
```

---

## Training

### Training Configuration
- **Learning Rate:** 3e-4
- **Batch Size:** 2 (effective batch size: 32 with gradient accumulation).
- **Epochs:** 40
- **Memory Optimization:** Gradient checkpointing and mixed precision training.

### Training Execution
Logs include loss values for real-time monitoring.

---

## Visualization
During preprocessing, visualize the following distributions:
- Context Length
- Filtered Context Length
- Response Length
- Filtered Response Length

---

## Dependencies

### Python Libraries
- `unsloth`, `torch`, `transformers`, `trl`
- `seaborn`, `matplotlib`, `pandas`, `numpy`

### Hardware
- **GPU:** Tesla V100-PCIE-16GB (or equivalent)

---

## Known Issues
1. **Memory Warnings:** Restart the kernel if CUDA memory issues arise.
2. **Jupyter Notebook Integration:** Update `ipywidgets` if `tqdm` warnings occur.
3. **Dataset Errors:** Some PubMedQA rows may fail to process due to format inconsistencies.

---

## Usage
1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Update dataset paths in the script.

3. Run the script in a GPU-enabled Python environment.

4. Monitor training and save the model for deployment.

---

## Contributing
We welcome contributions! Please open an issue or submit a pull request with suggestions or improvements.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
