# Brain-in-a-Box

## Finetuning an AI Model for Counseling Conversations

This script outlines the process of finetuning a machine learning model using the Unsloth framework and related tools. The goal is to adapt a pretrained model to provide counseling-style responses by combining two datasets: a mental health counseling dataset and a PubMedQA dataset.

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

---

## Overview

The script:
- Installs and configures required Python packages.
- Loads and preprocesses two datasets to align formats.
- Finetunes a pretrained LLaMA 3.2 model using Unsloth with LoRA (Low-Rank Adaptation) for efficient training.
- Visualizes key dataset statistics like context and response lengths.
- Saves the resulting model for downstream use in conversational AI tasks.

---

## Setup

### Package Installation

The script installs and updates key dependencies:
- Unsloth for efficient finetuning.
- Transformers and TRL for model handling and training.
- Visualization tools like seaborn and matplotlib.

Commands:
```bash
pip install unsloth seaborn
pip install --upgrade --no-cache-dir unsloth     git+https://github.com/huggingface/transformers.git     git+https://github.com/huggingface/trl.git
```

### Memory Optimization

CUDA memory is optimized using:
```python
import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()
```

---

## Data Preparation

### Loading Data

Two datasets are loaded:
1. **Mental Health Counseling Dataset**: A JSON file containing counseling conversations.
2. **PubMedQA Dataset**: A parquet file with medical question-answer pairs.

### Transformation

The PubMedQA dataset is transformed to align with the counseling dataset format:
- `Question` → `Context`
- `Long Answer` → `Response`

### Filtering

Data is filtered based on context and response lengths to ensure compatibility with the model:
- Context length ≤ 1500 characters.
- Response length ≤ 4000 characters.

### Visualization

Distribution of context and response lengths is visualized using histograms (Seaborn).

---

## Model Configuration

### Pretrained Model

The LLaMA 3.2 model is loaded with:
- Maximum sequence length: 5020 tokens.
- Support for 4-bit quantization for memory efficiency.

### LoRA Configuration

The model uses LoRA for finetuning:
- **LoRA rank**: 16
- **Alpha**: 16
- **Target modules**: Includes `q_proj`, `k_proj`, and others.

### Prompt Formatting

Prompts are structured as:
```plaintext
### Input:
{Context}

### Response:
{Response}
```

---

## Training

### Training Configuration

The model is finetuned using:
- Learning rate: 3e-4
- Batch size: 2 (effective batch size maintained with gradient accumulation steps: 32)
- Epochs: 40
- Memory optimization: Gradient checkpointing and mixed-precision training.

### Training Execution

Training logs include loss values for each step, ensuring real-time performance monitoring.

---

## Visualization

During preprocessing, the following distributions are visualized:
- Context Length Distribution
- Filtered Context Length Distribution
- Response Length Distribution
- Filtered Response Length Distribution

---

## Dependencies

### Python Libraries
- `unsloth`, `torch`, `transformers`, `trl`
- `seaborn`, `matplotlib`, `pandas`, `numpy`

### Hardware
- GPU: Tesla V100-PCIE-16GB

---

## Known Issues

### Memory Warnings
- May occur due to CUDA or TensorFlow incompatibility. Recommended to restart the kernel.

### Jupyter Notebook Integration
- If `tqdm` warnings occur, update `ipywidgets` as instructed.

### Dataset Errors
- Some rows in the PubMedQA dataset may fail to process due to unexpected data formats.

---

## Usage

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Update paths to datasets in the script.

3. Execute the script in a Python environment with GPU support.

4. Monitor the training loss and save the model for deployment.

---

## Additional Scripts

### `Llama-32-image-generation-modified.ipynb`

This script demonstrates the use of the LLaMA-3.2 Vision model to perform fine-tuning, inference, and customization for generating text or visual outputs based on multi-modal inputs (images and text). It leverages the Hugging Face Transformers and Diffusers libraries, along with PyTorch, for scalable and efficient deep learning workflows.

#### Features
- Multi-modal Processing: Integrates image and text data for generating coherent outputs.
- Custom Model Implementation: Includes custom classes extending `MllamaForConditionalGeneration` for advanced functionality.
- Pipeline Integration: Compatible with Hugging Face pipelines for streamlined workflows.
- On-the-Fly Customization: Supports dynamic model and pipeline customization.
- Resource Optimization: Provides configurations to manage GPU/CPU memory during training and inference.

#### Requirements
- Python 3.10+
- Hugging Face token authentication.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
