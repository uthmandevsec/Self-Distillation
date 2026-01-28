# On-Policy Self-Distillation

This is TRL based code  for reproducing the paper "Self-Distillation Enables Continual Learning".

 All experiments can be run with a single H200 GPU. Other setups may require refactoring and/or changing model sizes.

## Abstract
Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce **Self-Distillation Fine-Tuning (SDFT)**, a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.

##  Setup

### 1. Clone the repository

```bash
git clone https://github.com/Continual-Intelligence/Self-Distillation.git
cd Self-Distillation
```

### 2. Set up a virtual environment

Using **conda**:

```bash
conda create -n distillation python=3.12
conda activate distillation
```

Using **venv**:

```bash
python3.12 -m venv distillation
source distillation/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Usage

```bash
python main.py \
  --model_name <path_to_model> \
  --output_dir <output_path> \
  --learning_rate 2e-5 \
  --num_train_epochs 1
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | Qwen/Qwen2.5-7B-Instruct | Path to pretrained model |
| `--output_dir` | - | Output directory for checkpoints |
| `--learning_rate` | `2e-5` | Learning rate |
| `--num_train_epochs` | `1` | Number of training epochs |
| `--num_prompts_per_batch` | `32` | Prompts per batch |
| `--ref_model_mixup_alpha` | `0.01` | Reference model mixup alpha |
| `--seed` | `42` | Random seed |
