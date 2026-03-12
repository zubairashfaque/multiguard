# Training Guide

## Prerequisites

- GPU with >= 16GB VRAM (Tesla P100 / RTX 3090+ recommended)
- Hateful Memes dataset downloaded to `data/raw/`
- Poetry environment activated

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Ingest and preprocess data
make ingest

# 3. Train baseline model (late fusion)
make train-baseline

# 4. Train cross-attention fusion variant
make train-fusion

# 5. Knowledge distillation
make train-distill

# 6. Evaluate
make evaluate
```

## Training Configurations

| Config | Fusion | Backbone | Epochs | LR | Batch | Grad Accum |
|--------|--------|----------|:------:|:--:|:-----:|:----------:|
| baseline.yaml | Late (concat+MLP) | CLIP ViT-B/16 (512d) + RoBERTa (768d) | 15 | 1e-3 | 32 | 2 |
| fusion_ablation.yaml | Cross-attention / Tensor | CLIP ViT-B/16 (512d) + RoBERTa (768d) | 20 | 5e-4 | 32 | 2 |
| distillation.yaml | Late (student) | EfficientNet-B0 (1280d) + DistilRoBERTa (768d) | 20 | 1e-3 | 32 | 2 |

### Common Settings (All Configs)
- **Precision:** FP16 mixed precision
- **Effective batch size:** 64
- **Weight decay:** 1e-4
- **Warmup ratio:** 0.1
- **LR scheduler:** Cosine annealing
- **Max sequence length:** 77 tokens
- **Max grad norm:** 1.0

## Monitoring Training

- **W&B:** Set `WANDB_API_KEY` in `.env`, enable in config
- **MLflow:** `mlflow ui` at http://localhost:5000
- **Logs:** Check `reports/` directory

## Kaggle Training

For running the full pipeline on Kaggle, see:
- `notebooks/multiguard_kaggle_p100_full_training.ipynb` — complete training notebook
- `notebooks/KAGGLE_GUIDE.md` — setup instructions

### Hardware Used
- Tesla P100-PCIE-16GB
- FP16 mixed precision (P100 does not support BF16)

## Expected Results

| Model | AUROC | F1 | Accuracy |
|-------|:-----:|:--:|:--------:|
| Late Fusion (baseline) | 0.6440 | 0.5876 | 0.5880 |
| Cross-Attention (best) | 0.6567 | 0.6299 | 0.6300 |
| Tensor Fusion | 0.6502 | 0.5953 | 0.5980 |
| Student (distilled) | 0.6244 | 0.5221 | 0.5640 |
