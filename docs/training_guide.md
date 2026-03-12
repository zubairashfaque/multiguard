# Training Guide

## Prerequisites

- GPU with >= 16GB VRAM (RTX 3090/4090 recommended)
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

# 5. Evaluate
make evaluate
```

## Training Configurations

| Config | Fusion | Backbone | Epochs |
|--------|--------|----------|--------|
| baseline.yaml | Late (concat+MLP) | CLIP + RoBERTa | 10 |
| fusion_ablation.yaml | Cross-attention | CLIP + RoBERTa | 15 |
| distillation.yaml | Late (student) | EfficientNet + DistilRoBERTa | 20 |

## Monitoring Training

- **W&B:** Set `WANDB_API_KEY` in `.env`, enable in config
- **MLflow:** `mlflow ui` at http://localhost:5000
- **Logs:** Check `reports/` directory
