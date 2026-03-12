# Experiment Log

## Training Hardware
- **GPU:** Tesla P100-PCIE-16GB (Kaggle)
- **Precision:** FP16 mixed precision
- **Effective batch size:** 64 (batch=32, grad_accum=2)

## Final Results (Test Set)

| Model | Fusion | AUROC | F1 | Accuracy | Params (M) | Size (MB) |
|-------|--------|:-----:|:--:|:--------:|:----------:|:---------:|
| Late Fusion (baseline) | Concat + MLP | 0.6440 | 0.5876 | 0.5880 | 297.9 | 807.8 |
| **Cross-Attention (BEST)** | **4-layer bidirectional** | **0.6567** | **0.6299** | **0.6300** | **310.8** | **857.2** |
| Tensor Fusion | Low-rank (rank=32) | 0.6502 | 0.5953 | 0.5980 | 297.4 | 805.7 |
| Student (Distilled) | Late (EfficientNet-B0 + DistilRoBERTa) | 0.6244 | 0.5221 | 0.5640 | 86.7 | 331.1 |

## Classification Report (Cross-Attention, Best Model)

```
              precision    recall  f1-score   support
 Not Hateful       0.63      0.61      0.62       250
     Hateful       0.63      0.65      0.64       250
    accuracy                           0.63       500
```

## Teacher vs Student Comparison

| Metric | Teacher (Cross-Attn) | Student (Distilled) | Ratio |
|--------|:--------------------:|:-------------------:|:-----:|
| Parameters (M) | 224.6 | 86.7 | 2.6x compression |
| Model Size (MB) | 857.2 | 331.1 | 2.6x smaller |
| Batch Inference (ms) | 252.3 | 58.3 | 4.3x faster |
| AUROC | 0.6567 | 0.6244 | 95% retained |

## ONNX Quantization

| Metric | Value |
|--------|:-----:|
| Original student size | 331.1 MB |
| Quantized (int8) size | 206.3 MB |
| Compression ratio | 1.6x |

## Training Configurations

### Baseline (Late Fusion)
- Vision: CLIP ViT-B/16 (512-dim), Text: RoBERTa-base (768-dim)
- Epochs: 15, LR: 1e-3, Batch: 32, Grad Accum: 2
- Weight Decay: 1e-4, Warmup: 0.1, Max Seq Length: 77

### Fusion Ablation (Cross-Attention & Tensor)
- Vision: CLIP ViT-B/16 (512-dim), Text: RoBERTa-base (768-dim)
- Epochs: 20, LR: 5e-4, Batch: 32, Grad Accum: 2
- Weight Decay: 1e-4, Warmup: 0.1, Max Seq Length: 77

### Distillation
- Teacher: Cross-Attention (best fusion), frozen
- Student Vision: EfficientNet-B0 (1280-dim), Student Text: DistilRoBERTa (768-dim)
- Epochs: 20, LR: 1e-3, Batch: 32, Grad Accum: 2
- Temperature: 4.0, Alpha: 0.5, Warmup: 0.1, Max Seq Length: 77

## Notebook
Full training notebook: `notebooks/multiguard_kaggle_p100_full_training.ipynb`
