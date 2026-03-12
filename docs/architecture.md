# MultiGuard Architecture

## System Overview

MultiGuard is a production-grade multimodal content intelligence pipeline that ingests paired text+image inputs and performs classification, retrieval, and explainability analysis.

## Pipeline

```
Input (text + image)
    |
    v
[Text Backbone]    [Vision Backbone]
(RoBERTa/BERT)     (ViT/CLIP/DINOv2)
    |                    |
    v                    v
[Text Features]    [Vision Features]
    |                    |
    +--------+-----------+
             |
             v
      [Fusion Module]
      (Late/CrossAttn/Tensor)
             |
             v
      [Task Heads]
      - Classifier (safe/unsafe)
      - Retrieval (contrastive embeddings)
      - VQA (answer generation)
             |
             v
      [Explainability]
      - GradCAM heatmaps
      - Integrated Gradients
      - SHAP values
```

## Key Design Decisions

1. **Modular fusion** — swap fusion strategies without changing backbone or head code
2. **Registry pattern** — dynamically register and look up components by name
3. **Config-driven** — all hyperparameters externalized to YAML
4. **Multi-task capable** — shared backbone with multiple task heads
5. **Cascaded inference** — fast CLIP zero-shot filter before full model for efficiency
