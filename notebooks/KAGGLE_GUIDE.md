# Kaggle Training Guide for MultiGuard

Step-by-step instructions for running `06_kaggle_training.ipynb` on Kaggle's free T4 GPU.

---

## 1. Prerequisites

- **Kaggle account** (free at [kaggle.com](https://www.kaggle.com/))
- **~30 hours** of free GPU quota per week
- **Internet access** enabled in the notebook (for downloading pretrained models)

---

## 2. Setting Up the Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com/) and sign in
2. Click **"New Notebook"** (or navigate to **Code** → **New Notebook**)
3. Upload the notebook:
   - Click **File** → **"Import Notebook"**
   - Upload `06_kaggle_training.ipynb`

---

## 3. Adding the Dataset

1. In the notebook sidebar, click **"+ Add Data"**
2. Search for **"hateful memes"**
3. Select the **Facebook Hateful Memes** dataset (community upload)
4. Verify the data path shows as `/kaggle/input/hateful-memes/`
5. Expected contents:
   - `train.jsonl` — training annotations
   - `dev_seen.jsonl` — validation annotations
   - `test_seen.jsonl` — test annotations
   - `img/` — meme images folder

---

## 4. Configuring the GPU

1. Click the **"..."** menu (top-right) → **"Accelerator"** → Select **"GPU T4 x2"**
2. Verify in the Settings panel: Accelerator = GPU
3. Enable the **"Internet"** toggle (needed for downloading CLIP and RoBERTa weights)
4. Set **Persistence**: **"Files only"** (keeps outputs after the session ends)

---

## 5. Running the Training Pipeline

### Option A: Run All

Click **"Run All"** — takes approximately **6-8 hours** total.

### Option B: Run Stage by Stage (Recommended for First Time)

#### Stage 1: Baseline (Cells 1–21) — ~2-3 hours

| Cells | Content | Time |
|-------|---------|------|
| Cell 1 (Intro) | Overview markdown | — |
| Cells 2–18 | Setup: dependencies, imports, config, dataset, model definitions, trainer | < 5 min |
| Cell 20 | **Baseline training** (CLIP ViT-B/16 + RoBERTa + Late Fusion, 10 epochs) | ~2-3 hrs |
| Cell 21 | Plot training curves | < 1 min |

#### Stage 2: Fusion Ablation (Cells 22–25) — ~3-4 hours

| Cell | Content | Time |
|------|---------|------|
| Cell 22 | Stage 2 overview markdown | — |
| Cell 23 | **Cross-Attention fusion training** (15 epochs) | ~2 hrs |
| Cell 24 | **Tensor Fusion training** (15 epochs) | ~1.5 hrs |
| Cell 25 | Fusion comparison chart | < 1 min |

#### Stage 3: Knowledge Distillation (Cells 26–29) — ~1-2 hours

| Cell | Content | Time |
|------|---------|------|
| Cell 26 | Stage 3 overview markdown | — |
| Cell 27 | DistillationTrainer class definition | < 1 min |
| Cell 28 | **Load best teacher & train student** (20 epochs) | ~1 hr |
| Cell 29 | Teacher vs student comparison | < 1 min |

#### Evaluation (Cells 30–34) — < 10 minutes

| Cell | Content |
|------|---------|
| Cell 30 | Full test set evaluation for all models |
| Cell 31 | Confusion matrix, ROC curve, classification report |
| Cell 32 | Summary table |
| Cell 33 | Save best model checkpoint |
| Cell 34 | ONNX export with quantization (optional) |

---

## 6. Monitoring Training

- Each epoch logs: **loss**, **accuracy**, **F1**, **AUROC**
- **VRAM usage** is logged after each epoch
- **Checkpoints** are auto-saved to `/kaggle/working/` when AUROC improves
- If the kernel disconnects: re-run from the last completed stage (checkpoints are preserved)

---

## 7. Retrieving Results

1. After training completes, go to the **"Output"** tab in Kaggle
2. Download your files:
   - `multiguard_best.pt` — best overall model checkpoint
   - `*_best.pt` — per-stage checkpoints
   - `*.png` — training curves and evaluation plots
3. Alternatively, use **"Save Version"** → **"Save & Run All"** for a persistent run

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| **"No GPU detected"** | Go to Settings → Accelerator → GPU T4 |
| **OOM (Out of Memory) errors** | Reduce `batch_size` in the CONFIG cell (Cell 4) |
| **Dataset not found** | Check data path — run `!ls /kaggle/input/` to verify |
| **Kernel timeout (12hr limit)** | Run stages separately across sessions, or reduce epochs |
| **Internet errors downloading models** | Enable Internet in the Settings sidebar |
| **Kernel disconnects mid-training** | Re-run from the last completed stage; checkpoints are saved |

---

## 9. Expected Results (Approximate)

| Model | AUROC | F1 | Training Time |
|-------|-------|----|---------------|
| Baseline (Late Fusion) | 0.65–0.72 | 0.55–0.65 | ~2-3 hrs |
| Cross-Attention Fusion | 0.68–0.75 | 0.58–0.68 | ~2 hrs |
| Tensor Fusion | 0.66–0.73 | 0.56–0.66 | ~1.5 hrs |
| Student (Distilled) | 0.62–0.70 | 0.52–0.62 | ~1 hr |

> **Note:** Results vary based on dataset split, random seed, and GPU conditions. The notebook uses seed 42 for reproducibility.

---

## 10. GPU Quota Tips

- Kaggle provides **30 hours/week** of free GPU time
- Running **stage by stage** lets you spread training across multiple days
- **Save checkpoints frequently** — already built into the trainer
- **Turn off GPU when editing code**: toggle Accelerator to "None", switch back to GPU T4 only when ready to train
- Use **"Save Version" → "Save & Run All"** for unattended runs (runs even if you close the browser)
