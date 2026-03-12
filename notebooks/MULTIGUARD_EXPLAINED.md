# MultiGuard: The Complete Explainer

> A plain-language guide to the entire MultiGuard system — what it does, how it works, and why every piece exists.

---

## Table of Contents

1. [What MultiGuard Aims to Achieve](#1-what-multiguard-aims-to-achieve)
2. [The Problem — Why Training is Needed](#2-the-problem--why-training-is-needed)
3. [What We Train — 3 Stages](#3-what-we-train--3-stages)
   - [Stage 1: Baseline (Late Fusion)](#stage-1-baseline-late-fusion)
   - [Stage 2: Fusion Ablation (Cross-Attention + Tensor Fusion)](#stage-2-fusion-ablation-cross-attention--tensor-fusion)
   - [Stage 3: Knowledge Distillation](#stage-3-knowledge-distillation)
4. [Training Techniques and Why Each Was Chosen](#4-training-techniques-and-why-each-was-chosen)
5. [Tools Used and Why](#5-tools-used-and-why)
6. [What We See at the End](#6-what-we-see-at-the-end)
7. [System Diagrams](#7-system-diagrams)
8. [Glossary](#8-glossary)

---

## 1. What MultiGuard Aims to Achieve

### The Core Problem

A meme is text layered on an image. Separately, the text might be harmless and the image might be harmless — but **together** they can create a hateful message that neither component carries alone.

MultiGuard is a system that detects hateful memes by understanding **both** the text and the image **and the relationship between them**.

### Three Concrete Examples

| # | Text (alone) | Image (alone) | Combined Meaning |
|---|-------------|---------------|-----------------|
| 1 | "Look who's back!" | Photo of a smiling person | **Hateful** — targets a specific group with threatening subtext |
| 2 | "They all look the same to me" | Photo of identical objects | **Benign** — a joke about identical products |
| 3 | "Not all heroes wear capes" | Neutral photo of a public figure | **Hateful** — sarcastically dehumanizes the person depicted |

The same text or image can flip from benign to hateful depending on its partner. A system that only reads text or only sees images will miss the combined meaning.

### Analogy

> **Two detectives — one only hears, one only sees. Neither can understand a video.**
>
> Imagine a detective who only listens to audio and another who only watches surveillance footage (muted). If a suspect says "nice place you've got here" while smashing furniture, the audio detective hears a compliment and the video detective sees vandalism. Only by combining both can you understand the threat.

### Diagram: Text-Image Combination Grid

```
                        IMAGE
                  Benign        Offensive
              +-----------+-----------+
    Benign    |  BENIGN   |  HATEFUL  |
  T           | "Nice dog"|  Innocent |
  E           | + cute    |  text +   |
  X           |   puppy   |  offensive|
  T           |           |  imagery  |
              +-----------+-----------+
    Offensive |  HATEFUL  |  HATEFUL  |
              | Offensive | Both parts|
              | text +    | reinforce |
              | innocent  | the hate  |
              | image     |           |
              +-----------+-----------+

  KEY INSIGHT: The top-right cell is the hardest case.
  The text alone seems fine. The image alone seems fine.
  Only together do they become hateful.
```

---

## 2. The Problem — Why Training is Needed

### What Pre-Trained Models Already Know

MultiGuard uses two powerful pre-trained models:

- **CLIP ViT-L/14** — Trained by OpenAI on 400 million image-text pairs. It understands images deeply — objects, scenes, faces, styles.
- **RoBERTa-base** — Trained by Meta on 160GB of text. It understands language deeply — grammar, sentiment, context, sarcasm.

These models are *already* excellent at their individual jobs. We do **not** retrain them from scratch.

### What They Cannot Do (Yet)

Neither model can answer: **"Is this meme hateful?"**

They were trained to *understand* their modality, not to *classify hate*. More importantly, they have never learned to combine their signals into a joint verdict.

### What Needs Training

We add **new layers on top** of these pre-trained models:

1. **Fusion layers** — combine vision and text features into a single representation
2. **Classification head** — take the fused representation and output "hateful" or "not hateful"

Only these new layers are trained. The pre-trained backbones are either frozen or lightly fine-tuned.

### The Dataset

**Facebook Hateful Memes Challenge** — ~10,000 memes with binary labels (hateful / not hateful). It was specifically designed so that unimodal classifiers fail. The "confounders" — benign images paired with hateful text, or vice versa — force the model to truly understand the combination.

### Analogy

> **Two expert translators who have never worked in court.**
>
> You hire a French translator and a Japanese translator. Both are world-class at their language. But neither has ever been in a courtroom. They don't know legal procedure, when to object, or how to render a verdict. They need **court training** — not language training. Similarly, CLIP and RoBERTa need *fusion and classification training*, not more vision or language training.

### Diagram: Pre-Trained vs Trained-By-Us

```
  ┌─────────────────────────────────────────────────────────┐
  │                    MULTIGUARD MODEL                     │
  │                                                         │
  │  ┌─────────────────┐     ┌─────────────────┐           │
  │  │   CLIP ViT-L/14 │     │  RoBERTa-base   │           │
  │  │                  │     │                  │           │
  │  │  400M image-text │     │   160GB text     │           │
  │  │  pairs learned   │     │   learned        │           │
  │  │                  │     │                  │           │
  │  │  PRE-TRAINED     │     │  PRE-TRAINED     │           │
  │  │  (frozen or      │     │  (frozen or      │           │
  │  │   lightly tuned) │     │   lightly tuned) │           │
  │  └────────┬─────────┘     └────────┬─────────┘           │
  │           │ 768-dim                │ 768-dim              │
  │           │ features               │ features             │
  │           ▼                        ▼                      │
  │  ┌────────────────────────────────────────────┐          │
  │  │          FUSION LAYERS                      │          │
  │  │   (Late / Cross-Attention / Tensor)         │          │
  │  │                                              │          │
  │  │   *** TRAINED BY US ***                      │          │
  │  └──────────────────┬───────────────────────────┘          │
  │                     │ fused features                       │
  │                     ▼                                      │
  │  ┌────────────────────────────────────────────┐          │
  │  │       CLASSIFICATION HEAD                   │          │
  │  │   Linear → Softmax → [hateful, not hateful] │          │
  │  │                                              │          │
  │  │   *** TRAINED BY US ***                      │          │
  │  └──────────────────┬───────────────────────────┘          │
  │                     │                                      │
  │                     ▼                                      │
  │              Prediction: 0.87 hateful                      │
  └─────────────────────────────────────────────────┘

  LEGEND:
    PRE-TRAINED  = Already knows its job (not retrained)
    TRAINED BY US = New layers we add and train on Hateful Memes data
```

---

## 3. What We Train — 3 Stages

Training happens in three progressive stages. Each stage builds on insights from the previous one.

```
  STAGE 1              STAGE 2                STAGE 3
  Baseline             Fusion Ablation        Distillation
  ──────────           ───────────────        ─────────────
  Simple concat        Sophisticated fusion   Compress best
  + MLP                (cross-attn, tensor)   into small model

  ┌──────┐   learn     ┌──────┐   learn      ┌──────┐
  │      │ ────────►   │      │ ────────►    │      │
  │ 10   │   what      │ 15   │   which      │ 20   │
  │epochs│   works     │epochs│   fusion     │epochs│
  │      │             │      │   is best    │      │
  └──────┘             └──────┘              └──────┘
  LR=1e-4              LR=5e-5               LR=1e-3
  batch=32             batch=16              batch=64
```

---

### Stage 1: Baseline (Late Fusion)

**Source:** `configs/train/baseline.yaml`, `src/models/fusion/late_fusion.py`

#### Architecture

The simplest fusion approach: take the vision features and the text features, **concatenate** them side by side, and pass the combined vector through an MLP (multi-layer perceptron) to get a prediction.

```
  Image                      Text
    │                          │
    ▼                          ▼
  ┌──────────────┐   ┌──────────────┐
  │ CLIP ViT-L/14│   │ RoBERTa-base │
  └──────┬───────┘   └──────┬───────┘
         │ [768]             │ [768]
         │                   │
         └───────┬───────────┘
                 │
                 ▼
          CONCATENATE
           [1536-dim]
                 │
                 ▼
        ┌─────────────┐
        │ Linear(1536  │
        │   → 512)    │
        │ + GELU      │
        │ + Dropout   │
        ├─────────────┤
        │ Linear(512   │
        │   → 256)    │
        │ + GELU      │
        │ + Dropout   │
        └──────┬──────┘
               │ [256]
               ▼
        Classification
        Head → Predict
```

#### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch size | 32 |
| Gradient accumulation | 2 steps |
| Effective batch size | 64 |
| Learning rate | 1e-4 |
| Warmup ratio | 0.1 |
| LR scheduler | Cosine annealing |
| Max gradient norm | 1.0 |
| Precision | BF16 |
| Vision backbone | CLIP ViT-L/14 |
| Text backbone | RoBERTa-base |
| Hidden dims | [512, 256] |
| Dropout | 0.1 |

#### Analogy

> **Two experts write separate reports, a manager reads both and decides.**
>
> A vision expert writes a 768-word report about the image. A text expert writes a 768-word report about the words. The manager (MLP) reads both reports back-to-back (concatenated) and makes the final call. The experts never talk to each other — the manager does all the synthesis.

#### Why Start Here?

Late fusion is the simplest possible approach. It establishes a performance floor. If more sophisticated fusion strategies don't beat this baseline, they're not worth the extra complexity.

---

### Stage 2: Fusion Ablation (Cross-Attention + Tensor Fusion)

**Source:** `configs/train/fusion_ablation.yaml`, `src/models/fusion/cross_attention.py`, `src/models/fusion/tensor_fusion.py`

In this stage we test two advanced fusion strategies to see if letting the modalities interact more deeply improves results.

#### Configuration (Shared)

| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Batch size | 16 |
| Gradient accumulation | 4 steps |
| Effective batch size | 64 |
| Learning rate | 5e-5 |
| Warmup ratio | 0.1 |
| LR scheduler | Cosine annealing |
| Precision | BF16 |

---

#### Strategy A: Cross-Attention Fusion

**Source:** `src/models/fusion/cross_attention.py`

##### How It Works

Instead of just concatenating features, cross-attention lets each modality **ask questions of the other**:

1. **Vision-to-Text (V2T):** The image features attend to the text features. "Given what I see in this image, which words matter most?"
2. **Text-to-Vision (T2V):** The text features attend to the image features. "Given these words, which parts of the image are relevant?"
3. **Gated Combination:** A learned gate decides how much weight to give V2T vs T2V.

This runs through 4 layers with 8 attention heads each, allowing progressively deeper cross-modal understanding.

##### Architecture Diagram

```
  Vision features [B, 768]          Text features [B, 768]
         │                                  │
         ▼                                  ▼
    unsqueeze(1)                       unsqueeze(1)
    [B, 1, 768]                        [B, 1, 768]
         │                                  │
         │          ┌──────────┐            │
         │          │          │            │
         ▼          │          ▼            │
  ┌─────────────┐   │   ┌─────────────┐    │
  │ V2T Layer 1 │◄──┼───│ T2V Layer 1 │◄───┘
  │ (TransDec)  │   │   │ (TransDec)  │
  ├─────────────┤   │   ├─────────────┤
  │ V2T Layer 2 │   │   │ T2V Layer 2 │
  ├─────────────┤   │   ├─────────────┤
  │ V2T Layer 3 │   │   │ T2V Layer 3 │
  ├─────────────┤   │   ├─────────────┤
  │ V2T Layer 4 │   │   │ T2V Layer 4 │
  └──────┬──────┘   │   └──────┬──────┘
         │          │          │
    squeeze(1)      │     squeeze(1)
      [B,768]       │       [B,768]
         │          │          │
         └────┬─────┘──────────┘
              │
              ▼
       ┌─────────────┐
       │ CONCAT       │
       │ [B, 1536]    │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Gate =        │
       │ Sigmoid(      │
       │  Linear(1536  │
       │    → 768))    │
       └──────┬───────┘
              │
              ▼
       ┌──────────────────────────────┐
       │ fused = gate * V2T           │
       │       + (1 - gate) * T2V     │
       │                              │
       │ Output: [B, 768]             │
       └──────────────────────────────┘

  Cross-Attention Parameters:
    num_layers = 4
    num_heads  = 8
    hidden_dim = 768
    feedforward_dim = 2048
```

##### Analogy

> **Two detectives interviewing each other about the case.**
>
> Detective Vision says: "I see a person with a torn flag. What does the text say about flags?" Detective Text says: "The text says 'this is what they deserve.' What in the image shows who 'they' are?" Each detective's questions are shaped by what the other knows. After four rounds of questioning, both have a much richer understanding of the case than either had alone.

---

#### Strategy B: Tensor Fusion

**Source:** `src/models/fusion/tensor_fusion.py`

##### How It Works

Tensor fusion captures **multiplicative interactions** between modalities. Instead of just asking "what does each modality say?" it asks "what new meaning emerges when we multiply them together?"

1. **Project** vision and text features into a shared low-rank space (rank=32)
2. **Element-wise multiply** the projections — this creates cross-modal interaction terms
3. **Project back** to the hidden dimension
4. **Layer-normalize** the result

##### Architecture Diagram

```
  Vision features [B, 768]     Text features [B, 768]
         │                              │
         ▼                              ▼
  ┌──────────────┐              ┌──────────────┐
  │ vision_factor │              │ text_factor   │
  │ Linear(768    │              │ Linear(768    │
  │   → 32)      │              │   → 32)       │
  │ (no bias)     │              │ (no bias)     │
  └──────┬───────┘              └──────┬───────┘
         │ [B, 32]                     │ [B, 32]
         │                             │
         └──────────┬──────────────────┘
                    │
                    ▼
             ┌─────────────┐
             │  Element-    │
             │  wise        │
             │  Multiply    │
             │  v * t       │
             └──────┬──────┘
                    │ [B, 32]
                    │
                    ▼
             ┌─────────────┐
             │ fusion_wts   │
             │ Linear(32    │
             │   → 768)     │
             └──────┬──────┘
                    │ [B, 768]
                    ▼
              ┌──────────┐
              │ Dropout   │
              │ LayerNorm │
              └─────┬────┘
                    │
                    ▼
              Output [B, 768]

  Tensor Fusion Parameters:
    rank = 32
    vision_dim = 768
    text_dim = 768
    hidden_dim = 768
```

##### Analogy

> **Mixing paint colors — red + blue = purple (new meaning emerges).**
>
> If you put a red dot and a blue dot next to each other, you see two colors. But if you *mix* them, you get purple — a completely new color that neither dot contained. Tensor fusion is the "mixing." When vision says "person" (red) and text says "they deserve it" (blue), the element-wise product captures the *interaction* — the threatening combination (purple) that emerges only from mixing.

---

### Stage 3: Knowledge Distillation

**Source:** `configs/train/distillation.yaml`, `src/training/distillation_trainer.py`

#### Why Distill?

The Stage 2 model is accurate but heavy — CLIP ViT-L/14 and RoBERTa-base have hundreds of millions of parameters. For production deployment (real-time API, mobile, edge), we need a **smaller, faster** model that retains most of the accuracy.

#### Architecture

- **Teacher model:** Best model from Stage 2 (frozen — it does not learn further)
- **Student model:** Much smaller backbones
  - Vision: **EfficientNet-B0** (~5M params vs CLIP's ~300M)
  - Text: **DistilRoBERTa** (~82M params vs RoBERTa's ~125M)
  - Fusion: Late fusion (simplest and fastest)

#### The Loss Formula

The student learns from two sources simultaneously:

```
  total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

  Where:
    alpha     = 0.5 (equal weight to both losses)
    hard_loss = CrossEntropy(student_logits, true_labels)
    soft_loss = T^2 * KL_div(
                    log_softmax(student_logits / T),
                    softmax(teacher_logits / T)
                )
    T         = 4.0 (temperature)
```

**Hard loss:** Compare student's predictions to the ground truth labels (0 or 1).

**Soft loss:** Compare student's *probability distribution* to the teacher's *probability distribution*. The temperature T=4.0 "softens" these distributions, revealing the teacher's uncertainty patterns. The T^2 factor compensates for the gradient magnitude change caused by dividing by T.

##### Why Two Losses?

- **Hard loss alone** = the student learns from labels but ignores the teacher's nuanced understanding
- **Soft loss alone** = the student mimics the teacher but may learn the teacher's mistakes
- **Both together** = the student gets the best of both worlds

#### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch size | 64 |
| Gradient accumulation | 1 step |
| Effective batch size | 64 |
| Learning rate | 1e-3 |
| Warmup ratio | 0.05 |
| Temperature (T) | 4.0 |
| Alpha | 0.5 |
| Student vision | EfficientNet-B0 |
| Student text | DistilRoBERTa |
| Student fusion | Late fusion |
| Teacher | Best fusion model (frozen) |

#### Distillation Flow Diagram

```
                    ┌──────────────────────────────┐
                    │        TEACHER MODEL          │
                    │     (Best from Stage 2)       │
                    │         *** FROZEN ***         │
                    │                                │
                    │  CLIP ViT-L/14 + RoBERTa      │
                    │  + Best Fusion Strategy        │
                    └──────────┬───────────────────┘
                               │
                               │ teacher_logits
                               │ (soft targets)
                               ▼
                    ┌──────────────────────┐
                    │    SOFT LOSS          │
  ┌──────────┐     │                      │
  │          │     │  T^2 * KL_div(       │
  │  MEME    │     │    student/T,        │
  │  INPUT   │     │    teacher/T)        │     ┌──────────┐
  │          │     │                      │     │          │
  │ image +  │     │  T = 4.0            │     │  TRUE    │
  │ text     │     └──────────┬───────────┘     │  LABELS  │
  │          │                │                  │  (0 / 1) │
  └────┬─────┘                │ (1-alpha)        └────┬─────┘
       │                      │ = 0.5                  │
       │                      │                        │
       ▼                      ▼                        │
  ┌──────────────────────────────────────┐             │
  │         STUDENT MODEL                │             │
  │                                      │             │
  │  EfficientNet-B0 + DistilRoBERTa     │             │
  │  + Late Fusion                       │             │
  │                                      │             │
  │  *** THIS IS WHAT GETS TRAINED ***   │             │
  └──────────┬───────────────────────────┘             │
             │                                          │
             │ student_logits                           │
             │                                          │
             ▼                                          ▼
  ┌──────────────────────────────────────────────────────┐
  │                    HARD LOSS                         │
  │          CrossEntropy(student, labels)                │
  └──────────────────────┬───────────────────────────────┘
                         │ alpha = 0.5
                         │
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │  TOTAL LOSS = 0.5 * hard_loss + 0.5 * soft_loss     │
  │                                                      │
  │  Backpropagate through STUDENT only                  │
  │  (teacher is frozen)                                 │
  └──────────────────────────────────────────────────────┘
```

#### Analogy

> **Master chef explains reasoning to student chef.**
>
> A master chef doesn't just say "this dish is bad" (hard label). They say "this dish is 87% likely to need more salt, 10% likely to need acid, and 3% likely fine as-is" (soft probabilities). The student chef learns not just the right answer, but the *reasoning patterns* — which subtle cues indicate salt vs acid, how confident to be in edge cases. This is far richer than binary right/wrong feedback.

---

## 4. Training Techniques and Why Each Was Chosen

Every technique below solves a specific problem. For each: **What** it is, **Why** we need it, **How** it works, an **Analogy**, and the **Config values** used.

---

### 4.1 Focal Loss

**Source:** `src/models/losses.py` | **Config:** alpha=0.25, gamma=2.0

**What:** A modified cross-entropy loss that down-weights easy examples and focuses on hard ones.

**Why:** The Hateful Memes dataset is imbalanced — most memes are "not hateful." Standard cross-entropy lets the model coast by correctly classifying easy examples while ignoring the hard cases that actually matter.

**How:** The formula adds a modulating factor `(1 - pt)^gamma` where `pt` is the model's confidence in the correct answer. When the model is very confident (pt close to 1), the factor shrinks the loss nearly to zero. When the model is uncertain (pt close to 0.5), the loss stays high.

```
  focal_loss = alpha * (1 - pt)^gamma * cross_entropy_loss

  With alpha=0.25, gamma=2.0:
    - Easy example (pt=0.95): weight = 0.25 * (0.05)^2 = 0.000625  (nearly zero)
    - Hard example (pt=0.55): weight = 0.25 * (0.45)^2 = 0.050625  (80x higher)
```

**Analogy:** A teacher who stops reviewing problems the student already knows and spends all their time on the problems the student keeps getting wrong.

---

### 4.2 Mixed Precision Training (BF16)

**Config:** `bf16: true` in all training configs

**What:** Store model weights and compute gradients using 16-bit floating point (BFloat16) instead of the default 32-bit.

**Why:** BF16 uses half the memory and runs faster on modern GPUs, with virtually no loss in accuracy. It keeps the same exponent range as FP32 (so no overflow issues) while sacrificing some decimal precision.

**How:** PyTorch's autocast automatically converts operations to BF16 where safe and keeps critical operations (like loss computation) in FP32.

```
  FP32: 1 bit sign | 8 bits exponent | 23 bits mantissa = 32 bits total
  BF16: 1 bit sign | 8 bits exponent |  7 bits mantissa = 16 bits total
                                        ▲
                                   Less precision here,
                                   but same range as FP32
```

**Analogy:** Shorthand notes — same meaning, less paper. A doctor writes "pt c/o HA" instead of "patient complains of headache." The core information is preserved; only unnecessary precision is dropped.

---

### 4.3 Gradient Accumulation

**Config:** Baseline=2 steps, Fusion=4 steps, Distillation=1 step (all achieve effective batch=64)

**What:** Instead of updating the model after every batch, accumulate gradients over multiple batches and update once.

**Why:** Larger effective batch sizes produce more stable gradient estimates, but large batches may not fit in GPU memory. Gradient accumulation lets us simulate a large batch by spreading it across multiple smaller batches.

**How:**

```
  Without gradient accumulation (batch=64):
    Load 64 samples → compute gradients → update weights
    (May not fit in GPU memory!)

  With gradient accumulation (batch=16, accum=4):
    Load 16 samples → compute gradients → hold
    Load 16 samples → compute gradients → add to held
    Load 16 samples → compute gradients → add to held
    Load 16 samples → compute gradients → add to held
    Now update weights (using sum of all 4 * 16 = 64 gradients)
    (Fits in GPU memory!)
```

**Analogy:** Collecting opinions from small groups before making a decision. Instead of surveying 64 people at once (which requires a huge room), you survey 4 groups of 16 and combine the responses. Same total input, manageable logistics.

---

### 4.4 Cosine Annealing with Warmup

**Config:** `lr_scheduler: cosine`, `warmup_ratio: 0.1` (or 0.05 for distillation)

**What:** The learning rate starts at zero, linearly increases during warmup, then smoothly decreases following a cosine curve.

**Why:**
- **Warmup** prevents the model from making wild updates at the start when the loss landscape is unfamiliar.
- **Cosine decay** gradually reduces the step size so the model can fine-tune its weights precisely near the end of training.

**How:**

```
  Learning
  Rate
    │
    │           Warmup     Cosine Decay
    │           (10%)       (90%)
    │            ╱─╲
    │           ╱    ╲
    │          ╱      ╲
    │         ╱        ╲
    │        ╱          ╲
    │       ╱            ╲
    │      ╱              ╲
    │     ╱                ╲
    │    ╱                  ╲
    │   ╱                    ╲
    │──╱                      ╲──
    └─────────────────────────────► Epoch
         1    2    3    4    5
```

**Analogy:** Warming up before exercise, cooling down after. You don't sprint the moment you step onto the track (warmup). And you don't stop dead — you jog, then walk (cosine decay). The body (model) performs best with gradual transitions.

---

### 4.5 Data Augmentation

**Source:** `configs/data/augmentation.yaml`

**What:** Apply random transformations to training data so the model sees diverse variations of each example.

**Why:** With only ~10,000 memes, the model can memorize the training set instead of learning generalizable patterns. Augmentation artificially expands the dataset.

**How (Image):**

| Technique | Config | Effect |
|-----------|--------|--------|
| Random Resized Crop | 224px, scale 0.8-1.0 | Shows different portions of the image |
| Horizontal Flip | p=0.5 | Mirror image (50% chance) |
| Color Jitter | brightness/contrast/saturation=0.2, hue=0.1 | Slight color variations |
| Random Erasing | p=0.25, scale 0.02-0.1 | Randomly blocks out small patches |

**How (Text):**

| Technique | Config | Effect |
|-----------|--------|--------|
| Synonym Replacement | p=0.1, max 2 words | Replace words with synonyms |
| Random Deletion | p=0.05 | Randomly remove words |

**How (Multimodal):**

| Technique | Config | Effect |
|-----------|--------|--------|
| Mixup | alpha=0.2, p=0.3 | Blend two samples together |
| CutMix | alpha=1.0, p=0.3 | Paste patch from one sample onto another |

**Analogy:** Showing the same photo from different angles. A detective studying a suspect's face looks at photos from the left, right, slightly blurry, in shadow, partially obscured. Each view reinforces what the face really looks like versus what's just an artifact of one particular photo.

---

### 4.6 Gradient Clipping

**Config:** `max_grad_norm: 1.0` in all training configs

**What:** If the gradient magnitude exceeds a threshold (1.0), scale it down proportionally.

**Why:** Occasionally, a batch produces an unusually large gradient that would send the model parameters flying in an unhelpful direction. Gradient clipping prevents these "gradient explosions."

**How:**

```
  Before clipping: gradient = [3.0, 4.0]  → norm = 5.0
  max_norm = 1.0
  Scale factor = 1.0 / 5.0 = 0.2
  After clipping:  gradient = [0.6, 0.8]  → norm = 1.0

  Direction preserved, magnitude capped.
```

**Analogy:** A speed limit on a highway. The car (gradient) can go in any direction, but its speed is capped. This prevents crashes (training instability) while still allowing the car to reach its destination (good weights).

---

### 4.7 AdamW Optimizer

**Config:** `weight_decay: 0.01` in all training configs

**What:** An optimizer that maintains two running averages — one of the gradients (momentum) and one of the squared gradients (adaptive learning rate) — plus decoupled weight decay for regularization.

**Why:** AdamW adapts the learning rate per-parameter, which is critical for multimodal models where vision and text parameters may need very different step sizes. Decoupled weight decay prevents overfitting without interfering with the adaptive learning rates.

**How:** For each parameter, AdamW adjusts the learning rate based on:
- **First moment (mean of gradients):** Which direction has the gradient been pointing?
- **Second moment (mean of squared gradients):** How noisy is this gradient?

Parameters with consistent gradients get larger steps. Parameters with noisy gradients get smaller steps.

**Analogy:** A GPS with separate speed and route controls. The GPS (Adam) adjusts both your speed (learning rate) and direction (gradient) independently for each road (parameter). On a straight highway, it speeds up. On a winding mountain road, it slows down. Weight decay acts like a slight pull toward the center lane, preventing you from drifting too far off course.

---

### 4.8 Knowledge Distillation (Recap)

**Source:** `src/training/distillation_trainer.py` | **Config:** T=4.0, alpha=0.5

Already covered in detail in [Stage 3](#stage-3-knowledge-distillation). Key point: the student learns from **both** the true labels (hard loss) and the teacher's probability distributions (soft loss), getting richer training signal than labels alone.

**Analogy:** An expert teaching a quick student — not just the answers, but the reasoning behind every answer.

---

### Summary Table

| Technique | What It Solves | Key Values | Source |
|-----------|---------------|------------|--------|
| Focal Loss | Class imbalance | alpha=0.25, gamma=2.0 | `src/models/losses.py` |
| BF16 Mixed Precision | Memory + speed | `bf16: true` | All training configs |
| Gradient Accumulation | Effective batch size | 2/4/1 steps | All training configs |
| Cosine Annealing + Warmup | LR scheduling | warmup=0.1/0.05 | All training configs |
| Data Augmentation | Small dataset | See table above | `configs/data/augmentation.yaml` |
| Gradient Clipping | Gradient explosions | max_norm=1.0 | All training configs |
| AdamW | Per-param learning | weight_decay=0.01 | All training configs |
| Knowledge Distillation | Model compression | T=4.0, alpha=0.5 | `src/training/distillation_trainer.py` |

---

## 5. Tools Used and Why

### Core Deep Learning

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **PyTorch** | Core deep learning framework | Industry standard, dynamic computation graphs, excellent GPU support |
| **HuggingFace Transformers** | Pre-trained model hub | Easy access to CLIP, RoBERTa, DistilRoBERTa, EfficientNet |

### Vision

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **CLIP ViT-L/14** | Image encoder (teacher) | State-of-the-art vision-language understanding, already aligned with text |
| **timm (EfficientNet-B0)** | Image encoder (student) | Extremely efficient — 5M params, fast inference, good accuracy |
| **Albumentations** | Image augmentation | Fast, GPU-friendly, huge library of transforms |

### Language

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **RoBERTa-base** | Text encoder (teacher) | Strong language understanding, 125M params |
| **DistilRoBERTa** | Text encoder (student) | 40% smaller than RoBERTa, retains 97% performance |

### Training Infrastructure

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **AdamW** | Optimizer | Decoupled weight decay, per-param learning rates |
| **Focal Loss** | Class-imbalanced loss | Focuses on hard examples, down-weights easy ones |
| **OmegaConf** | Configuration management | YAML-based configs with type safety and overrides |

### Explainability

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **Captum** | Model interpretability | GradCAM (what image regions matter), Integrated Gradients (what text tokens matter) |
| **SHAP** | Feature attribution | Global and local explanations for model decisions |

### Serving and Deployment

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **FastAPI** | REST API framework | Async, auto-docs, type validation |
| **Docker** | Containerization | Reproducible deployments |
| **ONNX** | Model export format | Framework-agnostic, optimized inference runtime |
| **Redis** | Caching layer | Cache repeated predictions, reduce latency |

### MLOps

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **MLflow** | Experiment tracking | Log metrics, params, artifacts; compare runs |
| **W&B (Weights & Biases)** | Experiment visualization | Real-time training dashboards, team collaboration |
| **DVC** | Data version control | Track large datasets without git bloat |
| **Evidently** | Model monitoring | Detect data drift, model degradation in production |

### Code Quality

| Tool | What It Does | Why We Chose It |
|------|-------------|----------------|
| **ruff** | Fast Python linter | 10-100x faster than flake8, comprehensive rules |
| **black** | Code formatter | Deterministic formatting, zero config |
| **mypy** | Static type checker | Catch type errors before runtime |
| **pytest** | Testing framework | Simple, powerful, great fixture system |

---

## 6. What We See at the End

### Training Curves

After each stage completes, we get training curves showing how the model improved over time:

```
  Loss                              Accuracy
  │                                 │
  │╲                                │                    ╱─────
  │ ╲                               │                 ╱──
  │  ╲                              │              ╱──
  │   ╲                             │           ╱──
  │    ╲                            │        ╱──
  │     ╲                           │     ╱──
  │      ╲──                        │  ╱──
  │         ╲──                     │╱──
  │            ╲────                │
  │                ╲────────        │
  └────────────────────────► Epoch  └────────────────────────► Epoch
    1  2  3  4  5  6  7  8  9 10     1  2  3  4  5  6  7  8  9 10

  ── Train     ── Validation         ── Train     ── Validation
```

Key metrics tracked per epoch:
- **Loss** — should decrease (lower is better)
- **Accuracy** — should increase
- **F1 Score** — harmonic mean of precision and recall (critical for imbalanced data)
- **AUROC** — area under the ROC curve (how well the model separates classes)

### Fusion Strategy Comparison

A bar chart comparing all three fusion strategies on the same metrics:

```
  AUROC Comparison (higher is better)
  │
  │  ┌───┐
  │  │   │  ┌───┐
  │  │   │  │   │  ┌───┐
  │  │   │  │   │  │   │
  │  │   │  │   │  │   │
  │  │   │  │   │  │   │
  │  │   │  │   │  │   │
  │  │ L │  │ C │  │ T │
  │  │ A │  │ R │  │ E │
  │  │ T │  │ O │  │ N │
  │  │ E │  │ S │  │ S │
  │  │   │  │ S │  │ O │
  │  │   │  │   │  │ R │
  └──┴───┴──┴───┴──┴───┴──
     Late   Cross   Tensor
    Fusion  Attn    Fusion
```

### Teacher vs Student Comparison

| Metric | Teacher (large) | Student (distilled) | Retention |
|--------|----------------|--------------------|----|
| AUROC | — | — | — |
| F1 | — | — | — |
| Accuracy | — | — | — |
| Params | ~430M | ~87M | 80% smaller |
| Inference (ms) | — | — | — |
| Model size (MB) | — | — | — |

### Confusion Matrix

```
                    Predicted
                Not-Hateful  Hateful
  Actual  ┌──────────┬──────────┐
  Not-    │   TN     │   FP     │
  Hateful │ (correct)│ (false   │
          │          │  alarm)  │
          ├──────────┼──────────┤
  Hateful │   FN     │   TP     │
          │ (missed! │ (correct)│
          │  danger) │          │
          └──────────┴──────────┘

  Goal: Minimize FN (missed hateful memes)
  while keeping FP (false alarms) acceptable.
```

### ROC Curve

```
  True Positive Rate
  (Sensitivity)
  │
  1│         ╱──────────
   │       ╱─
   │     ╱─
   │   ╱─
   │  ╱    Model curve
   │ ╱     (higher = better)
   │╱
   │╱
   │       ╱  Random baseline
   │     ╱    (diagonal)
   │   ╱
   │ ╱
  0│╱──────────────────
   0                   1
     False Positive Rate
     (1 - Specificity)

  AUROC = Area under the model curve
  Perfect model: AUROC = 1.0
  Random model:  AUROC = 0.5
```

### Saved Artifacts

After training completes, the following artifacts are saved:

```
  models/
  ├── checkpoints/
  │   ├── baseline/           # Stage 1 checkpoints
  │   │   ├── step_500.pt
  │   │   ├── step_1000.pt
  │   │   └── best.pt
  │   ├── fusion_ablation/    # Stage 2 checkpoints
  │   │   ├── cross_attention/
  │   │   └── tensor_fusion/
  │   └── distillation/       # Stage 3 checkpoints
  │       └── best.pt
  ├── final/
  │   ├── best_fusion/        # Best model from Stage 2
  │   └── student/            # Distilled model
  └── exports/
      └── student.onnx        # ONNX export for deployment

  outputs/
  ├── plots/
  │   ├── training_curves.png
  │   ├── fusion_comparison.png
  │   ├── confusion_matrix.png
  │   ├── roc_curve.png
  │   └── gradcam_examples.png
  └── metrics/
      └── results.json        # All numeric results
```

### Output Dashboard Mockup

```
  ┌─────────────────────────────────────────────────────────────┐
  │  MULTIGUARD TRAINING RESULTS DASHBOARD                      │
  ├────────────────────────────┬────────────────────────────────┤
  │                            │                                │
  │  Training Loss Curves      │  Fusion Strategy Comparison    │
  │  ┌────────────────────┐    │  ┌────────────────────┐       │
  │  │ ╲                  │    │  │ ██  ██  ██         │       │
  │  │  ╲──               │    │  │ ██  ██  ██         │       │
  │  │     ╲───           │    │  │ ██  ██  ██         │       │
  │  │        ╲────       │    │  │ L   C   T          │       │
  │  └────────────────────┘    │  └────────────────────┘       │
  │                            │                                │
  ├────────────────────────────┼────────────────────────────────┤
  │                            │                                │
  │  Confusion Matrix          │  ROC Curve                     │
  │  ┌────────────────────┐    │  ┌────────────────────┐       │
  │  │  TN  │  FP         │    │  │      ╱─────        │       │
  │  │──────┼─────        │    │  │    ╱─              │       │
  │  │  FN  │  TP         │    │  │  ╱─                │       │
  │  └────────────────────┘    │  │╱                    │       │
  │                            │  └────────────────────┘       │
  ├────────────────────────────┴────────────────────────────────┤
  │  SUMMARY                                                     │
  │  Best Model: [fusion strategy]  |  AUROC: [score]           │
  │  Student Retention: [%]         |  Inference: [ms]          │
  └─────────────────────────────────────────────────────────────┘
```

---

## 7. System Diagrams

### Diagram 1: End-to-End System Architecture

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         END-TO-END SYSTEM                                │
  │                                                                          │
  │  DATA LAYER              MODEL LAYER              SERVING LAYER          │
  │  ─────────              ───────────              ─────────────          │
  │                                                                          │
  │  ┌──────────┐          ┌──────────────┐          ┌──────────────┐      │
  │  │ Hateful  │          │  Stage 1:    │          │              │      │
  │  │ Memes    │───DVC───►│  Baseline    │          │   FastAPI    │      │
  │  │ Dataset  │          │  (Late)      │          │   Server     │      │
  │  │ (~10K)   │          ├──────────────┤          │              │      │
  │  └──────────┘          │  Stage 2:    │   ONNX   │  ┌────────┐ │      │
  │       │                │  Fusion      │──export──►│  │ Model  │ │      │
  │       ▼                │  Ablation    │          │  │ (ONNX) │ │      │
  │  ┌──────────┐          ├──────────────┤          │  └────────┘ │      │
  │  │ Augment  │          │  Stage 3:    │          │       │     │      │
  │  │ (Album-  │          │  Distill     │          │       ▼     │      │
  │  │ entations│          │  → Student   │          │  ┌────────┐ │      │
  │  │ + text)  │          └──────────────┘          │  │ Redis  │ │      │
  │  └──────────┘                │                   │  │ Cache  │ │      │
  │       │                      │                   │  └────────┘ │      │
  │       │                      │                   └──────┬──────┘      │
  │       │                ┌─────┴──────┐                   │             │
  │       │                │  MLflow /  │             ┌─────┴──────┐      │
  │       │                │  W&B       │             │  Docker    │      │
  │       └───────────────►│  Tracking  │             │  Container │      │
  │                        └────────────┘             └────────────┘      │
  │                                                                        │
  │  MONITORING: Evidently (data drift, model degradation)                 │
  └──────────────────────────────────────────────────────────────────────────┘
```

### Diagram 2: Three Fusion Strategies Side by Side

```
  LATE FUSION              CROSS-ATTENTION           TENSOR FUSION
  ───────────              ───────────────           ─────────────

  Vision   Text            Vision   Text             Vision   Text
  [768]    [768]           [768]    [768]            [768]    [768]
    │        │               │        │                │        │
    │        │               │        │                │        │
    ▼        ▼               ▼        ▼                ▼        ▼
  ┌────────────┐        ┌──────┐  ┌──────┐       ┌────────┐┌────────┐
  │ CONCATENATE │        │ V2T  │  │ T2V  │       │ Linear ││ Linear │
  │ [1536]      │        │ Attn │  │ Attn │       │ 768→32 ││ 768→32 │
  └──────┬─────┘        │ x4   │  │ x4   │       └───┬────┘└───┬────┘
         │              │layers│  │layers│            │         │
         ▼              └──┬───┘  └──┬───┘            ▼         ▼
  ┌────────────┐           │         │           ┌──────────────┐
  │ MLP        │           ▼         ▼           │  Element-    │
  │ 1536→512   │      ┌─────────────────┐       │  wise        │
  │ 512→256    │      │  Gated Combine  │       │  Multiply    │
  │ GELU+Drop  │      │  g*V2T+(1-g)*T2V│       │  v * t       │
  └──────┬─────┘      └────────┬────────┘       └──────┬───────┘
         │                     │                       │
         ▼                     ▼                       ▼
      [256]                 [768]               ┌──────────┐
                                                │ Linear   │
  Simplest.              Deepest               │ 32→768   │
  No cross-modal         interaction.          │ +LayerNorm│
  interaction.           Modalities            └─────┬────┘
  Baseline.              question each               │
                         other.                      ▼
                                                   [768]

                                                Multiplicative
                                                interaction.
                                                Captures joint
                                                meaning.
```

### Diagram 3: Training Stages Progression

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    TRAINING STAGES TIMELINE                          │
  │                                                                      │
  │  STAGE 1               STAGE 2                STAGE 3                │
  │  Baseline              Fusion Ablation        Distillation           │
  │                                                                      │
  │  ┌──────────┐          ┌──────────────┐       ┌──────────────┐      │
  │  │ CLIP +   │          │ CLIP +       │       │ Teacher      │      │
  │  │ RoBERTa  │          │ RoBERTa      │       │ (frozen)     │      │
  │  │ + Late   │          │ + Cross-Attn │       │      │       │      │
  │  │   Fusion │          │ + Tensor     │       │      ▼       │      │
  │  │          │          │   Fusion     │       │ EfficientNet │      │
  │  │ 10 epochs│          │              │       │ + DistilRoBT │      │
  │  │ LR=1e-4  │          │ 15 epochs    │       │ + Late Fusion│      │
  │  │ batch=32 │          │ LR=5e-5      │       │              │      │
  │  └────┬─────┘          │ batch=16     │       │ 20 epochs    │      │
  │       │                └────┬─────────┘       │ LR=1e-3      │      │
  │       │                     │                  │ batch=64     │      │
  │       ▼                     ▼                  └────┬─────────┘      │
  │                                                     │                │
  │  Establishes           Finds best              Compresses best       │
  │  performance           fusion                  model into            │
  │  floor                 strategy                deployable size       │
  │       │                     │                       │                │
  │       ▼                     ▼                       ▼                │
  │  ┌─────────┐          ┌──────────┐            ┌──────────┐          │
  │  │Baseline │          │ Best     │            │ Student  │          │
  │  │ AUROC   │──compare─│ Fusion   │──teacher──►│ Model    │          │
  │  │         │          │ AUROC    │   for      │ (deploy) │          │
  │  └─────────┘          └──────────┘            └──────────┘          │
  │                                                     │                │
  │                                                     ▼                │
  │                                               ┌──────────┐          │
  │                                               │  ONNX    │          │
  │                                               │  Export   │          │
  │                                               │  → API   │          │
  │                                               └──────────┘          │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 8. Glossary

Each term includes a **Definition**, a **Simple Example**, and an **Analogy**.

---

### AUROC (Area Under the Receiver Operating Characteristic)

**Definition:** A metric that measures how well a model separates two classes across all possible classification thresholds. Ranges from 0.5 (random) to 1.0 (perfect).

**Example:** If AUROC = 0.85, then given a random hateful meme and a random benign meme, the model correctly ranks the hateful one higher 85% of the time.

**Analogy:** A judge's ability to sort guilty and innocent people into two lines. AUROC measures how cleanly the lines are separated, regardless of where you draw the dividing rope.

---

### F1 Score

**Definition:** The harmonic mean of precision (of the memes you flagged, how many were truly hateful?) and recall (of all hateful memes, how many did you catch?).

**Example:** If F1 = 0.72, the model has a balanced trade-off between catching hateful memes and not crying wolf.

**Analogy:** A smoke alarm's F1 score: precision is how often it beeps for real fires (not burnt toast), and recall is how many actual fires it catches. F1 balances both.

---

### CLIP (Contrastive Language-Image Pre-training)

**Definition:** A model trained by OpenAI on 400M image-text pairs to understand the relationship between images and text descriptions.

**Example:** Given a photo of a cat and the text "a photo of a cat," CLIP produces similar embeddings for both. Given "a photo of a dog," it produces a different embedding.

**Analogy:** A bilingual person who can look at a photo and describe it, or read a description and imagine the photo. They've built this skill by seeing millions of captioned images.

---

### RoBERTa (Robustly Optimized BERT Approach)

**Definition:** A text encoder trained by Meta that produces rich numerical representations (embeddings) of text, capturing meaning, grammar, and context.

**Example:** RoBERTa converts "they all look the same" into a 768-dimensional vector. Different contexts ("identical products" vs "people") produce different vectors.

**Analogy:** A court stenographer who doesn't just record words but captures tone, emphasis, and implied meaning in their notes.

---

### Fusion

**Definition:** The process of combining features from different modalities (vision + text) into a single representation for joint reasoning.

**Example:** Vision says [0.8, 0.2, 0.5] and text says [0.1, 0.9, 0.3]. Fusion combines these into a single vector like [0.6, 0.7, 0.4] that represents the joint meaning.

**Analogy:** A jury deliberation where witnesses from different departments share their perspectives to reach a single verdict. The verdict reflects combined knowledge, not just one witness.

---

### Cross-Attention

**Definition:** A mechanism where one modality uses its features as "queries" to search through the other modality's features for relevant information.

**Example:** The text "look who's back" attends to the image and focuses on the person's face rather than the background scenery, because the text suggests a person is relevant.

**Analogy:** Two detectives interviewing each other. Detective A says "tell me about the knife" and Detective B describes the kitchen scene. Then B asks "what did the suspect say about cooking?" and A plays back the relevant audio clip.

---

### Tensor Product (Outer Product)

**Definition:** A mathematical operation that computes all pairwise interactions between elements of two vectors, creating a matrix of cross-modal feature combinations.

**Example:** If vision = [a, b] and text = [x, y], the tensor product is [[ax, ay], [bx, by]] — every vision feature paired with every text feature.

**Analogy:** A dinner party seating chart where every guest from Group A is seated next to every guest from Group B. The conversations that emerge (interactions) reveal things neither group would say on their own.

---

### Knowledge Distillation

**Definition:** Training a small "student" model to mimic a large "teacher" model's behavior, transferring knowledge from the teacher's soft probability outputs.

**Example:** The teacher says "82% hateful, 18% benign" for a borderline meme. The student learns this nuanced distribution, not just "hateful" (the hard label).

**Analogy:** A master chess player teaching a student by narrating their thought process: "I'm 70% sure this is the best move because of the bishop position, 25% considering the knight fork, 5% worried about the pawn chain." The student learns the reasoning, not just the moves.

---

### Focal Loss

**Definition:** A loss function that reduces the contribution of easy-to-classify examples, focusing training on hard examples where the model is uncertain.

**Example:** If the model predicts 0.99 for an obviously benign meme, focal loss makes that example contribute almost nothing to the gradient. A confusing meme predicted at 0.55 contributes much more.

**Analogy:** A teacher who stops assigning homework on topics the student has mastered and doubles down on the topics they struggle with.

---

### Epoch

**Definition:** One complete pass through the entire training dataset. If you have 10,000 memes and train for 10 epochs, the model sees each meme 10 times.

**Example:** Stage 1 uses 10 epochs, Stage 2 uses 15, Stage 3 uses 20. More complex learning tasks need more passes.

**Analogy:** Re-reading a textbook. The first read gives you the basics. By the third read, you catch subtleties. By the tenth, you've internalized the material.

---

### Batch Size

**Definition:** The number of samples processed together in one forward/backward pass. Larger batches give more stable gradient estimates but use more memory.

**Example:** With batch_size=32 and 10,000 samples, each epoch has 312 batches. Each batch processes 32 memes simultaneously.

**Analogy:** Class size for a teacher. With 5 students, feedback is personalized but noisy. With 64 students, the average performance gives a more reliable signal of what's working.

---

### Learning Rate

**Definition:** The step size for each parameter update. Too high = the model overshoots good solutions. Too low = training takes forever or gets stuck.

**Example:** Stage 1 uses LR=1e-4 (moderate), Stage 2 uses 5e-5 (smaller, for fine-tuning), Stage 3 uses 1e-3 (larger, because the student starts from scratch).

**Analogy:** The size of your steps when searching for the lowest point in a valley while blindfolded. Big steps cover ground fast but might step over the lowest spot. Small steps are precise but slow.

---

### Gradient Accumulation Steps

**Definition:** The number of mini-batches whose gradients are summed before performing a single weight update.

**Example:** With batch_size=16 and gradient_accumulation=4, the model processes 4 batches of 16, accumulates their gradients, then updates once — simulating a batch of 64.

**Analogy:** Instead of voting after asking one group of 16 people, you poll four groups and combine all 64 opinions before making a decision.

---

### Warmup Ratio

**Definition:** The fraction of total training steps during which the learning rate linearly increases from zero to its target value.

**Example:** With warmup_ratio=0.1 and 1000 total steps, the LR ramps up during the first 100 steps before beginning cosine decay.

**Analogy:** Preheating an oven. You don't put the souffl (model) in a cold oven. You let it warm up first so conditions are stable before the real cooking begins.

---

### ONNX (Open Neural Network Exchange)

**Definition:** A standardized format for exporting trained models so they can run on any platform (not just PyTorch), often with optimized inference speed.

**Example:** The distilled student model is exported to `student.onnx` and loaded by the FastAPI server using ONNX Runtime, which is faster than PyTorch for inference.

**Analogy:** A universal power adapter. Your device (model) was built for one outlet (PyTorch), but the ONNX adapter lets it plug into any outlet (TensorFlow, web browsers, mobile) with optimized power delivery.

---

*End of document.*
