# Data Dictionary

## Hateful Memes Dataset

| Field | Type | Description |
|-------|------|-------------|
| id | int | Unique sample identifier |
| img | str | Relative path to meme image |
| text | str | OCR-extracted text from the meme |
| label | int | 0 = not hateful, 1 = hateful |

## Processed Features

| Feature | Shape | Description |
|---------|-------|-------------|
| pixel_values | [B, 3, 224, 224] | Normalized image tensor |
| input_ids | [B, 128] | Tokenized text IDs |
| attention_mask | [B, 128] | Token attention mask |
| labels | [B] | Binary labels |

## Embedding Outputs

| Feature | Shape | Description |
|---------|-------|-------------|
| vision_features | [B, D] | Vision encoder CLS token |
| text_features | [B, D] | Text encoder CLS token |
| fused_features | [B, D_fused] | Post-fusion representation |
| logits | [B, num_labels] | Classification logits |
