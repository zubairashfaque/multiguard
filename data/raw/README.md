# Raw Data

Place raw datasets here. The primary dataset is the **Facebook Hateful Memes Challenge**.

## Hateful Memes Dataset

- **Source:** [Facebook AI Research](https://ai.facebook.com/tools/hatefulmemes/)
- **Size:** ~10,000 memes (image + text pairs)
- **Labels:** Binary (hateful / not-hateful)
- **Access:** Requires approval from Facebook Research

### Download Instructions

1. Request access at the dataset page
2. Download and extract to `data/raw/hateful_memes/`
3. Expected structure:
   ```
   data/raw/hateful_memes/
   ├── img/           # Meme images
   ├── train.jsonl    # Training annotations
   ├── dev_seen.jsonl # Validation (seen)
   ├── dev_unseen.jsonl
   └── test_seen.jsonl
   ```
4. Run `python scripts/ingest_data.py` to preprocess
