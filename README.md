# fursearch - Fursuit Character Recognition
Available on Telegram as [@fursearchbot](https://t.me/fursearchbot)
# Fursearch - Fursuit Character Recognition

Fursuit character recognition system using SAM3 + SigLIP. Identifies fursuit characters from photos by matching against a database of known characters.

## How It Works

```
Image -> SAM3 (detect "fursuiter head") -> Background Isolation -> SigLIP (embedding) -> FAISS (similarity search) -> Results
```

1. **SAM3** segments all fursuiters in the image using the text prompt `"fursuiter head"`
2. **Background Isolation** replaces the background with a solid color or blur to reduce noise
3. **SigLIP** generates an embedding for each isolated fursuiter crop
4. **FAISS** finds the most similar embeddings in the database
5. Results are returned with character names and confidence scores

## Requirements

- Python 3.10+
- CUDA GPU (recommended) or CPU
- ~4GB disk space for SAM3 model
- HuggingFace account with SAM3 access

## Installation

### 1. Clone and setup environment

```bash
git clone https://github.com/fursearch/fursearch
cd fursearch

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install package
uv pip install -e .
```

### 2. Get SAM3 access (required)

SAM3 requires HuggingFace authentication:

1. Create account at https://huggingface.co
2. Request access at https://huggingface.co/facebook/sam3
3. Wait for approval (usually hours)
4. Create a token at https://huggingface.co/settings/tokens
5. Login locally:

```bash
pip install huggingface_hub
hf auth login
```

### 3. Download SAM3 model (~3.5GB)

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='.')"
```

Or manually download `sam3.pt` from https://huggingface.co/facebook/sam3 and place in project root.

### 4. Verify installation

```bash
python -c "
from sam3_fursearch.models.segmentor import SAM3FursuitSegmentor
s = SAM3FursuitSegmentor()
print('SAM3 ready!')
"
```

Expected output:
```
Loading SAM3 model: sam3.pt on cuda
SAM3 loaded successfully - text prompts enabled!
SAM3 ready!
```

### 5. Troubleshooting

In case you see errors loading shared libraries, install these dependencies:
```bash
sudo apt-get update
sudo apt-get install libxcb1 libgl1
```

## Quick Start

### Identify a character

```bash
fursearch identify photo.jpg
fursearch identify photo.jpg --segment --concept "mascot"
```

### Add images for a new character

```bash
fursearch add -c "CharacterName" -s manual img1.jpg img2.jpg img3.jpg
fursearch add -c "CharacterName" -s nfc25 img1.jpg img2.jpg --save-crops
```

### Search by text description

```bash
fursearch search "blue fox with white markings"
fursearch search "red wolf" --top-k 10 --json
```

Note: Text search requires a dataset built with a CLIP or SigLIP embedder (SigLIP is the default). DINOv2 datasets do not support text search.

### Test segmentation on an image

```bash
fursearch segment photo.jpg
fursearch segment photo.jpg --output-dir ./crops/ --json
```

### View database entries

```bash
fursearch show --by-character "CharacterName"
fursearch show --by-id 42 --json
fursearch show --by-post "uuid-here"
```

### Bulk ingest images

```bash
# From directory structure: data_dir/character_name/*.jpg
fursearch ingest directory --data-dir ./characters/ --source manual

# From directory, sourced from tg_download
fursearch ingest directory --data-dir ./datasets/fursearch/tg_download --source tgbot

# From NFC25 dataset
fursearch ingest nfc25 --data-dir ./nfc25-fursuits/
```

### View database statistics

```bash
fursearch stats
fursearch stats --json
```

### Working with multiple datasets

Use `--dataset` (`-d` or `-ds`) to work with different datasets. Each dataset has its own `.db` and `.index` files.

```bash
# Add to default dataset (fursearch.db)
fursearch add -c "CharName" -s manual img1.jpg

# Add to a different dataset (validation.db)
fursearch -ds validation add -c "CharName" -s manual img1.jpg

# View stats for a specific dataset
fursearch -ds validation stats
```

### Validation workflow

Build a validation set and evaluate model accuracy:

```bash
# 1. Download validation images (auto-excludes main dataset, outputs to datasets/validation/nfc25/)
fursearch -ds validation download nfc25 -c "CharName" -m 5
fursearch -ds validation download nfc25 --all -m 2

# 2. Evaluate validation set against main dataset
fursearch evaluate
fursearch evaluate --from validation --against fursearch --top-k 5
fursearch evaluate --json
```

The evaluate command outputs:
- Top-1 and top-k accuracy
- Breakdown by source, preprocessing config, and character
- Confidence calibration (accuracy per confidence bucket)

### Combine datasets

Merge multiple datasets into a single target dataset (non-destructive, source datasets unchanged):

```bash
# Combine two datasets into one
fursearch combine fursearch validation --output merged

# Combine three datasets
fursearch combine fursearch validation test --output all_data
```

Duplicates (same `post_id` + `preprocessing_info` + `source`) are automatically skipped.

### Split a dataset

Extract a subset of a dataset by criteria (non-destructive, source dataset unchanged):

```bash
# Extract only nfc25 entries
fursearch split fursearch --output nfc25_only --by-source nfc25

# Extract specific characters
fursearch split fursearch --output subset --by-character "CharA,CharB"

# Split into shards (creates nfc25_split_0, nfc25_split_1)
fursearch split fursearch --output nfc25_split --by-source nfc25 --shards 2

# Combine filters
fursearch split fursearch --output char_a --by-source tgbot --by-character "CharA"
```

At least one filter (`--by-source` or `--by-character`) is required. Sharding uses `hash(post_id) % shards` for deterministic assignment.

### Run Telegram bot

```bash
# Single bot
export TG_BOT_TOKEN="your_bot_token"
python tgbot.py

# Multiple bots (comma-separated tokens, shared database)
export TG_BOT_TOKENS="token1,token2,token3"
python tgbot.py
```

## Python API

```python
from sam3_fursearch import FursuitIdentifier, FursuitIngestor, Config
from sam3_fursearch.api.identifier import discover_datasets
from sam3_fursearch.models.preprocessor import IsolationConfig
from PIL import Image

# --- Identification (read-only, one identifier per dataset) ---

# Single dataset (embedder is auto-detected from DB metadata)
identifier = FursuitIdentifier(
    db_path=Config.DB_PATH,
    index_path=Config.INDEX_PATH,
)

# Multiple datasets: create one identifier per dataset
datasets = discover_datasets()  # discovers *.db/*.index pairs in Config.BASE_DIR
identifiers = [
    FursuitIdentifier(db_path=db, index_path=idx)
    for db, idx in datasets
]

# Identify across all identifiers (segmentation is cached after the first)
image = Image.open("photo.jpg")
all_results = [ident.identify(image, top_k=5) for ident in identifiers]

# Merge results per segment
results = all_results[0] if all_results else []
for other in all_results[1:]:
    for seg, other_seg in zip(results, other):
        seg.matches.extend(other_seg.matches)
        seg.matches.sort(key=lambda x: x.confidence, reverse=True)
        seg.matches = seg.matches[:5]

for segment in results:
    print(f"Segment {segment.segment_index} at {segment.segment_bbox}:")
    for match in segment.matches:
        print(f"  {match.character_name}: {match.confidence:.1%}")

# Search by text (only works on identifiers with CLIP/SigLIP embedder)
text_identifiers = [i for i in identifiers if hasattr(i.pipeline.embedder, "embed_text")]
for ident in text_identifiers:
    results = ident.search_text("blue fox with white markings", top_k=5)
    for match in results:
        print(f"  {match.character_name}: {match.confidence:.1%}")

# Get statistics
stats = identifier.get_stats()
print(f"Database contains {stats['unique_characters']} characters")

# Combined stats across multiple identifiers
combined = FursuitIdentifier.get_combined_stats([i.get_stats() for i in identifiers])

# --- Ingestion (writes to a single dataset) ---

ingestor = FursuitIngestor()

# Or customize background isolation
isolation_config = IsolationConfig(
    mode="solid",                    # "solid", "blur", or "none"
    background_color=(128, 128, 128),  # Gray background
    blur_radius=25                   # For blur mode
)
ingestor = FursuitIngestor(isolation_config=isolation_config)

# Add images for characters
ingestor.add_images(["MyCharacter", "Zygote"], ["img1.jpg", "img2.jpg"])
```

### Using the segmentor directly

```python
from sam3_fursearch.models.segmentor import SAM3FursuitSegmentor
from PIL import Image

segmentor = SAM3FursuitSegmentor()
image = Image.open("photo.jpg")

# Segment with default concept ("fursuiter")
results = segmentor.segment(image)

for r in results:
    print(f"Found: bbox={r.bbox}, confidence={r.confidence:.2f}")
    r.crop.save(f"crop_{r.bbox[0]}.jpg")
    # Also available: r.mask, r.crop_mask for background isolation
```

### Using background isolation

```python
from sam3_fursearch.models.preprocessor import BackgroundIsolator, IsolationConfig
from PIL import Image
import numpy as np

# Configure isolation
config = IsolationConfig(mode="solid", background_color=(128, 128, 128))
isolator = BackgroundIsolator(config)

# Isolate foreground from background using a mask
crop = Image.open("crop.jpg")
mask = np.ones((crop.height, crop.width), dtype=np.uint8)  # Binary mask
isolated = isolator.isolate(crop, mask)
```

### Using mask storage

```python
from sam3_fursearch.storage.mask_storage import MaskStorage
import numpy as np

# Initialize mask storage (uses default fursearch_masks/ directory)
mask_storage = MaskStorage()

# Save a segmentation mask
mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
mask_path = mask_storage.save_mask(mask, "character_001", search=False)
print(f"Mask saved to: {mask_path}")

# Load a mask back
loaded_mask = mask_storage.load_mask(mask_path)

# Check if a mask exists
exists = mask_storage.mask_exists("character_001", search=False)
```

## Building a Database

### Option 1: Add images manually

```bash
# Add images for individual characters
fursearch add -c "CharName1" -s manual char1_*.jpg
fursearch add -c "CharName2" -s manual char2_*.jpg

# With segmentation (for multi-character photos)
fursearch add -c "CharName1" -s manual photo.jpg
```

### Option 2: Bulk ingest from directory

Organize images as `characters/CharacterName/*.jpg`:

```bash
fursearch ingest directory --data-dir ./characters/ --source manual
```

### Option 3: Index NFC25 database

If you have the NFC25 fursuit badge dataset:

```bash
fursearch ingest nfc25 --data-dir /path/to/nfc25-fursuits
```

## Storage Files

| File | Description |
|------|-------------|
| `sam3.pt` | SAM3 model weights (~3.5GB) |
| `fursearch.db` | SQLite database with detection metadata (default dataset) |
| `fursearch.index` | FAISS index with embeddings (default dataset) |
| `<name>.db` | Database for custom dataset (e.g., `validation.db`) |
| `<name>.index` | Index for custom dataset (e.g., `validation.index`) |
| `fursearch_crops/` | Saved crop images for debugging (when using `--save-crops`) |
| `fursearch_masks/` | Saved segmentation masks (when using `--save-crops`) |
| `datasets/<dataset>/<source>/` | Default download/ingest directory for non-default datasets |

These files are gitignored. Use `--dataset` (`-ds`) to switch between datasets.

## Configuration

Key settings in `sam3_fursearch/config.py`:

```python
# Dataset name (change this to rename all default files)
DEFAULT_DATASET = "fursearch"            # Used for db/index/crops/masks naming

# File paths (derived from DEFAULT_DATASET)
DEFAULT_DB_NAME = f"{DEFAULT_DATASET}.db"
DEFAULT_INDEX_NAME = f"{DEFAULT_DATASET}.index"
DEFAULT_CROPS_DIR = f"{DEFAULT_DATASET}_crops"
DEFAULT_MASKS_DIR = f"{DEFAULT_DATASET}_masks"

# Models
SAM3_MODEL = "sam3"                    # Model name
DEFAULT_EMBEDDER = "siglip"            # Default embedder (SigLIP)
SIGLIP_MODEL = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768                    # Embedding output dimension

# Detection
DEFAULT_CONCEPT = "fursuiter head"     # SAM3 text prompt
DETECTION_CONFIDENCE = 0.5             # Minimum confidence threshold
MAX_DETECTIONS = 10                    # Max segments per image

# Background isolation
DEFAULT_BACKGROUND_MODE = "solid"      # "solid", "blur", or "none"
DEFAULT_BACKGROUND_COLOR = (128, 128, 128)  # Gray background
DEFAULT_BLUR_RADIUS = 25               # Blur radius for "blur" mode

# Image processing
TARGET_IMAGE_SIZE = 630                # Resize target
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TG_BOT_TOKEN` | Telegram bot token (single bot) | For bot only |
| `TG_BOT_TOKENS` | Comma-separated tokens (multiple bots) | Alternative to above |
| `HF_TOKEN` | HuggingFace token | For SAM3 download |

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Test SAM3 segmentation on an image
fursearch segment photo.jpg --output-dir ./debug/

# Test with different concept
fursearch segment photo.jpg --concept "mascot" --json

# View database entries for debugging
fursearch show --by-character "CharName" --json
```

## Troubleshooting

### "FileNotFoundError: sam3.pt"

Download the SAM3 model:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='.')"
```

### "Access denied" when downloading SAM3

1. Ensure you've requested access at https://huggingface.co/facebook/sam3
2. Wait for approval email
3. Run `huggingface-cli login` and enter your token

### "CUDA out of memory"

SAM3 requires ~4GB VRAM. Options:
- Use a smaller batch size
- Process images sequentially
- Use CPU (slower): set `device="cpu"` in config

### Segmentation finds no fursuits

- Try different text prompts: `"mascot"`, `"costume"`, `"character"`
- Lower the confidence threshold in config
- Check if the image is clear and well-lit

## Device Support

Automatic device selection: CUDA → MPS → CPU

Force specific device:
```python
ingestor = FursuitIngestor(device="cuda")  # or "cpu", "mps"
```

## References

- [SAM3 Paper](https://arxiv.org/abs/2511.16719) - Segment Anything with Concepts
- [SAM3 on HuggingFace](https://huggingface.co/facebook/sam3)
- [Ultralytics SAM3 Docs](https://docs.ultralytics.com/models/sam-3/)
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) - Default embedder
- [DINOv2](https://github.com/facebookresearch/dinov2) - Alternative embedder (no text search)

## License

MIT

## TODO:
- [ ] Create a periodic schedule job to download and ingest new images from public sources
- [ ] I noticed that the submitted tgbot images do not come into effect until the bot is restarted
- [ ] Write the data to the metadata table on ingestion. Move the uploaded_by column to the metadata too. Then get_source_image_url and _get_page_url can entirely be implemented in the database.py
- [ ] Deprioritize low quality segments (small bbox compared to other fursuits, low confidence, low res, unusual aspect crop for the prompt etc)
- [ ] Fix the nsfw filter
- [ ] Number the segments on the drawn label
- [ ] Rename the confusing "segment" naming when replying to the user, instead use "fursuit"
- [ ] Create a periodic job to backup datasets & databases
- [ ] User interface for submitting new pictures should be nice and fun to use
- [ ] Make a webapp for in-browser fursuit identification (e.g. tag all images from my camera sdcard)
- [ ] Create an incentive for user data submission (game, find your fursuit parents, lookalikes, scores? etc) - make it beneficial or interesting to use
- [ ] Prioritize most recently seen fursuits when scoring results
- [ ] Allow users to say "@bot this is character_name" and "@bot this not character_name"
- [ ] Create nice icons for tg bots (I'm thinking of the fursuit with a labeled bounding box, so that it is obvious what this bot does)
- [ ] Call to action to submit your own pictures to the database
- [ ] Parse text (e.g. badges) and add to the search database
- [ ] Document how to set up the bot in group chats
- [ ] Make the bot respond to edited messages
- [ ] Add a link to the webapp to the bot hello message.
- [ ] Create call to action to upload some of the pictures you took to furtrack? Labels optional because I don't want to pollute furtrack with bad labels just yet. This will hurt indexing.
- [ ] Try to make it possible to tag several people in the picture correctly (left to right?)
- [ ] Create an alias database so that we can cross-validate characters appearing in e.g. nfc25, nfc26 and tgbot
- [ ] Deprioritize low frequency or very old fursuits?
- [ ] Overlapping segments - can we resolve them? Esp. relevant for manual tagging from left to right
- [ ] Find pictures with multiple fursuits and if we have at least 3, we can pick out which segment is the real tag by running self-similarity on a character pictures and mark all other segments as someone else. That way we reduce the noise.
- [ ] Run a self-similarity search on all database and cluster all potential segments to potentially assign it a tag, this is an extension of the previous point.
- [ ] Identify not just the "fursuiter head" but the whole body
- [ ] Create other preprocessing pipelines and assess their score on the validation dataset, such as black-and-white preprocessing, brightness normalization, etc.
- [ ] Make a feature to import the index data from another instance (to make it sync new fursuits across several running instances of the detector)
- [ ] Double-check if the /show command actually does not leak the user-submitted pictures from telegram
- [ ] Create an app that finds fursuit pictures in the camera roll (and maybe monitors it) and keeps a list of who you took pictures of
- [ ] Add social login to the bot hello message, ie login with google/furaffinity/barq/twitter etc so that we can link the fursuit to their social profile
- [ ] Use Filesystem API in the webapp to list contents of users' folder periodically without having to select individual photos.
- [ ] Add mode to sync the index and database periodically to an upstream S3 bucket, not sure how to do that exactly but maybe shard them into append-only pieces (kinda like wal or wal itself) and upload those periodically, and merge on the client?
- [ ] Use image infill to add occluded part of the fursuit head
- [ ] Add the full fursuit scanner to index on other parts of the body
- [ ] Use e.g. depth-anything to generate extra angles on the fursuit
- [ ] Run clip on the fursuit crop and create a text search index e.g. /search blue fox (done but not working)
- [ ] Run a SAM2 + clip on each fragment instead of heavy SAM3 to segment where the fursuit head is.
