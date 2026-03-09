# BetaFinder CNX — Architecture Documentation

## Overview

BetaFinder CNX is a climbing beta finder system that uses AI-powered image similarity search to help climbers find route information from Instagram posts. The system combines CLIP image embeddings, FAISS vector indexing, and CLIP-based wall classification.

**Tech Stack:**
- Image Embedding: CLIP (Contrastive Language-Image Pre-training)
- Vector Search: FAISS (Facebook AI Similarity Search) with IndexFlatIP
- Web UI: Streamlit
- Data Source: Instagram (instaloader)
- Configuration: YAML-based centralized config
- Language: Python 3.10+

---

## Directory Structure

```
betafinder-cnx/
├── config/
│   └── config.yaml                  # Central YAML configuration
├── src/
│   ├── __init__.py
│   ├── config.py                    # Config loader with helpers
│   ├── logger.py                    # Centralized logging
│   ├── scrape.py                    # Instagram scraper
│   ├── filter.py                    # Wall image classifier
│   ├── embed.py                     # CLIP embedding & FAISS indexing
│   ├── search.py                    # Similarity search engine
│   ├── update.py                    # Auto-update pipeline
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py           # Image I/O utilities
├── app.py                           # Streamlit web UI
├── scrape.py, filter.py, embed.py   # Root wrappers (backward compatibility)
├── search.py, update.py             # Root wrappers
├── data/
│   ├── images/                      # Downloaded Instagram images
│   ├── gym_index.json               # Metadata + wall classification
│   ├── contributors.json            # Community climber accounts
│   ├── filter_cache.json            # Wall scores cache
│   ├── embeddings.pkl               # CLIP embeddings (deprecated, now in FAISS)
│   ├── faiss.index                  # FAISS vector index
│   └── faiss.paths.json             # Filename → FAISS ID mapping
├── doc/
│   ├── plan.md                      # Original project plan
│   ├── REFACTORING_SUMMARY.md       # Refactoring changes
│   └── architecture.md              # This file
├── .claude/
│   ├── launch.json                  # Dev server configurations
│   └── plans/                       # Planning documents
├── requirements.txt                 # Python dependencies
└── README.md                        # Project README
```

---

## Core Modules

### 1. **config.py** — Configuration Management

Centralized configuration loading from `config/config.yaml`.

**Key Functions:**
- `load_config(path)` — Load and cache YAML configuration
- `get_gym_names()` — List of gym keys (alpine, mainwall, progression, community)
- `get_gym_info(gym_key)` — Get gym details (name, instagram handle, color)
- `get_path(key)` — Resolve paths from config (creates parent directories)
- `get_nested(path)` — Access nested config values via dot notation (e.g., "embedding.model_name")
- `get_instagram_handle(gym_key)` — Instagram username for gym

**Example Usage:**
```python
from src.config import load_config, get_path, get_nested

config = load_config()
model = get_nested("embedding.model_name")  # "ViT-B-32"
data_dir = get_path("data_dir")             # Path("data/images")
```

---

### 2. **logger.py** — Centralized Logging

Setup logging with config-driven level and format.

**Key Functions:**
- `setup_logger(name, level=None)` — Create logger with config settings

**Example Usage:**
```python
from src.logger import setup_logger

log = setup_logger(__name__)
log.info("Processing image...")
log.error("Failed to embed image")
```

---

### 3. **scrape.py** — Instagram Data Collection

Downloads images and metadata from Instagram gym accounts and community climber accounts.

**Sources:**
- **Official Gyms:** Alpine Outpost, Main Wall CNX, Progression Vertical (configured in config.yaml)
- **Community Contributors:** Climber accounts (contributors.json) — community-maintained list

**Key Functions:**
- `scrape_gyms(gym_list, limit, delay)` — Scrape official gym accounts
- `scrape_contributors(usernames, limit, delay)` — Scrape community climbers
- `load_contributors()` — Load community account list
- `add_contributor(username, note, gyms)` — Add new contributor account

**Features:**
- Beta keyword filtering (English, Thai, climbing grades)
- Image relevance scoring
- Delay between requests (respectful scraping)
- Metadata extraction (date, captions, engagement)

**Output:**
```json
{
  "source_type": "official|contributor",
  "source_key": "alpine|username",
  "gym": "alpine|climb.with.poom",
  "filename": "data/images/official/alpine/ABC123_0.jpg",
  "caption": "V3 left side route",
  "date": "2026-03-08",
  "is_relevant": true
}
```

---

### 4. **filter.py** — Wall Image Classification

CLIP-based zero-shot image classification to distinguish climbing wall photos from marketing content (ads, events, selfies).

**Classification:**
- **Wall:** Climbing wall with holds, routes, volumes visible
- **Non-wall:** Gym announcements, product ads, selfies, events

**Text Prompts:** "climbing wall", "bouldering wall", "climbing gym", "indoor climbing", "route on wall"

**Key Functions:**
- `load_clip_model(model_name, device)` — Load CLIP model
- `get_text_embeddings(model, device)` — Create text embeddings for wall prompts
- `score_wall_image(image, model, preprocess, device, text_embeds)` — Score single image
- `filter_images(metadata, cache, model, preprocess, device, text_embeds, threshold)` — Classify all images
- `load_filter_cache()` / `save_filter_cache()` — Cache wall scores for fast re-runs

**Features:**
- Zero-shot learning (no fine-tuning required)
- Caching system (filter_cache.json) for fast re-runs
- Configurable threshold (0.05 = very permissive, 0.6 = strict)
- Per-gym statistics reporting

**Output:**
```json
{
  ...image metadata...,
  "is_wall": true,        # Boolean classification
  "wall_score": 0.087     # Float [-1, 1] cosine similarity
}
```

---

### 5. **embed.py** — Image Embedding & FAISS Indexing

CLIP image embedding and FAISS vector index creation for similarity search.

**Pipeline:**
1. Load images from gym_index.json
2. Encode images with CLIP (ViT-B-32 or ViT-L-14)
3. L2 normalize vectors
4. Create FAISS IndexFlatIP index (inner product = cosine similarity for normalized vectors)
5. Save index and filename mapping

**Key Functions:**
- `load_clip_model(model_name, pretrained, device)` — Load CLIP model
- `encode_image(image_path, model, preprocess, device)` — Embed single image
- `build_faiss_index(model_name, force_rebuild)` — Build index from all wall images
- `load_faiss_index()` — Load existing index

**Features:**
- Batch processing for efficiency
- GPU support (auto-detects CUDA)
- Fallback to CPU for compatibility
- Corrupt image handling
- Progress bars

**Output:**
- `data/faiss.index` — FAISS vector index (~50MB for 1000 images)
- `data/faiss.paths.json` — Mapping {image_path: faiss_id}

---

### 6. **search.py** — Similarity Search Engine

Find similar climbing routes using CLIP embeddings and FAISS index.

**Search Process:**
1. Encode query image with CLIP
2. Search FAISS index for top-K most similar images
3. Filter by gym (optional)
4. Return results with similarity scores and metadata

**Key Functions:**
- `encode_image(image_path, model, preprocess, device)` — Embed query image
- `search(image_path, top_k, gym_filter, model_name, pretrained)` — Find similar images

**Features:**
- Gym filtering (Alpine, MainWall, Progression, or all)
- Model selection (ViT-B-32 fast vs ViT-L-14 accurate)
- Configurable result count (3-20)
- Similarity percentage scoring
- Instagram URL linking

**Output:**
```python
[
  {
    "filename": "data/images/official/alpine/ABC123_0.jpg",
    "gym": "alpine",
    "caption": "V3 left side",
    "date": "2026-03-08",
    "score": 0.8234,  # Cosine similarity [0, 1]
    "url": "https://instagram.com/..."
  },
  ...
]
```

---

### 7. **update.py** — Auto-Update Pipeline

Scheduled auto-scraper for incremental dataset updates (runs via cron or manual trigger).

**Pipeline:**
1. **Scrape:** Pull new posts from Instagram sources
2. **Filter:** Classify new images (walls vs non-walls)
3. **Embed:** Add new images to FAISS index
4. **Log:** Save update statistics

**Key Functions:**
- `update_dataset(args)` — Run full pipeline
- `step_scrape(stats, args)` — Scrape new images
- `step_filter(stats)` — Wall classification
- `step_embed(stats)` — CLIP embedding & FAISS update
- `save_update_log(stats)` — Log statistics

**Features:**
- Incremental updates (only new images)
- Skip/force individual steps
- Contributors-only mode
- Configurable delays and limits
- Error tracking and reporting
- JSON log output

**Usage:**
```bash
python update.py                          # Full update
python update.py --contributors-only      # Community accounts only
python update.py --skip-scrape --force-embed  # Re-embed without scraping
python update.py --dry-run                # Show what would run
```

---

### 8. **app.py** — Streamlit Web UI

Interactive web interface for searching climbing routes by photo.

**Features:**
- **Photo Upload:** Drag-drop or file picker for wall photos
- **Gym Filter:** Select specific gyms or search all
- **Model Selection:** Choose between fast (ViT-B-32) or accurate (ViT-L-14) CLIP model
- **Live Search:** Real-time similarity search with progress indicator
- **Results Display:** Ranked list with similarity %, gym badges, dates, captions, Instagram links
- **Sidebar Stats:** Dataset size, gym breakdown, tips, about info
- **Theme:** Climbing gym aesthetic (dark wall, colorful holds)

**UI Elements:**
- Header with climbing emoji and project name
- 4-column gym selector
- Photo upload with preview and metadata
- Model/results settings sidebar
- Results metrics (best match, average score, count)
- Per-result cards with ranking, emoji, similarity bar, metadata

**Integration:**
- Imports config from src.config
- Uses search.py for similarity queries
- Caches gym_index for performance
- Respects all config.yaml settings

---

## Data Flow

### Initialization Flow
```
config.yaml
    ↓
config.py (load_config)
    ↓
scrape.py (scrape_gyms, scrape_contributors)
    ↓
data/gym_index.json (image metadata)
    ↓
filter.py (wall classification)
    ↓
gym_index + is_wall, wall_score
    ↓
embed.py (CLIP embedding)
    ↓
FAISS index + paths mapping
    ↓
[System Ready]
```

### Search Flow
```
user uploads photo
    ↓
app.py (file_uploader)
    ↓
search.py (encode_image)
    ↓
CLIP model (image → embedding)
    ↓
FAISS index (similarity search)
    ↓
filter by gym (optional)
    ↓
rank by similarity score
    ↓
fetch metadata from gym_index.json
    ↓
display results to user
```

### Update Flow
```
update.py --trigger
    ↓
step_scrape: Instagram sources → new images
    ↓
gym_index.json (add new entries)
    ↓
step_filter: CLIP wall classification
    ↓
update gym_index with is_wall, wall_score
    ↓
step_embed: encode new images, update FAISS
    ↓
update faiss.index, faiss.paths.json
    ↓
save_update_log (JSON stats)
```

---

## Configuration System

### config.yaml Structure

**Gyms Section:**
```yaml
gyms:
  alpine:
    name: "Alpine Outpost"
    instagram: "the_alpine_outpost"
    color: "#4CAF50"
```

**Paths Section:**
```yaml
paths:
  data_dir: "data/images"
  index_file: "data/gym_index.json"
  faiss_file: "data/faiss.index"
  # ... etc
```

**Scraping Section:**
```yaml
scraping:
  default_limit: 100
  default_delay: 2.0
  instaloader:
    download_videos: false
    # ... Instagram scraper config
  beta_keywords:
    english: [climb, boulder, route, beta, wall, grade]
    thai: [ปีน, บอลเดอร์]
    grades: [v0, v1, v2, ...]
```

**Embedding Section:**
```yaml
embedding:
  model_name: "ViT-B-32"  # or ViT-L-14
  pretrained: "openai"
  default_batch_size: 16
  normalize: true
```

**Wall Filter Section:**
```yaml
wall_filter:
  enabled: true
  model: "ViT-B-32"
  threshold: 0.05          # 0.05=permissive, 0.6=strict
  exclude_non_walls: true  # Skip non-walls in FAISS
```

**Search Section:**
```yaml
search:
  default_top_k: 5         # Default result count
  max_top_k: 20            # Maximum allowed
  oversample_factor: 3     # Fetch 3x before gym filter
```

---

## Model Details

### CLIP Models

**ViT-B-32 (Fast)**
- Input: 224×224 RGB images
- Output: 512-dim vector
- Speed: ~10s per image on CPU, ~0.1s on GPU
- Memory: ~350MB model, ~1MB per image

**ViT-L-14 (Accurate)**
- Input: 224×224 RGB images
- Output: 768-dim vector
- Speed: ~30s per image on CPU, ~0.3s on GPU
- Memory: ~600MB model, ~1.5MB per image

### FAISS Index Type

**IndexFlatIP (Flat Inner Product)**
- Inner product on L2-normalized vectors = cosine similarity
- Exact search (no approximation)
- O(n) search complexity — suitable for <100k images
- Memory: ~2KB per dimension per image (~1MB per image for 512-dim)

---

## Performance Metrics

### Dataset Statistics (as of 2026-03-08)
- **Total Images:** 1,342
- **Wall Images:** 721 (53.7%)
- **Non-wall Images:** 621 (46.3%)
- **Gyms:** 8 (3 official + 5 community)
- **Date Range:** Varies by source

### Speed Benchmarks
| Task | CPU | GPU |
|------|-----|-----|
| Scrape 100 posts | ~10 min | N/A |
| Filter 1,342 images | ~20 sec (cached) | N/A |
| Embed 1,342 images | ~30 min | ~3 min |
| Single search | ~5 sec | ~0.5 sec |

### Storage
- Images: ~500MB (depending on resolution)
- FAISS index: ~2MB (1,342 images × 512 dims)
- Metadata (JSON): ~2MB
- Cache: ~500KB

---

## Extension Points

### Adding New Gyms
1. Edit `config/config.yaml` — Add gym to `gyms` section
2. Run `python scrape.py --gym newgym`
3. Run `python filter.py && python embed.py` to update index

### Custom Search Filters
- Modify `search.py` to add gym selector logic
- Add filters to query (e.g., date range, difficulty grade)
- Update Streamlit UI form

### Model Swapping
- Change `config.yaml` — `embedding.model_name: "ViT-L-14"`
- Run `python embed.py --rebuild` to re-embed all images
- Search automatically uses new model

### Wall Filter Tuning
- Adjust `config.yaml` — `wall_filter.threshold`
- Run `python filter.py --rebuild` to re-classify
- Lower threshold = more lenient, higher = stricter

---

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Create conda environment (recommended)
conda create -n beta-finder python=3.10
conda activate beta-finder
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# App available at http://localhost:8501
```

### Scheduled Updates (Cron)
```bash
# Update every day at 2 AM
0 2 * * * cd /path/to/betafinder && python update.py --contributors-only

# Full update (scrape + filter + embed) every Sunday at midnight
0 0 * * 0 cd /path/to/betafinder && python update.py
```

### Backups
- Backup `data/` directory regularly
- Critical files:
  - `gym_index.json` (metadata)
  - `faiss.index` (vector search index)
  - `contributors.json` (community account list)

---

## Troubleshooting

### Issue: "No module named 'torch'"
- **Fix:** `pip install torch torchvision`
- Or use conda environment: `conda activate beta-finder`

### Issue: "FAISS index corrupted"
- **Fix:** Rebuild with `python embed.py --rebuild`
- Backup `faiss.index` before rebuilding

### Issue: Search results irrelevant
- **Check:** Are non-walls being filtered? Run `python filter.py --rebuild`
- Try different model: Change `embedding.model_name` in config.yaml
- Increase `search.oversample_factor` for better ranking

### Issue: Instagram scraper rate limited
- **Fix:** Increase `scraping.default_delay` in config.yaml
- Or run `python scrape.py --limit 20` (fewer posts per account)

---

## Future Enhancements

1. **Multi-language Support** — Expand Thai/English translations
2. **Grade Filtering** — Filter results by climbing grade (V0-V10)
3. **Route Recognition** — ML model to detect and label specific routes
4. **Mobile App** — React Native or Flutter mobile version
5. **Community Features** — User accounts, saved routes, comments
6. **Real-time Collaboration** — Live route marking/discussion
7. **Hardware Optimization** — ONNX models for mobile/edge inference
8. **API Server** — REST API for external integrations

---

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make changes following the existing code style
4. Test with `python -m pytest` (when tests are added)
5. Submit a pull request

---

## License

[Add your license information here]

---

## Contact

Built for the Chiang Mai climbing community.

GitHub: https://github.com/spatipan/beta-finder
