# BetaFinder CNX

**A climbing beta (route) finder using CLIP image embeddings + FAISS similarity search to discover similar wall photos from Instagram climbing gym accounts in Chiang Mai.**

![Status](https://img.shields.io/badge/status-Phase%202%20Complete-brightgreen)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![Node](https://img.shields.io/badge/node-18+-blue)

> Find your beta on the wall. — ค้นหา beta จากรูปผนัง

## 🎯 Overview

BetaFinder CNX helps climbers discover new routes and bouldering problems by searching for similar wall photos in a database of gym images. Upload a photo of a wall you like, and find similar ones posted on Instagram by Chiang Mai climbing gyms.

**Key Insight**: Climbers tag gym Instagram accounts in every post, making tagged posts feeds the richest source of community content—enabling **6x content growth** vs. official accounts alone.

### Supported Sources

| Type | Account | Gym |
|------|---------|-----|
| Official | @the_alpine_outpost | Alpine Outpost |
| Official | @mainwallcnx | Main Wall |
| Official | @progressionvertical | Progression Vertical |
| Community | `data/contributors.json` | User-added beta creators |
| **NEW** | **Tagged Posts** | Anywhere gym is tagged (6x growth!) |

---

## 🚀 Quick Start

### 1. Install Dependencies

**Using uv** (recommended, faster and more reliable):
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies and create virtual environment
uv sync

# Node dependencies (for React dev server)
npm install

# Set Instagram credentials (optional, to avoid rate limits)
export INSTALOADER_USER="your_username"
export INSTALOADER_PASS="your_password"
```

**Or using pip** (traditional approach):
```bash
pip install -r requirements.txt
npm install
export INSTALOADER_USER="your_username"
export INSTALOADER_PASS="your_password"
```

### 2. Scrape, Embed & Index

```bash
# Scrape images from gyms + contributors + tagged posts
python scrape.py

# Build embeddings & FAISS index
# Default: CLIP ViT-B-32
python embed.py

# Or use other embedding methods (see Embedding Methods below)
python embed.py --model ViT-L-14 --pretrained openai      # CLIP ViT-L-14
python embed.py --backbone dinov2_vitb14                  # DINOv2 ViT-B-14
python embed.py --backbone superpoint                     # SuperPoint+SuperGlue (GPU)
python embed.py --backbone sift                           # SIFT (CPU, no GPU needed)

# Optional: Filter to keep only climbing walls
python filter.py
```

### 3. Start the Application

**Terminal 1 — FastAPI Backend**
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2 — React Frontend**
```bash
npm run dev
```

Then open **http://localhost:5173**

### 4. Search

Upload a wall photo and click "Search". Results show similar climbs ranked by similarity score.

---

## 🏗️ Architecture

### Full-Stack System

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                    │
│              http://localhost:5173                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /api/search (FormData)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (uvicorn)                   │
│              http://localhost:8000/api/*                    │
│  - POST /search (image upload + CLIP embedding)            │
│  - GET /stats (index statistics)                           │
│  - GET /thumb/{path} (serve local images)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ┌──────────────┐  ┌──────────────┐
        │ FAISS Index  │  │  gym_index   │
        │              │  │  .json       │
        │ Vector DB    │  │  (metadata)  │
        └──────────────┘  └──────────────┘
```

### Data Pipeline

```
Instagram Scraping (official + contributors + tagged posts)
    ↓
[scrape.py] → downloads images, captions, dates, extract keyframes from Reels
    ↓
Local Images + Metadata (source_type, media_type, keyframe_index)
    ↓
[embed.py] → CLIP embeddings (ViT-B-32 or ViT-L-14)
    ↓
[filter.py] → classify walls vs. non-walls (optional)
    ↓
FAISS Index + Metadata → Ready for search
```

### File Structure

```
beta-finder-cnx/
├── src/
│   ├── scrape.py       # Instagram image scraper (with keyframe extraction)
│   ├── embed.py        # Embeddings + FAISS indexing
│   ├── embeddings.py   # Shared embedding functions (CLIP, DINOv2, SIFT, SuperPoint)
│   ├── search.py       # Similarity search function (auto-detects model)
│   ├── filter.py       # Wall classification (zero-shot)
│   ├── update.py       # Auto-update pipeline
│   ├── discover.py     # Snowball account discovery (BFS + @mentions)
│   ├── feedback.py     # Feedback loop for fine-tuning
│   ├── config.py       # Config loader
│   └── logger.py       # Logging setup
│
├── src/ui/
│   ├── App.jsx         # React main component
│   ├── index.html      # HTML entry point
│   └── main.jsx        # React DOM mount
│
├── api.py              # FastAPI server (HEIC support, endpoints)
│
├── config/
│   └── config.yaml     # YAML configuration
│
├── data/
│   ├── images/
│   │   ├── official/{alpine|mainwall|progression}/
│   │   └── contributor/{username}/
│   ├── gym_index.json      # Image metadata
│   ├── contributors.json   # Community contributors list
│   ├── feedback.json       # User feedback (for fine-tuning)
│   ├── graph.json          # Account mention graph (discovery)
│   ├── faiss.index         # FAISS vector index (binary)
│   ├── faiss.paths.json    # Image path list for index lookup
│   └── faiss.model.json    # Model metadata (auto-detected by search.py)
│
├── pyproject.toml      # Project metadata and dependencies (uv-managed)
├── uv.lock             # Dependency lock file (uv)
├── requirements.txt    # Alternative: Python dependencies (pip)
├── package.json        # Node dependencies
├── vite.config.js      # Vite configuration
├── CLI_GUIDE.md        # Complete CLI command reference
└── README.md           # This file
```

---

## ✨ Features

### Phase 1 ✅ Core Search
- **Multiple Embedding Methods** — CLIP, SigLIP, EVA-CLIP, DINOv2, SIFT, SuperPoint+SuperGlue
- **FAISS Vector Index** — Fast similarity search with automatic model matching
- **Instagram Scraping** — Official gyms + community contributors
- **CLI Tools** — scrape, embed, search, filter, update, discover

### Phase 2 ✅ Accuracy & Discovery
- **Phase 2.1: Tagged Posts Feed** — Auto-scrape where gyms are tagged (6x growth!)
- **Feedback Loop** — Users mark results as helpful/unhelpful (👍/👎)
- **Keyframe Extraction** — Extract 4 frames per Instagram Reel for better coverage
- **Snowball Discovery** — Auto-discover new beta creators via BFS + @mention extraction
- **Full-Stack UI** — React + Vite frontend + FastAPI backend (HEIC support)
- **Comprehensive Docs** — CLI_GUIDE.md with all commands and workflows

### Phase 3 🔜 Advanced
- **Fine-tuning** — Retrain CLIP on climbing-specific feedback pairs (200+ required)
- **Relevance Boosting** — Emphasize highly-rated results in rankings
- **Batch Updates** — Scheduled daily/weekly scraping + indexing
- **User Accounts** — Sessions and personalized rankings

---

## 🎮 Usage

### Command-Line Workflow

```bash
# 1. Scrape images (all sources: official + contributors + tagged)
python scrape.py

# 2. Build embeddings & index (choose one embedding method)
python embed.py                          # Default: CLIP ViT-B-32
# python embed.py --model ViT-L-14 --pretrained openai  # CLIP ViT-L-14 (better accuracy)
# python embed.py --backbone dinov2_vitb14              # DINOv2 (texture/structure)
# python embed.py --backbone superpoint                 # SuperPoint (GPU, best localization)
# python embed.py --backbone sift                       # SIFT (CPU-only)

# 3. Search via CLI (automatically uses same model as index)
python search.py wall_photo.jpg --top 5

# OR start the full-stack app:
# Terminal 1: API
uvicorn api:app --reload --port 8000
# Terminal 2: Frontend
npm run dev
# Then open http://localhost:5173
```

### Scraping Modes

```bash
# Tagged posts only (newest content)
python scrape.py --mode tagged --limit 100

# Both account posts + tagged posts
python scrape.py --mode both --limit 200

# Account posts only (default for official)
python scrape.py --mode posts

# Discover new contributors
python discover.py --depth 2 --min-score 0.3
```

### Search Options

```bash
# Top 5 results (default)
python search.py photo.jpg

# Top 10 results
python search.py photo.jpg --top 10

# Filter by specific gym
python search.py photo.jpg --gym alpine

# Open in browser
python search.py photo.jpg --open

# JSON output (for scripts)
python search.py photo.jpg --json
```

**See [CLI_GUIDE.md](./CLI_GUIDE.md) for complete command reference.**

---

## 🔌 API Reference

### POST `/api/search`

```bash
curl -X POST http://localhost:8000/api/search \
  -F "file=@wall_photo.jpg" \
  -F "gym=alpine" \
  -F "topK=5" \
  -F "model=ViT-B-32"
```

**Response:**
```json
[
  {
    "rank": 1,
    "score": 0.842,
    "filename": "data/images/official/alpine/ABC123_0.jpg",
    "gym": "alpine",
    "url": "https://instagram.com/p/ABC123/",
    "caption": "Fun slopers today! 🧗",
    "date": "2026-03-08",
    "mediaType": "image",
    "sourceType": "official",
    "username": "the_alpine_outpost",
    "thumbnailUrl": "http://localhost:8000/api/thumb/..."
  }
]
```

### GET `/api/stats`

```bash
curl http://localhost:8000/api/stats
```

Returns index statistics (total images, per-gym breakdown, model used, etc).

### GET `/api/thumb/{encoded_path}`

Serve local image thumbnails (max 800x600px).

---

## 🧠 Embedding Methods

BetaFinder supports 6 different embedding methods with different speed/accuracy trade-offs:

| Method | Type | Dim | Speed | Accuracy | GPU | Notes |
|--------|------|-----|-------|----------|-----|-------|
| **CLIP ViT-B-32** | Semantic | 512 | ⚡⚡⚡ | ⭐⭐⭐ | Optional | Default, fast & accurate |
| **CLIP ViT-L-14** | Semantic | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Optional | Better accuracy, slower |
| **SigLIP ViT-B-16** | Semantic | 512 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Optional | Competitive with CLIP |
| **EVA-CLIP E-14** | Semantic | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | Optional | Highest accuracy, slowest |
| **DINOv2 ViT-B-14** | Self-supervised | 768 | ⚡⚡ | ⭐⭐⭐ | Recommended | Good for texture/structure |
| **SIFT** | Local features | 128* | ⚡⚡ | ⭐⭐ | None | CPU-only, traditional CV |
| **SuperPoint+SuperGlue** | Local features | 256* | ⚡ | ⭐⭐⭐ | Required | GPU accelerated, best localization |

*Fixed to 128-dim and 256-dim via mean aggregation of variable-length descriptors

### Usage

```bash
# CLIP models (semantic embeddings)
python embed.py                                    # ViT-B-32 (default, fast)
python embed.py --model ViT-L-14 --pretrained openai  # ViT-L-14 (accurate)

# SigLIP models (better than CLIP, open-source)
python embed.py --model ViT-B-16-SigLIP --pretrained webli
python embed.py --model ViT-SO400M-14-SigLIP --pretrained webli  # Largest, slowest

# EVA-CLIP (highest accuracy semantic embeddings)
python embed.py --model EVA02-E-14 --pretrained laion2b_s4b_b115k

# DINOv2 (self-supervised, good for texture/structure)
python embed.py --backbone dinov2_vitb14
python embed.py --backbone dinov2_vitl14         # Larger model

# SIFT (CPU-only, traditional computer vision)
python embed.py --backbone sift

# SuperPoint+SuperGlue (GPU required, best for precise localization)
python embed.py --backbone superpoint            # Falls back to SIFT on CPU-only
```

### Automatic Model Detection

When you run `search.py`, it automatically detects which model was used to build the index and loads the matching model. This ensures consistency:

```bash
# search.py will automatically use the same model that built the index
python search.py wall_photo.jpg --top 5

# Or override the model if needed (expert use)
python search.py wall_photo.jpg --model ViT-L-14 --pretrained openai
```

---

## ⚙️ Configuration

### Dependency Management

This project uses **uv** for fast, reliable dependency management:

```bash
# Install dependencies and create venv
uv sync

# Run scripts in venv context
uv run python scrape.py
uv run python embed.py
uv run python search.py photo.jpg

# Add a new dependency
uv add package-name

# Update all dependencies
uv update
```

See `pyproject.toml` for:
- Project metadata and all dependencies
- Optional dependency groups (dev, gpu)
- Development tools configuration

See `config/config.yaml` for:
- Gym Instagram handles
- Scraping modes (official, contributors, tagged posts)
- Default embedding model (can override via CLI)
- Data paths
- Keyframe extraction settings
- Discovery parameters

### Environment Variables

```bash
export INSTALOADER_USER="your_username"
export INSTALOADER_PASS="your_password"
export BETAFINDER_CONFIG="path/to/config.yaml"
export BETAFINDER_DATA_DIR="/path/to/data"
```

---

## 🐛 Troubleshooting

**HEIC/iPhone photos not working:**
```bash
pip install pillow-heif
```

**401 Unauthorized errors:**
```bash
export INSTALOADER_USER="your_username"
export INSTALOADER_PASS="your_password"
python scrape.py
```

**FAISS dimension mismatch (embedding vector sizes don't match):**
```bash
# Rebuild index with consistent model
python embed.py --rebuild
```

**SuperPoint not available (kornia-moons missing):**
```bash
pip install kornia-moons
# Falls back to SIFT automatically if CUDA not available
```

**Out of memory during embedding:**
```bash
# Reduce batch size (default: auto-detected)
python embed.py --batch 8
# Or with uv:
uv run python embed.py --batch 8
```

**Dependencies out of sync:**
```bash
# Regenerate lock file and venv
uv sync
```

**Model mismatch between embed and search:**
```bash
# Automatic detection: search.py reads faiss.model.json
# If mismatch occurs, rebuild:
python embed.py --rebuild --model ViT-L-14 --pretrained openai
```

**Port already in use:**
```bash
uvicorn api:app --port 8001
# or kill the process
lsof -ti :8000 | xargs kill -9
```

See [CLI_GUIDE.md](./CLI_GUIDE.md) for more troubleshooting tips and [UV_GUIDE.md](./UV_GUIDE.md) for dependency management details.

---

## 🗺️ Project Roadmap

### ✅ Phase 1: Core Search
- [x] Instagram scraping
- [x] CLIP embeddings
- [x] FAISS indexing
- [x] CLI tools

### ✅ Phase 2: Accuracy & Discovery
- [x] Tagged posts feed (6x growth)
- [x] Feedback loop UI
- [x] Keyframe extraction from Reels
- [x] Snowball account discovery
- [x] React + FastAPI full-stack
- [x] CLI documentation

### 🔜 Phase 3: Advanced
- [ ] Fine-tune CLIP on feedback pairs
- [ ] User accounts & sessions
- [ ] Personalized rankings
- [ ] Scheduled updates
- [ ] Mobile app (React Native)

---

## 📧 Contact

Built for the Chiang Mai climbing community.

**Last updated:** 2026-03-09