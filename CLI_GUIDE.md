# BetaFinder CNX — CLI Command Guide

Complete reference for all command-line tools in the BetaFinder project.

## Setup

### Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies (for React dev)
npm install
```

### Set Instagram Credentials (Optional)

To avoid 401 Unauthorized errors when scraping, set your Instagram credentials:

```bash
# Option 1: Environment variables (recommended)
export INSTALOADER_USER="your_username"
export INSTALOADER_PASS="your_password"

# Option 2: Save to .env file (will be loaded automatically)
echo "INSTALOADER_USER=your_username" >> .env
echo "INSTALOADER_PASS=your_password" >> .env
```

> **Note:** Credentials are optional. Instaloader will try to work without login first, and only use credentials if needed to access private content or avoid rate limits.

---

## Running the Application

### Start the Full Stack

**Terminal 1 — API Server (Python/FastAPI)**
```bash
conda activate beta-finder  # or your python env
uvicorn api:app --reload --port 8000
```

**Terminal 2 — React Dev Server (Node/Vite)**
```bash
npm run dev
```

Then open: **http://localhost:5173**

---

## Python CLI Tools

### 1. **scrape.py** — Fetch images from Instagram

```bash
# Scrape all official gyms + contributors
python scrape.py

# Scrape one gym with custom limit
python scrape.py --gym alpine --limit 50

# Scrape official gyms only (skip contributors)
python scrape.py --gym all

# Scrape just contributor accounts
python scrape.py --contributors-only

# Scrape one contributor
python scrape.py --contributor climber_username

# Add a new contributor and scrape immediately
python scrape.py --add-contributor new_username --note "Posts mostly alpine beta"

# Use community-tagged posts only (new Phase 2.1!)
python scrape.py --mode tagged --limit 100

# Get both own + community tagged posts
python scrape.py --mode both --limit 200

# List all contributors
python scrape.py --list-contributors

# Add contributor without scraping
python scrape.py --add-contributor username --no-scrape
```

### 2. **embed.py** — Build CLIP embeddings & FAISS index

```bash
# Build index (default: ViT-B-32 model, faster)
python embed.py

# Rebuild with ViT-L-14 (more accurate but slower)
python embed.py --model ViT-L-14

# Force rebuild (clear cache)
python embed.py --rebuild

# Show stats
python embed.py --stats
```

### 3. **search.py** — Query the index with a photo

```bash
# Search with defaults (top 5, all gyms)
python search.py wall_photo.jpg

# Get top 10 results
python search.py wall_photo.jpg --top 10

# Filter by gym
python search.py wall_photo.jpg --gym alpine

# Open results in browser
python search.py wall_photo.jpg --open

# Output as JSON (for scripts)
python search.py wall_photo.jpg --json

# Use specific model
python search.py wall_photo.jpg --model ViT-L-14
```

### 4. **filter.py** — Wall classification (CLIP zero-shot)

```bash
# Filter indexed images to keep only walls
python filter.py

# Show filter stats
python filter.py --stats

# Force rebuild filters
python filter.py --rebuild

# Change threshold (default: 0.05)
python filter.py --threshold 0.1
```

### 5. **update.py** — Auto-update pipeline (cron-friendly)

```bash
# Update all sources
python update.py

# Update just official gyms
python update.py --official-only

# Update just contributors
python update.py --contributors-only

# Dry-run (show what would happen)
python update.py --dry-run

# Custom delay between requests
python update.py --delay 5.0
```

### 6. **discover.py** — Find new climbing accounts (BFS discovery)

```bash
# Discover from official gym accounts (depth=2, score >= 0.3)
python discover.py

# Shallow search (faster)
python discover.py --depth 1

# Stricter relevance filter
python discover.py --min-score 0.5

# Show suggestions without saving
python discover.py --dry-run

# Auto-add top suggestions to contributors.json
python discover.py --add-suggested

# Custom delay
python discover.py --delay 3.0
```

---

## Node/Vite Commands

```bash
# Start dev server (http://localhost:5173)
npm run dev

# Build production bundle
npm run build

# Preview production build
npm run preview
```

---

## API Server (FastAPI)

### Start
```bash
uvicorn api:app --reload --port 8000
```

### Endpoints

**Search**
```bash
# Upload image and search
curl -X POST http://localhost:8000/api/search \
  -F "file=@wall_photo.jpg" \
  -F "gym=alpine" \
  -F "topK=5" \
  -F "model=ViT-B-32"
```

**Statistics**
```bash
# Get index stats
curl http://localhost:8000/api/stats
```

**Image Serving**
```bash
# Get thumbnail (encoded path)
curl http://localhost:8000/api/thumb/{encoded_path}
```

---

## Environment Variables

```bash
# Instagram credentials (for scraping)
export INSTALOADER_USER="username"
export INSTALOADER_PASS="password"

# Optional: Custom config path
export BETAFINDER_CONFIG="path/to/config.yaml"

# Optional: Custom data directory
export BETAFINDER_DATA_DIR="/path/to/data"
```

---

## Config Files

### `config/config.yaml` — Main configuration
- `gyms`: Official gym Instagram handles
- `paths`: Data directory structure
- `scraping`: Scraper settings (limits, delays, modes)
- `embedding`: CLIP model configuration
- `search`: Search parameters
- `discovery`: Snowball discovery settings

### `data/contributors.json` — Community beta contributors
- Format: `[{"username": "name", "gyms": ["alpine"], "active": true, "note": "..."}]`

### `data/gym_index.json` — Image metadata index
- Format: `[{"filename": "...", "gym": "alpine", "url": "...", "caption": "..."}]`

### `data/feedback.json` — User feedback for fine-tuning
- Format: `[{"session_id": "...", "feedback": "positive|negative", ...}]`

---

## Common Workflows

### 1. **Initial Setup**
```bash
# Scrape all sources
python scrape.py

# Build embeddings
python embed.py

# Filter walls
python filter.py

# Test search
python search.py test.jpg --top 5
```

### 2. **Daily Update**
```bash
# Quick update (community-tagged posts only)
python scrape.py --mode tagged --limit 50

# Rebuild index
python embed.py
```

### 3. **Development (Full Stack)**
```bash
# Terminal 1: API
uvicorn api:app --reload --port 8000

# Terminal 2: Frontend
npm run dev

# Terminal 3: Monitoring (optional)
watch -n 5 'ls -lh data/*.index data/*.json'
```

### 4. **Discover New Creators**
```bash
# Find mentioned accounts
python discover.py --depth 2 --min-score 0.3

# Review suggestions
cat data/graph.json

# Add top suggestions
python discover.py --add-suggested

# Re-scrape with new contributors
python scrape.py
```

---

## Troubleshooting

### HEIC/iPhone photos not working in API
```bash
# Make sure pillow-heif is installed
pip install pillow-heif
```

### 401 Unauthorized errors
```bash
# Set Instagram credentials
export INSTALOADER_USER="your_username"
export INSTALOADER_PASS="your_password"

# Re-run scraper
python scrape.py
```

### FAISS dimension mismatch
```bash
# Rebuild embeddings with correct model
python embed.py --rebuild
```

### Port already in use
```bash
# Change port
uvicorn api:app --port 8001

# Kill process on port
lsof -ti :8000 | xargs kill -9
```

---

## Tips & Tricks

- **Batch scraping:** Use `--limit 200 --delay 5.0` for large batches
- **Testing:** Use `--dry-run` flags to preview changes before committing
- **Monitoring:** Watch `data/` for index updates: `watch -n 2 'ls -lh data/'`
- **Debugging:** Add `--verbose` flag (if supported) or check logs
- **Performance:** Use `ViT-B-32` for speed, `ViT-L-14` for accuracy

---

## Full Example: Start to Search

```bash
# 1. Setup credentials
export INSTALOADER_USER="your_ig_user"
export INSTALOADER_PASS="your_ig_pass"

# 2. Scrape images
python scrape.py

# 3. Build embeddings
python embed.py

# 4. Filter walls
python filter.py

# 5. Start API (Terminal 1)
uvicorn api:app --port 8000 &

# 6. Start frontend (Terminal 2)
npm run dev &

# 7. Open browser
open http://localhost:5173

# 8. Upload a wall photo and search!
```

---

Last updated: 2026-03-09
