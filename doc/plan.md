Build "Beta Finder CNX" — an AI-powered climbing beta finder for Chiang Mai gyms.

## What it does

Climber photographs a wall → app returns ranked Instagram posts (photos/Reels) 

showing beta for that route. Like reverse image search but for climbing walls.

## Gyms

- Alpine Outpost (@the_alpine_outpost)

- Main Wall CNX (@mainwallcnx)  

- Progression Vertical (@progressionvertical)

## Core Pipeline

Instagram (tagged posts + official feed)

  → scrape.py        # instaloader, tagged feed priority, video keyframe extraction (OpenCV)

  → filter.py        # CLIP zero-shot wall classifier (contrastive scoring, threshold 0.05)

  → embed.py         # CLIP ViT-B-32 → FAISS flat cosine index

  → search.py/app.py # query image → top-K results with Instagram links

## Key Design Decisions

- Source priority: tagged posts of gym > official feed > manual contributor list

  (climbers tag gyms intentionally so others find beta — this is the main data source)

- Videos/Reels: download → extract 4 keyframes evenly (skip first/last 5%) → delete video

- Dedup by shortcode (not filename) across all sources

- Instagram auth required: session via `instaloader --login USERNAME`

- Wall filter runs before embed to exclude non-wall images from FAISS index

## Data Schema (gym_index.json entries)

source_type: tagged|official|contributor

gym, username, shortcode, url, caption, date, likes

media_type: image|video_thumb|keyframe

is_wall, wall_score, filename

## Streamlit UI — Climbing Gym Aesthetic (Modern Minimalistic)

### Color Palette (Gym Wall & Holds Theme)
- **Background:** Deep charcoal (#1a1a1a, #0f0f0f) — like climbing gym wall
- **Primary holds:** Warm accents (#ff6b35, #ffa500, #ffd700) — typical gym hold colors
- **Secondary holds:** Cool accents (#00d4ff, #00ff88) — volume tape markers
- **Stripes:** Subtle (#2a2a2a) — like volume tape on wall
- **Text:** Clean (#ffffff, #e0e0e0, #a0a0a0)

### Layout
- **Header:** Logo 🧗 + "BetaFinder CNX" title + Thai tagline
- **Gym filter:** Checkbox pills (Alpine=🟢#4CAF50, MainWall=🔵#2196F3, Progression=🟠#FF9800, All=🌐#FFD700)
- **Upload section:** Drag-drop image input with preview + metadata display
- **Search button:** Gradient (#ff6b35→#ff8c42) with hover animation
- **Results:**
  - Score bars (segmented colors: 🔥high, ⭐medium, ●low)
  - Result cards with left border (#ff6b35) and gradient background
  - Gym emoji badges, progress bars, caption, date, username
  - IG link (clickable)

### Sidebar
- **Settings:** Model selector (ViT-B-32 fast vs ViT-L-14 accurate), top-k slider
- **Statistics:** Total images, images per gym (cards with metrics)
- **Tips:** Photography best practices
- **About:** Project description + GitHub link

### Interactive Elements
- Buttons: gradient orange with shadow, lift on hover
- Input fields: dark background (#2a2a2a), warm border on focus
- Progress bars: gradient (#ff6b35→#ffd700)
- Score display: color-coded (green≥0.85, gold≥0.75, orange<0.75)
- Volume badges: cyan, green, orange, gold with transparency + border

## Infrastructure

- Host: Minisforum UM890 Pro (Ryzen 9 8945HS, 32GB RAM)

- GPU: RTX 3060 12GB via eGPU dock + PCIe (for CLIP embed + future fine-tuning)

- Deploy: Streamlit app behind Cloudflare tunnel (consistent with other self-hosted services)

- Nightly cron: update.py --contributors-only --ig-user USERNAME

## Planned Features (don't build yet, just be aware)

- Phase 2: user feedback loop → CLIP fine-tuning on climbing pairs, snowball account discovery

- Phase 2: frame occlusion scoring (pick clearest wall frame from Reel)

- Phase 3: hold detection (color clustering → SAM), route fingerprinting, ORB pre-filter

## File Structure

**Core Scripts:**
- scrape.py (Instagram scraper — done, needs enhancement)
- filter.py (Wall classifier — TODO)
- embed.py (CLIP embeddings — done)
- search.py (CLI search — done)
- update.py (Auto-scraper for cron — TODO)
- app.py (Streamlit UI — TODO, design finalized)

**Data Files:**
- data/gym_index.json (metadata for all images)
- data/contributors.json (contributor list)
- data/faiss.index (FAISS binary index)
- data/faiss.index.paths.json (ordered path list)
- data/embeddings.pkl (CLIP vector cache)

**Config:**
- .streamlit/config.toml (Streamlit theme config)
- requirements.txt (Python dependencies)

## Stack

Python, instaloader, open-clip-torch, faiss-cpu, opencv-python, streamlit, networkx, tqdm

## Credits

Inspired by original BetaScan by @thangman22 (ResNet50 + ORB hybrid)

This project: @patipan_poty / github.com/spatipan/beta-finder