### Revised Workflow

INDEXING
────────
Scrape Reels (IG) -> with instaloader
      ↓
Extract Frames (ทุก 0.5s) -> with ffmpeg
      ↓
Filter: Is Climbing Wall? -> with clip ViT-B/16
      ↓
Score & Select Top 4 Frames -> with laplacian (sharpness), wall ratio, no person ratio
      ↓
Embed ด้วย DINOv2 -> with dinov2-base
      ↓
Average → 1 Vector per Reel -> with average pooling
      ↓
Store + Metadata (gym, date, url)
      ↓
VectorDB


SEARCH
──────
Upload Image
      ↓
Filter: Is Climbing Wall? -> with clip ViT-B/16 
      ↓
Embed ด้วย DINOv2 -> with dinov2-base
      ↓
Cosine Similarity → Top 20
      ↓
Re-rank by (gym filter, recency)
      ↓
Return Top 5