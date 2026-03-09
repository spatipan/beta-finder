"""
embed.py - สร้าง embeddings (CLIP, SigLIP, EVA-CLIP, DINOv2, หรือ SIFT) สำหรับรูปทั้งหมด
และ build vector index สำหรับ similarity search

Usage:
    python embed.py                                              # default: CLIP ViT-B-32
    python embed.py --model ViT-L-14 --pretrained openai       # CLIP ViT-L-14
    python embed.py --model ViT-B-16-SigLIP --pretrained webli # SigLIP ViT-B-16
    python embed.py --model ViT-SO400M-14-SigLIP --pretrained webli  # SigLIP ViT-SO400M-14 (largest)
    python embed.py --model EVA02-E-14 --pretrained laion2b_s4b_b115k  # EVA-CLIP E-14
    python embed.py --backbone dinov2_vitb14                   # DINOv2 ViT-B14
    python embed.py --backbone sift                             # SIFT (no GPU needed)
    python embed.py --batch 32                                 # ปรับ batch size
"""

import json
import pickle
import argparse
from pathlib import Path

import numpy as np

from src.config import load_config, get_path, get_nested
from src.logger import setup_logger
from src.embeddings import load_model, embed_batch

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray):
    """
    Build FAISS flat L2 index
    (สำหรับ dataset เล็ก ~1000 รูป FlatL2 เร็วพอ ไม่ต้อง IVF)
    """
    import faiss

    dim = embeddings.shape[1]
    # ใช้ inner product (เหมาะกับ L2-normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_config()  # Load config at startup

    parser = argparse.ArgumentParser(description="BetaFinder CNX - Embedder")
    parser.add_argument("--backbone",   default=None,
                        help="Model backbone: CLIP (default) or dinov2_vitb14/dinov2_vitl14")
    parser.add_argument("--model",     default=None,
                        help="CLIP model name (e.g., ViT-B-32)")
    parser.add_argument("--pretrained", default=None,
                        help="CLIP pretrained weights (e.g., openai)")
    parser.add_argument("--batch",     type=int, default=None)
    parser.add_argument("--rebuild",   action="store_true",
                        help="Re-embed รูปทั้งหมด แม้มี cache แล้ว")
    args = parser.parse_args()

    # Get paths from config
    index_file = get_path("index_file")
    embed_file = get_path("embeddings_file")
    faiss_file = get_path("faiss_file")

    # โหลด metadata index
    if not index_file.exists():
        log.error(f"❌ {index_file} not found. Run scrape.py first.")
        return

    with open(index_file) as f:
        metadata = json.load(f)

    log.info(f"📂 {len(metadata)} entries in index")

    # Filter to only include entries marked as walls (is_wall: true)
    walls_only = [m for m in metadata if m.get("is_wall", False)]
    log.info(f"   Filtered to {len(walls_only)} wall entries (is_wall: true)")
    metadata = walls_only

    # โหลด embeddings cache
    embed_cache: dict[str, np.ndarray] = {}
    if embed_file.exists() and not args.rebuild:
        with open(embed_file, "rb") as f:
            embed_cache = pickle.load(f)
        log.info(f"   Loaded {len(embed_cache)} cached embeddings")

    # หารูปที่ยังไม่ได้ embed โดยเผื่อกรณีไฟล์มีนามสกุล .jpg ซ้ำซ้อน
    all_paths = []
    for m in metadata:
        p = Path(m["filename"])
        if p.exists():
            all_paths.append(p)
        elif Path(str(p) + ".jpg").exists():
            all_paths.append(Path(str(p) + ".jpg"))

    pending      = [p for p in all_paths if str(p) not in embed_cache]
    log.info(f"   Pending: {len(pending)} images to embed")

    if pending:
        model, preprocess, device = load_model(args.backbone, args.model, args.pretrained)
        # Determine model type
        if args.backbone == "sift" or args.model == "sift":
            model_type = "sift"
        elif args.backbone and args.backbone.startswith("dinov2"):
            model_type = "dinov2"
        else:
            model_type = "clip"
        new_embeds = embed_batch(pending, model, preprocess, device, model_type=model_type, batch_size=args.batch)

        # Store model metadata for search.py to use matching model
        model_metadata = {
            "model_type": model_type,
            "backbone": args.backbone,
            "model_name": args.model or get_nested("embedding.model_name"),
            "pretrained": args.pretrained or get_nested("embedding.pretrained"),
        }

        for path, emb in zip(pending, new_embeds):
            embed_cache[str(path)] = emb

        # save cache
        embed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(embed_file, "wb") as f:
            pickle.dump(embed_cache, f)
        log.info(f"✅ Saved embeddings → {embed_file}")

    # Build FAISS index จาก cache
    valid_paths  = [p for p in all_paths if str(p) in embed_cache]
    embeddings   = np.stack([embed_cache[str(p)] for p in valid_paths])
    path_list    = [str(p) for p in valid_paths]  # ordered list สำหรับ lookup

    import faiss
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(faiss_file))

    # save path_list (เพื่อ map index → filename)
    with open(faiss_file.with_suffix(".paths.json"), "w") as f:
        json.dump(path_list, f)

    # save model metadata (สำหรับ search.py ใช้ model เดียวกัน)
    if pending:
        with open(faiss_file.with_suffix(".model.json"), "w") as f:
            json.dump(model_metadata, f)
        log.info(f"✅ Model metadata saved → {faiss_file.with_suffix('.model.json')}")

    log.info(f"✅ FAISS index saved → {faiss_file}")
    log.info(f"   Total indexed: {len(path_list)} images")


if __name__ == "__main__":
    main()