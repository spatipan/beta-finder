"""
search.py - รับรูปผนัง → return top-K รูปที่คล้ายกันจาก Instagram

Usage:
    python search.py photo.jpg                                  # auto-detect model from index
    python search.py photo.jpg --top 10                         # top 10 results
    python search.py photo.jpg --gym alpine                     # filter by gym
    python search.py photo.jpg --open                           # open results in browser
    python search.py photo.jpg --backbone dinov2_vitl14         # override with DINOv2
    python search.py photo.jpg --model EVA02-E-14 --pretrained laion2b_s4b_b115k  # override with EVA-CLIP
"""

import json
import pickle
import argparse
import webbrowser
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from src.config import load_config, get_path, get_gym_names, get_nested
from src.logger import setup_logger
from src.embeddings import load_model, embed_single

log = setup_logger(__name__)


def load_index():
    """โหลด FAISS index + metadata + model info"""
    import faiss

    index_file = get_path("index_file")
    faiss_file = get_path("faiss_file")
    paths_file = faiss_file.with_suffix(".paths.json")
    model_file = faiss_file.with_suffix(".model.json")

    if not faiss_file.exists():
        raise FileNotFoundError(f"❌ Run embed.py first to build the index.")

    index      = faiss.read_index(str(faiss_file))
    path_list  = json.loads(paths_file.read_text())
    metadata   = json.loads(index_file.read_text())

    # โหลด model metadata (ใช้ default ถ้าไม่มี)
    model_metadata = {}
    if model_file.exists():
        model_metadata = json.loads(model_file.read_text())

    # สร้าง lookup: filename → metadata
    meta_by_file = {}
    for m in metadata:
        meta_by_file[m["filename"]] = m
        meta_by_file[m["filename"] + ".jpg"] = m

    return index, path_list, meta_by_file, model_metadata


def search(query_path: Path, top_k: int = 5, gym_filter: str | None = None,
           model_name: str = None, pretrained: str = None, backbone: str = None):
    """
    Main search function
    Returns: list of dicts ที่มี filename, gym, url, score, caption

    Automatically uses the same model as the index, unless overridden by arguments
    """
    import torch

    # โหลด index และ model metadata
    index, path_list, meta_by_file, model_metadata = load_index()

    # Determine which model to use: prefer stored metadata, fallback to args/config
    use_backbone = backbone or model_metadata.get("backbone")
    use_model_name = model_name or model_metadata.get("model_name") or get_nested("search.default_model")
    use_pretrained = pretrained or model_metadata.get("pretrained") or get_nested("search.default_pretrained")
    model_type = model_metadata.get("model_type", "clip")

    # โหลด model (ตรงกับ index) using shared embedding module
    model, preprocess, device = load_model(use_backbone, use_model_name, use_pretrained)
    model_type = "dinov2" if use_backbone and use_backbone.startswith("dinov2") else "clip"

    log.info(f"Using {model_type.upper()} model for embedding")

    # Embed query using shared embedding module
    log.info(f"🔍 Searching for: {query_path}")
    query_embed = embed_single(query_path, model, preprocess, device, model_type=model_type)

    # Search with oversample factor from config
    oversample_factor = get_nested("search.oversample_factor")
    scores, indices = index.search(query_embed, top_k * oversample_factor)

    # Get caption max length from config
    caption_max_length = get_nested("search.caption_max_length")

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(path_list):
            continue

        filename = path_list[idx]
        meta     = meta_by_file.get(filename, {})

        # filter by gym ถ้าระบุ
        if gym_filter and meta.get("gym") != gym_filter:
            continue

        caption_text = meta.get("caption", "")
        if caption_text:
            caption_text = caption_text[:caption_max_length] + "..."

        results.append({
            "rank":     len(results) + 1,
            "score":    float(score),
            "filename": filename,
            "gym":      meta.get("gym") or "?",
            "url":      meta.get("url", ""),
            "caption":  caption_text,
            "date":     meta.get("date", "")[:10],
        })

        if len(results) >= top_k:
            break

    return results


def print_results(results: list[dict], open_browser: bool = False):
    """แสดงผล + เปิดเบราว์เซอร์ถ้าต้องการ"""
    print(f"\n{'='*60}")
    print(f"  🧗 BetaFinder CNX — Top {len(results)} Results")
    print(f"{'='*60}")

    for r in results:
        print(f"\n  #{r['rank']}  [{r['gym'].upper()}]  score={r['score']:.4f}")
        print(f"      📅 {r['date']}")
        print(f"      💬 {r['caption'] or '(no caption)'}")
        print(f"      🔗 {r['url']}")

        if open_browser and r["url"]:
            webbrowser.open(r["url"])

    print(f"\n{'='*60}\n")


def main():
    load_config()  # Load config at startup

    # Get gym choices from config
    gym_choices = get_gym_names()

    parser = argparse.ArgumentParser(description="BetaFinder CNX - Search")
    parser.add_argument("image",       type=Path, help="รูปผนังที่ต้องการค้นหา")
    parser.add_argument("--top",       type=int,  default=5,    help="จำนวน results")
    parser.add_argument("--gym",       choices=gym_choices,
                        default=None,  help="filter เฉพาะยิมนี้")
    parser.add_argument("--open",      action="store_true",     help="เปิดเบราว์เซอร์")
    parser.add_argument("--backbone",  default=None,
                        help="Model backbone (auto-detected from index, or override with dinov2_vitb14/dinov2_vitl14)")
    parser.add_argument("--model",     default=None,
                        help="CLIP model name (auto-detected from index, or override)")
    parser.add_argument("--pretrained", default=None,
                        help="CLIP pretrained weights (auto-detected from index, or override)")
    parser.add_argument("--json",      action="store_true",     help="output JSON")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"❌ Image not found: {args.image}")
        return

    results = search(
        args.image,
        top_k=args.top,
        gym_filter=args.gym,
        backbone=args.backbone,
        model_name=args.model,
        pretrained=args.pretrained,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_results(results, open_browser=args.open)


if __name__ == "__main__":
    main()