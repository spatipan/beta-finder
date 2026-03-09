"""
search.py - รับรูปผนัง → return top-K รูปที่คล้ายกันจาก Instagram

Usage:
    python search.py photo.jpg              # top 5 results
    python search.py photo.jpg --top 10     # top 10
    python search.py photo.jpg --gym alpine # เฉพาะ gym นั้น
    python search.py photo.jpg --open       # เปิดเบราว์เซอร์ไปยัง IG post
"""

import json
import pickle
import argparse
import webbrowser
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

from src.config import load_config, get_path, get_gym_names, get_nested
from src.index import load_posts_index, load_frames_index
from src.logger import setup_logger

log = setup_logger(__name__)


def load_index():
    """โหลด FAISS index + metadata (supports new two-level schema)"""
    import faiss

    index_file = get_path("index_file")
    faiss_file = get_path("faiss_file")
    paths_file = faiss_file.with_suffix(".paths.json")

    if not faiss_file.exists():
        raise FileNotFoundError(f"❌ Run embed.py first to build the index.")

    index     = faiss.read_index(str(faiss_file))
    path_list = json.loads(paths_file.read_text())

    # Try new two-level schema first
    posts  = load_posts_index()
    frames = load_frames_index()

    if posts and frames:
        # Build lookup: filename → merged post+frame metadata
        posts_by_shortcode = {p["shortcode"]: p for p in posts}
        meta_by_file = {}
        for frame in frames:
            shortcode = frame.get("shortcode")
            post = posts_by_shortcode.get(shortcode, {})
            merged = {**post, **frame}   # frame fields take precedence
            fname = frame["filename"]
            meta_by_file[fname] = merged
            meta_by_file[fname + ".jpg"] = merged
    else:
        # Fallback: flat gym_index
        metadata = json.loads(index_file.read_text())
        meta_by_file = {}
        for m in metadata:
            meta_by_file[m["filename"]] = m
            meta_by_file[m["filename"] + ".jpg"] = m

    return index, path_list, meta_by_file


def embed_query(image_path: Path, model, preprocess, device) -> np.ndarray:
    """Embed รูป query 1 รูป"""
    import torch

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy().astype("float32")


def search(query_path: Path, top_k: int = 5, gym_filter: str | None = None,
           model_name: str = None, pretrained: str = None):
    """
    Main search function
    Returns: list of dicts ที่มี filename, gym, url, score, caption
    """
    import open_clip, torch

    # Use config defaults if not provided
    if model_name is None:
        model_name = get_nested("search.default_model")
    if pretrained is None:
        pretrained = get_nested("search.default_pretrained")

    # โหลด model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval().to(device)

    # โหลด index
    index, path_list, meta_by_file = load_index()

    # Embed query
    log.info(f"🔍 Searching for: {query_path}")
    query_embed = embed_query(query_path, model, preprocess, device)

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

        # Filter by gym — support both old (gym: str) and new (gyms: list) schema
        gyms_list = meta.get("gyms") or ([meta["gym"]] if meta.get("gym") else [])
        if gym_filter and gym_filter not in gyms_list:
            continue

        caption_text = meta.get("caption", "")
        if caption_text:
            caption_text = caption_text[:caption_max_length] + "..."

        # Primary gym for display (first in list, or fallback)
        primary_gym = gyms_list[0] if gyms_list else (meta.get("gym") or "?")

        results.append({
            "rank":     len(results) + 1,
            "score":    float(score),
            "filename": filename,
            "gym":      primary_gym,
            "gyms":     gyms_list,
            "url":      meta.get("url", ""),
            "username": meta.get("username", ""),
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
    parser.add_argument("--model",     default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--json",      action="store_true",     help="output JSON")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"❌ Image not found: {args.image}")
        return

    results = search(
        args.image,
        top_k=args.top,
        gym_filter=args.gym,
        model_name=args.model,
        pretrained=args.pretrained,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_results(results, open_browser=args.open)


if __name__ == "__main__":
    main()