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
import logging
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INDEX_FILE  = Path("data/gym_index.json")
EMBED_FILE  = Path("data/embeddings.pkl")
FAISS_FILE  = Path("data/faiss.index")
PATHS_FILE  = Path("data/faiss.paths.json")


def load_index():
    """โหลด FAISS index + metadata"""
    import faiss

    if not FAISS_FILE.exists():
        raise FileNotFoundError(f"❌ Run embed.py first to build the index.")

    index      = faiss.read_index(str(FAISS_FILE))
    path_list  = json.loads(PATHS_FILE.read_text())
    metadata   = json.loads(INDEX_FILE.read_text())

    # สร้าง lookup: filename → metadata
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
           model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """
    Main search function
    Returns: list of dicts ที่มี filename, gym, url, score, caption
    """
    import open_clip, torch

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

    # Search
    scores, indices = index.search(query_embed, top_k * 3)  # fetch เยอะขึ้น เผื่อ filter

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(path_list):
            continue

        filename = path_list[idx]
        meta     = meta_by_file.get(filename, {})

        # filter by gym ถ้าระบุ
        if gym_filter and meta.get("gym") != gym_filter:
            continue

        results.append({
            "rank":     len(results) + 1,
            "score":    float(score),
            "filename": filename,
            "gym":      meta.get("gym") or "?",
            "url":      meta.get("url", ""),
            "caption":  (meta.get("caption", "")[:100] + "...") if meta.get("caption") else "",
            "date":     meta.get("date", "")[:10],
        })

        if len(results) >= top_k:
            break

    return results


def print_results(results: list[dict], open_browser: bool = False):
    """แสดงผล + เปิดเบราว์เซอร์ถ้าต้องการ"""
    print(f"\n{'='*60}")
    print(f"  🧗 BetaScan CNX — Top {len(results)} Results")
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
    parser = argparse.ArgumentParser(description="BetaScan CNX - Search")
    parser.add_argument("image",       type=Path, help="รูปผนังที่ต้องการค้นหา")
    parser.add_argument("--top",       type=int,  default=5,    help="จำนวน results")
    parser.add_argument("--gym",       choices=["alpine", "mainwall", "progression"],
                        default=None,  help="filter เฉพาะยิมนี้")
    parser.add_argument("--open",      action="store_true",     help="เปิดเบราว์เซอร์")
    parser.add_argument("--model",     default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
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