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
from src.logger import setup_logger

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


def load_model(backbone: str = None, model_name: str = None, pretrained: str = None, device: str = None):
    """โหลด embedding model (CLIP หรือ DINOv2) - ใช้ร่วมกับ embed.py"""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine if using DINOv2 or CLIP
    if backbone and backbone.startswith("dinov2"):
        return load_dinov2_model(backbone, device), "dinov2"
    else:
        return load_clip_model(model_name, pretrained, device), "clip"


def load_clip_model(model_name: str = None, pretrained: str = None, device: str = None):
    """โหลด CLIP model via open_clip"""
    import open_clip
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use config defaults if not provided
    if model_name is None:
        model_name = get_nested("embedding.model_name")
    if pretrained is None:
        pretrained = get_nested("embedding.pretrained")

    log.info(f"Loading CLIP {model_name} ({pretrained}) on {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval().to(device)
    return model, preprocess, device


def load_dinov2_model(backbone: str, device: str = None):
    """โหลด DINOv2 model"""
    import torch
    import torchvision.transforms as transforms

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading DINOv2 {backbone} on {device}")

    # Load DINOv2 model
    model = torch.hub.load("facebookresearch/dinov2", backbone)
    model.eval().to(device)

    # DINOv2 preprocess: normalize to ImageNet stats
    preprocess = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return model, preprocess, device


def embed_query(image_path: Path, model, preprocess, device, model_type: str = "clip") -> np.ndarray:
    """Embed รูป query 1 รูป (รองรับ CLIP และ DINOv2)"""
    import torch

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "clip":
            feat = model.encode_image(tensor)
        else:  # dinov2
            feat = model(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy().astype("float32")


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

    # โหลด model (ตรงกับ index)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_backbone and use_backbone.startswith("dinov2"):
        model, preprocess, device = load_dinov2_model(use_backbone, device)
        model_type = "dinov2"
    else:
        model, preprocess, device = load_clip_model(use_model_name, use_pretrained, device)
        model_type = "clip"

    log.info(f"Using {model_type.upper()} model for embedding")

    # Embed query
    log.info(f"🔍 Searching for: {query_path}")
    query_embed = embed_query(query_path, model, preprocess, device, model_type=model_type)

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