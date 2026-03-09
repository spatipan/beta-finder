"""
embed.py - สร้าง embeddings (CLIP หรือ DINOv2) สำหรับรูปทั้งหมด
และ build vector index สำหรับ similarity search

Usage:
    python embed.py              # embed รูปทั้งหมดใน data/images/ (default: CLIP)
    python embed.py --batch 32   # ปรับ batch size
    python embed.py --backbone dinov2_vitb14  # ใช้ DINOv2
"""

import json
import pickle
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import load_config, get_path, get_nested
from src.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Model loader - รองรับ CLIP และ DINOv2
# ---------------------------------------------------------------------------

def load_model(backbone: str = None, model_name: str = None, pretrained: str = None):
    """
    โหลด embedding model (CLIP หรือ DINOv2)

    CLIP แนะนำ:
      - "ViT-B-32" + "openai"   → เร็ว, RAM น้อย (~350MB)
      - "ViT-L-14" + "openai"   → แม่นขึ้น (~900MB)

    DINOv2 แนะนำ:
      - "dinov2_vitb14"  → 768-dim, เร็ว, ดี
      - "dinov2_vitl14"  → 1024-dim, แม่นกว่า
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine if using DINOv2 or CLIP
    if backbone and backbone.startswith("dinov2"):
        return load_dinov2_model(backbone, device)
    else:
        return load_clip_model(model_name, pretrained, device)


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


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_images(image_paths: list[Path], model, preprocess, device, model_type: str = "clip",
                 batch_size: int = None) -> np.ndarray:
    """
    รับ list ของ image paths → return numpy array shape (N, embed_dim)
    รองรับ CLIP และ DINOv2
    """
    import torch

    # Use config default if not provided
    if batch_size is None:
        batch_size = get_nested("embedding.default_batch_size")

    all_embeds = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_paths = image_paths[i: i + batch_size]
        tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
            except Exception as e:
                log.warning(f"Skip {p}: {e}")
                # blank fallback - preprocess expects PIL Image or tensor depending on model
                blank = Image.new("RGB", (224, 224))
                tensors.append(preprocess(blank))

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            if model_type == "clip":
                features = model.encode_image(batch_tensor)
            else:  # dinov2
                features = model(batch_tensor)

            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

        all_embeds.append(features.cpu().numpy())

    return np.vstack(all_embeds).astype("float32")


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
        model_type = "dinov2" if args.backbone and args.backbone.startswith("dinov2") else "clip"
        new_embeds = embed_images(pending, model, preprocess, device, model_type=model_type, batch_size=args.batch)

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