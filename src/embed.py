"""
embed.py - สร้าง CLIP embeddings สำหรับรูปทั้งหมด
และ build vector index สำหรับ similarity search

Usage:
    python embed.py              # embed รูปทั้งหมดใน data/images/
    python embed.py --batch 32   # ปรับ batch size
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
# Model loader - ใช้ open_clip (รองรับหลาย backbone)
# ---------------------------------------------------------------------------

def load_model(model_name: str = None, pretrained: str = None):
    """
    โหลด CLIP model
    แนะนำ:
      - "ViT-B-32" + "openai"   → เร็ว, RAM น้อย (~350MB)
      - "ViT-L-14" + "openai"   → แม่นขึ้น (~900MB)
    """
    import open_clip
    import torch

    # Use config defaults if not provided
    if model_name is None:
        model_name = get_nested("embedding.model_name")
    if pretrained is None:
        pretrained = get_nested("embedding.pretrained")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {model_name} ({pretrained}) on {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval().to(device)
    return model, preprocess, device


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_images(image_paths: list[Path], model, preprocess, device,
                 batch_size: int = None) -> np.ndarray:
    """
    รับ list ของ image paths → return numpy array shape (N, embed_dim)
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
                tensors.append(preprocess(Image.new("RGB", (224, 224))))  # blank fallback

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)
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
    parser.add_argument("--model",     default=None)
    parser.add_argument("--pretrained", default=None)
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
        model, preprocess, device = load_model(args.model, args.pretrained)
        new_embeds = embed_images(pending, model, preprocess, device, batch_size=args.batch)

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

    log.info(f"✅ FAISS index saved → {faiss_file}")
    log.info(f"   Total indexed: {len(path_list)} images")


if __name__ == "__main__":
    main()