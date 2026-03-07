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
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR     = Path("data/images")
INDEX_FILE   = Path("data/gym_index.json")
EMBED_FILE   = Path("data/embeddings.pkl")   # {filename: np.array}
FAISS_FILE   = Path("data/faiss.index")


# ---------------------------------------------------------------------------
# Model loader - ใช้ open_clip (รองรับหลาย backbone)
# ---------------------------------------------------------------------------

def load_model(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """
    โหลด CLIP model
    แนะนำ:
      - "ViT-B-32" + "openai"   → เร็ว, RAM น้อย (~350MB)
      - "ViT-L-14" + "openai"   → แม่นขึ้น (~900MB)
    """
    import open_clip
    import torch

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
                 batch_size: int = 16) -> np.ndarray:
    """
    รับ list ของ image paths → return numpy array shape (N, embed_dim)
    """
    import torch

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
    parser = argparse.ArgumentParser(description="BetaScan CNX - Embedder")
    parser.add_argument("--model",     default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--rebuild",   action="store_true",
                        help="Re-embed รูปทั้งหมด แม้มี cache แล้ว")
    args = parser.parse_args()

    # โหลด metadata index
    if not INDEX_FILE.exists():
        log.error(f"❌ {INDEX_FILE} not found. Run scrape.py first.")
        return

    with open(INDEX_FILE) as f:
        metadata = json.load(f)

    log.info(f"📂 {len(metadata)} entries in index")

    # โหลด embeddings cache
    embed_cache: dict[str, np.ndarray] = {}
    if EMBED_FILE.exists() and not args.rebuild:
        with open(EMBED_FILE, "rb") as f:
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
        EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBED_FILE, "wb") as f:
            pickle.dump(embed_cache, f)
        log.info(f"✅ Saved embeddings → {EMBED_FILE}")

    # Build FAISS index จาก cache
    valid_paths  = [p for p in all_paths if str(p) in embed_cache]
    embeddings   = np.stack([embed_cache[str(p)] for p in valid_paths])
    path_list    = [str(p) for p in valid_paths]  # ordered list สำหรับ lookup

    import faiss
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(FAISS_FILE))

    # save path_list (เพื่อ map index → filename)
    with open(FAISS_FILE.with_suffix(".paths.json"), "w") as f:
        json.dump(path_list, f)

    log.info(f"✅ FAISS index saved → {FAISS_FILE}")
    log.info(f"   Total indexed: {len(path_list)} images")


if __name__ == "__main__":
    main()