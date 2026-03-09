"""
embeddings.py - Shared embedding functionality for embed.py and search.py

Provides model loading and embedding functions to be used by both:
- embed.py: Generate embeddings for the full image database
- search.py: Generate embeddings for query images
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import get_nested
from src.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Model loader - รองรับ CLIP, SigLIP, EVA-CLIP, DINOv2
# ---------------------------------------------------------------------------

def load_model(backbone: str = None, model_name: str = None, pretrained: str = None):
    """
    โหลด embedding model (CLIP, SigLIP, EVA-CLIP, หรือ DINOv2)

    CLIP-based models (via open_clip):
      CLIP (OpenAI):
        - "ViT-B-32" + "openai"        → เร็ว, RAM น้อย (~350MB)
        - "ViT-L-14" + "openai"        → แม่นขึ้น (~900MB)
      SigLIP (Google, better text-image matching):
        - "ViT-B-16-SigLIP" + "webli"           → เร็ว
        - "ViT-SO400M-14-SigLIP" + "webli"      → ใหญ่, แม่นที่สุด
      EVA-CLIP (Baidu, strongest vision):
        - "EVA02-E-14" + "laion2b_s4b_b115k"    → ประสิทธิภาพสูง, รายละเอียดมาก

    DINOv2 (Meta, self-supervised):
      - "dinov2_vitb14"  → 768-dim, เร็ว, ดี
      - "dinov2_vitl14"  → 1024-dim, แม่นกว่า
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine if using DINOv2 or CLIP-based
    if backbone and backbone.startswith("dinov2"):
        return load_dinov2_model(backbone, device)
    else:
        return load_clip_model(model_name, pretrained, device)


def load_clip_model(model_name: str = None, pretrained: str = None, device: str = None):
    """
    โหลด CLIP-based model via open_clip

    Supported models:
      CLIP (OpenAI):
        - "ViT-B-32" + "openai"   → เร็ว, RAM น้อย (~350MB)
        - "ViT-L-14" + "openai"   → แม่นขึ้น (~900MB)
      SigLIP (Google):
        - "ViT-B-16-SigLIP" + "webli"      → เร็ว, สมดุล
        - "ViT-SO400M-14-SigLIP" + "webli" → ใหญ่, แม่นที่สุด
      EVA-CLIP (Baidu):
        - "EVA02-E-14" + "laion2b_s4b_b115k" → ประสิทธิภาพสูง, รายละเอียดมาก
    """
    import open_clip
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use config defaults if not provided
    if model_name is None:
        model_name = get_nested("embedding.model_name")
    if pretrained is None:
        pretrained = get_nested("embedding.pretrained")

    log.info(f"Loading {model_name} ({pretrained}) on {device}")

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
# Embedding functions - shared between embed.py and search.py
# ---------------------------------------------------------------------------

def embed_batch(image_paths: list[Path], model, preprocess, device, model_type: str = "clip",
                batch_size: int = None) -> np.ndarray:
    """
    Embed a batch of images.

    Args:
        image_paths: List of Path objects pointing to images
        model: Loaded model (CLIP or DINOv2)
        preprocess: Preprocessing function from model loader
        device: torch device (cuda or cpu)
        model_type: "clip" or "dinov2"
        batch_size: Batch size for embedding (uses config default if None)

    Returns:
        numpy array of embeddings, shape (N, embed_dim), dtype float32
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
                # blank fallback - preprocess expects PIL Image
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


def embed_single(image_path: Path, model, preprocess, device, model_type: str = "clip") -> np.ndarray:
    """
    Embed a single image.

    Args:
        image_path: Path to the image file
        model: Loaded model (CLIP or DINOv2)
        preprocess: Preprocessing function from model loader
        device: torch device (cuda or cpu)
        model_type: "clip" or "dinov2"

    Returns:
        numpy array of shape (1, embed_dim), dtype float32
    """
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
