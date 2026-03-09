"""
embeddings.py - Shared embedding functionality for embed.py and search.py

Provides model loading and embedding functions to be used by both:
- embed.py: Generate embeddings for the full image database
- search.py: Generate embeddings for query images

Supports multiple embedding methods:
- Deep learning semantic: CLIP, SigLIP, EVA-CLIP, DINOv2
- Deep learning local: SuperPoint + SuperGlue (GPU required)
- Traditional: SIFT (Scale-Invariant Feature Transform, CPU only)
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import get_nested
from src.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Model loader - รองรับ CLIP, SigLIP, EVA-CLIP, DINOv2, SIFT
# ---------------------------------------------------------------------------

def load_model(backbone: str = None, model_name: str = None, pretrained: str = None):
    """
    โหลด embedding model (CLIP, SigLIP, EVA-CLIP, DINOv2, SuperPoint, หรือ SIFT)

    Semantic Deep Learning models (via open_clip):
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

    Local Feature Deep Learning (GPU required):
      SuperPoint + SuperGlue:
        - "superpoint"  → GPU-accelerated keypoint detection + matching

    Traditional Computer Vision (CPU only):
      SIFT (Scale-Invariant Feature Transform):
        - "sift"  → ไม่ต้อง GPU, ดีสำหรับ climbing wall features
    """
    import torch

    # Handle SIFT specially (no GPU needed, returns None as device)
    if backbone == "sift" or model_name == "sift":
        return load_sift_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Handle SuperPoint (GPU-required local features)
    if backbone == "superpoint" or model_name == "superpoint":
        if not torch.cuda.is_available():
            log.warning("SuperPoint requires GPU but CUDA not available. Falling back to SIFT.")
            return load_sift_model()
        return load_superpoint_model(device)

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


def load_sift_model():
    """
    โหลด SIFT detector/descriptor (OpenCV)

    Returns:
        tuple: (sift_detector, None, "cpu")
    """
    import cv2

    log.info("Loading SIFT detector (OpenCV)")
    sift = cv2.SIFT_create()
    return sift, None, "cpu"


def load_superpoint_model(device: str = "cuda"):
    """
    โหลด SuperPoint detector + SuperGlue matcher

    Requires:
    - pip install kornia-moons
    - GPU (CUDA)

    Returns:
        tuple: (superpoint_model, None, device)
    """
    import torch

    try:
        from kornia_moons.feature import SuperPoint, SuperGlue

        log.info(f"Loading SuperPoint detector + SuperGlue matcher on {device}")

        # Initialize SuperPoint
        superpoint = SuperPoint(max_num_keypoints=256).to(device).eval()

        # Initialize SuperGlue
        superglue = SuperGlue(pretrained="outdoor").to(device).eval()

        # Return SuperPoint detector (SuperGlue is used for matching during search)
        return superpoint, None, device

    except ImportError:
        log.error("SuperPoint requires: pip install kornia-moons")
        log.warning("Falling back to SIFT instead")
        return load_sift_model()


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
        model: Loaded model (CLIP, DINOv2, SIFT, or SuperPoint)
        preprocess: Preprocessing function from model loader (None for SIFT/SuperPoint)
        device: torch device (cuda, cpu, or "cpu" for SIFT)
        model_type: "clip", "dinov2", "sift", or "superpoint"
        batch_size: Batch size for embedding (uses config default if None)

    Returns:
        numpy array of embeddings, shape (N, embed_dim), dtype float32
    """
    # Handle SIFT separately
    if model_type == "sift":
        return embed_batch_sift(image_paths, model)

    # Handle SuperPoint separately
    if model_type == "superpoint":
        return embed_batch_superpoint(image_paths, model, device)

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


def embed_batch_sift(image_paths: list[Path], sift_detector) -> np.ndarray:
    """
    Embed a batch of images using SIFT keypoints and descriptors.

    Args:
        image_paths: List of Path objects pointing to images
        sift_detector: OpenCV SIFT detector object

    Returns:
        numpy array of SIFT descriptors (variable-length), shape (N, embed_dim)
        Note: For fixed-size embedding, computes mean of all descriptors per image
    """
    import cv2

    all_embeds = []

    for p in tqdm(image_paths, desc="Extracting SIFT"):
        try:
            # Read image
            img_cv = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                log.warning(f"Skip {p}: Cannot read image")
                # Use zero vector as fallback
                all_embeds.append(np.zeros(128, dtype="float32"))
                continue

            # Detect keypoints and compute descriptors
            kp, des = sift_detector.detectAndCompute(img_cv, None)

            if des is None or len(des) == 0:
                log.warning(f"Skip {p}: No SIFT descriptors found")
                # Use zero vector as fallback
                all_embeds.append(np.zeros(128, dtype="float32"))
            else:
                # Aggregate descriptors: use mean of all descriptors
                # This creates a fixed-size embedding for each image
                agg_embed = np.mean(des, axis=0).astype("float32")
                all_embeds.append(agg_embed)

        except Exception as e:
            log.warning(f"Skip {p}: {e}")
            all_embeds.append(np.zeros(128, dtype="float32"))

    return np.vstack([e.reshape(1, -1) for e in all_embeds]).astype("float32")


def embed_batch_superpoint(image_paths: list[Path], superpoint, device: str) -> np.ndarray:
    """
    Embed a batch of images using SuperPoint keypoint detection.

    Args:
        image_paths: List of Path objects pointing to images
        superpoint: Loaded SuperPoint detector model
        device: torch device (cuda)

    Returns:
        numpy array of SuperPoint descriptors, shape (N, 256), dtype float32
    """
    import torch
    import cv2

    all_embeds = []

    for p in tqdm(image_paths, desc="Extracting SuperPoint"):
        try:
            # Read image
            img_cv = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                log.warning(f"Skip {p}: Cannot read image")
                all_embeds.append(np.zeros(256, dtype="float32"))
                continue

            # Resize for consistency
            h, w = img_cv.shape
            scale = 320.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_cv, (new_w, new_h))

            # Convert to tensor
            img_tensor = torch.from_numpy(img_resized).float()[None, None].to(device) / 255.0

            with torch.no_grad():
                # Detect keypoints and descriptors
                lafs, resp, desc = superpoint({"image": img_tensor})

            if desc is None or desc.shape[1] == 0:
                log.warning(f"Skip {p}: No SuperPoint descriptors found")
                all_embeds.append(np.zeros(256, dtype="float32"))
            else:
                # Aggregate descriptors: use mean of all descriptors
                agg_embed = desc.squeeze().mean(dim=0).cpu().numpy().astype("float32")
                if agg_embed.ndim == 0:
                    # Single descriptor case
                    agg_embed = desc.squeeze().cpu().numpy().astype("float32")
                all_embeds.append(agg_embed)

        except Exception as e:
            log.warning(f"Skip {p}: {e}")
            all_embeds.append(np.zeros(256, dtype="float32"))

    return np.vstack([e.reshape(1, -1) for e in all_embeds]).astype("float32")


def embed_single(image_path: Path, model, preprocess, device, model_type: str = "clip") -> np.ndarray:
    """
    Embed a single image.

    Args:
        image_path: Path to the image file
        model: Loaded model (CLIP, DINOv2, SIFT, or SuperPoint)
        preprocess: Preprocessing function from model loader (None for SIFT/SuperPoint)
        device: torch device (cuda, cpu, or "cpu" for SIFT)
        model_type: "clip", "dinov2", "sift", or "superpoint"

    Returns:
        numpy array of shape (1, embed_dim), dtype float32
    """
    # Handle SIFT separately
    if model_type == "sift":
        return embed_single_sift(image_path, model)

    # Handle SuperPoint separately
    if model_type == "superpoint":
        return embed_single_superpoint(image_path, model, device)

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


def embed_single_sift(image_path: Path, sift_detector) -> np.ndarray:
    """
    Embed a single image using SIFT keypoints and descriptors.

    Args:
        image_path: Path to the image file
        sift_detector: OpenCV SIFT detector object

    Returns:
        numpy array of shape (1, 128), dtype float32
        Note: Returns mean of all SIFT descriptors
    """
    import cv2

    try:
        # Read image
        img_cv = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            log.warning(f"Cannot read image: {image_path}")
            return np.zeros((1, 128), dtype="float32")

        # Detect keypoints and compute descriptors
        kp, des = sift_detector.detectAndCompute(img_cv, None)

        if des is None or len(des) == 0:
            log.warning(f"No SIFT descriptors found: {image_path}")
            return np.zeros((1, 128), dtype="float32")

        # Aggregate descriptors: use mean of all descriptors
        agg_embed = np.mean(des, axis=0).astype("float32")
        return agg_embed.reshape(1, -1)

    except Exception as e:
        log.warning(f"Error embedding {image_path}: {e}")
        return np.zeros((1, 128), dtype="float32")


def embed_single_superpoint(image_path: Path, superpoint, device: str) -> np.ndarray:
    """
    Embed a single image using SuperPoint keypoint detection.

    Args:
        image_path: Path to the image file
        superpoint: Loaded SuperPoint detector model
        device: torch device (cuda)

    Returns:
        numpy array of shape (1, 256), dtype float32
    """
    import torch
    import cv2

    try:
        # Read image
        img_cv = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            log.warning(f"Cannot read image: {image_path}")
            return np.zeros((1, 256), dtype="float32")

        # Resize for consistency
        h, w = img_cv.shape
        scale = 320.0 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_cv, (new_w, new_h))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).float()[None, None].to(device) / 255.0

        with torch.no_grad():
            # Detect keypoints and descriptors
            lafs, resp, desc = superpoint({"image": img_tensor})

        if desc is None or desc.shape[1] == 0:
            log.warning(f"No SuperPoint descriptors found: {image_path}")
            return np.zeros((1, 256), dtype="float32")

        # Aggregate descriptors: use mean of all descriptors
        agg_embed = desc.squeeze().mean(dim=0).cpu().numpy().astype("float32")
        if agg_embed.ndim == 0:
            # Single descriptor case
            agg_embed = desc.squeeze().cpu().numpy().astype("float32")
        return agg_embed.reshape(1, -1)

    except Exception as e:
        log.warning(f"Error embedding {image_path}: {e}")
        return np.zeros((1, 256), dtype="float32")
