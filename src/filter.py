"""
filter.py - Wall classification using CLIP zero-shot learning
Classifies climbing wall images vs. non-wall content (ads, events, selfies, etc.)

Uses:
  - CLIP model (ViT-B-32 or ViT-L-14) for zero-shot classification
  - Text embeddings for "climbing wall" and related prompts
  - Caches results in filter_cache.json for fast re-runs
  - Updates gym_index.json with is_wall (bool) and wall_score (float)

Usage:
    python filter.py                    # Filter all images (use cache)
    python filter.py --rebuild          # Re-score all images
    python filter.py --threshold 0.3    # Use custom threshold
    python filter.py --model ViT-L-14   # Use more accurate model
"""

import json
import torch
import open_clip
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from src.config import load_config, get_path, get_nested
from src.index import load_frames_index
from src.logger import setup_logger
from src.utils.image_utils import load_image_safe

log = setup_logger(__name__)

# Wall classification prompts
WALL_PROMPTS = [
    "climbing wall",
    "bouldering wall",
    "climbing gym",
    "indoor climbing",
    "route on wall",
]


def load_clip_model(model_name: str, device: str) -> Tuple:
    """
    Load CLIP model with preprocessing function

    Args:
        model_name: "ViT-B-32" or "ViT-L-14"
        device: "cuda" or "cpu"

    Returns:
        (model, preprocess, device)
    """
    log.info(f"Loading CLIP model: {model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    model.eval().to(device)
    log.info(f"Model loaded on {device}")
    return model, preprocess, device


def get_text_embeddings(model, device: str) -> torch.Tensor:
    """
    Create CLIP text embeddings for wall classification prompts

    Args:
        model: CLIP model
        device: "cuda" or "cpu"

    Returns:
        Tensor of shape (N_prompts, embedding_dim)
    """
    log.debug(f"Creating text embeddings for {len(WALL_PROMPTS)} prompts")

    with torch.no_grad():
        text_tokens = open_clip.tokenize(WALL_PROMPTS)
        text_tokens = text_tokens.to(device)
        text_embeds = model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    return text_embeds.cpu()


def score_wall_image(
    image_path: Path,
    model,
    preprocess,
    device: str,
    text_embeds: torch.Tensor
) -> float:
    """
    Score an image for wall-likeness using CLIP zero-shot classification

    Args:
        image_path: Path to image file
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: "cuda" or "cpu"
        text_embeds: Text embeddings for wall prompts

    Returns:
        Float in range [-1, 1], higher = more wall-like
    """
    try:
        # Load image
        img = load_image_safe(image_path)

        # Preprocess and encode
        with torch.no_grad():
            tensor = preprocess(img).unsqueeze(0).to(device)
            image_embed = model.encode_image(tensor)
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # Move text embeds to device and compute similarity
        text_embeds_device = text_embeds.to(device)
        similarity = image_embed @ text_embeds_device.T  # shape (1, N_prompts)

        # Use max similarity as wall score
        wall_score = similarity.max().item()

        return wall_score

    except Exception as e:
        log.warning(f"Error scoring {image_path}: {e}")
        return -1.0  # Default to non-wall if error


def load_filter_cache() -> Dict[str, float]:
    """
    Load cached wall scores from filter_cache.json

    Returns:
        {filename: wall_score, ...}
    """
    cache_path = get_path("filter_cache_file") if "filter_cache_file" in load_config().get("paths", {}) else Path("data/filter_cache.json")
    cache_path = Path("data/filter_cache.json")  # Hardcoded for now

    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load cache: {e}. Starting fresh.")

    return {}


def save_filter_cache(cache: Dict[str, float]):
    """Save wall scores cache to file"""
    cache_path = Path("data/filter_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    log.debug(f"Saved cache with {len(cache)} entries")


def filter_images(
    metadata: List[dict],
    cache: Dict[str, float],
    model,
    preprocess,
    device: str,
    text_embeds: torch.Tensor,
    threshold: float,
    rebuild: bool = False
) -> List[dict]:
    """
    Score all images and update metadata with wall classification

    Args:
        metadata: List of gym_index.json entries
        cache: Cached scores from previous runs
        model: CLIP model
        preprocess: CLIP preprocessing
        device: "cuda" or "cpu"
        text_embeds: Text embeddings
        threshold: Classification threshold
        rebuild: If True, ignore cache and re-score everything

    Returns:
        Updated metadata list with is_wall and wall_score fields
    """
    updated = []
    new_scores = 0
    cached_scores = 0

    for entry in tqdm(metadata, desc="Filtering images"):
        filename = entry.get("filename")

        # Check cache first (unless rebuild)
        if not rebuild and filename in cache:
            wall_score = cache[filename]
            cached_scores += 1
        else:
            # Score image
            wall_score = score_wall_image(filename, model, preprocess, device, text_embeds)
            cache[filename] = wall_score
            new_scores += 1

        # Update entry
        entry["wall_score"] = wall_score
        entry["is_wall"] = wall_score > threshold
        updated.append(entry)

    log.info(f"Scoring complete:")
    log.info(f"  New scores: {new_scores}")
    log.info(f"  From cache: {cached_scores}")
    log.info(f"  Total: {len(updated)}")

    return updated


def main():
    parser = argparse.ArgumentParser(description="BetaFinder — Wall Filter")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-score all images (ignore cache)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override config threshold (0-1)")
    parser.add_argument("--model", default=None,
                        help="Override config model (ViT-B-32 or ViT-L-14)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score images but don't update index")

    args = parser.parse_args()

    # Load config
    cfg = load_config()
    wall_filter_cfg = cfg.get("wall_filter", {})

    if not wall_filter_cfg.get("enabled"):
        log.warning("Wall filter disabled in config. Enable with: wall_filter.enabled: true")
        log.info("To enable, edit config/config.yaml")
        return

    # Setup
    model_name = args.model or wall_filter_cfg.get("model", "ViT-B-32")
    threshold = args.threshold or wall_filter_cfg.get("threshold", 0.05)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"========================================")
    log.info(f"  BetaFinder Wall Filter")
    log.info(f"========================================")
    log.info(f"Model: {model_name}")
    log.info(f"Threshold: {threshold}")
    log.info(f"Device: {device}")
    log.info(f"Rebuild: {args.rebuild}")

    # Load model and text embeddings
    log.info("Loading CLIP model...")
    model, preprocess, device = load_clip_model(model_name, device)

    log.info("Creating text embeddings...")
    text_embeds = get_text_embeddings(model, device)

    # Load cache
    log.info("Loading cache...")
    cache = load_filter_cache()
    log.info(f"Cached: {len(cache)} images")

    # Load gym index
    log.info("Loading gym index...")
    index_path = get_path("index_file")
    with open(index_path, "r") as f:
        metadata = json.load(f)
    log.info(f"Total images: {len(metadata)}")

    # Filter images
    log.info(f"Filtering images (threshold={threshold})...")
    updated_metadata = filter_images(
        metadata, cache, model, preprocess, device, text_embeds,
        threshold, rebuild=args.rebuild
    )

    # Save cache
    log.info("Saving cache...")
    save_filter_cache(cache)

    # Update index (unless dry-run)
    if not args.dry_run:
        log.info("Updating gym_index.json...")
        with open(index_path, "w") as f:
            json.dump(updated_metadata, f, ensure_ascii=False, indent=2)

        # Also update frames_index.json if it exists
        frames_file = index_path.parent / "frames_index.json"
        frames = load_frames_index()
        if frames:
            # Build lookup: filename → wall scores from updated gym_index
            score_by_file = {
                m["filename"]: {"is_wall": m.get("is_wall"), "wall_score": m.get("wall_score")}
                for m in updated_metadata
            }
            f_updated = 0
            for frame in frames:
                fname = frame.get("filename", "")
                if fname in score_by_file:
                    frame["is_wall"]   = score_by_file[fname]["is_wall"]
                    frame["wall_score"] = score_by_file[fname]["wall_score"]
                    f_updated += 1
            with open(frames_file, "w", encoding="utf-8") as f:
                json.dump(frames, f, ensure_ascii=False, indent=2)
            log.info(f"✅ frames_index.json updated: {f_updated} frames with wall scores")
    else:
        log.info("(Dry-run mode — not updating index)")

    # Stats
    walls = sum(1 for m in updated_metadata if m.get("is_wall", False))
    non_walls = len(updated_metadata) - walls

    log.info(f"========================================")
    log.info(f"  Results")
    log.info(f"========================================")
    log.info(f"Walls: {walls} ({100*walls/len(updated_metadata):.1f}%)")
    log.info(f"Non-walls: {non_walls} ({100*non_walls/len(updated_metadata):.1f}%)")

    # Per-gym stats
    by_gym = {}
    for entry in updated_metadata:
        gym = entry.get("gym") or entry.get("source_key", "unknown")
        if gym not in by_gym:
            by_gym[gym] = {"walls": 0, "total": 0}
        by_gym[gym]["total"] += 1
        if entry.get("is_wall"):
            by_gym[gym]["walls"] += 1

    log.info("")
    log.info("By Gym:")
    for gym, stats in sorted(by_gym.items()):
        pct = 100 * stats["walls"] / stats["total"] if stats["total"] > 0 else 0
        log.info(f"  {gym}: {stats['walls']}/{stats['total']} ({pct:.1f}%)")

    log.info(f"========================================")
    log.info(f"✅ Filter complete!")


if __name__ == "__main__":
    main()
