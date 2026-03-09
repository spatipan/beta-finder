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
    python filter.py --stats            # Show score distribution and recommend threshold
"""

import json
import torch
import open_clip
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from src.config import load_config, get_path, get_nested
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


def analyze_score_distribution(scores: List[float]) -> Dict:
    """
    Analyze distribution of wall scores and recommend threshold
    
    Args:
        scores: List of wall scores
        
    Returns:
        Dictionary with statistics and recommended threshold
    """
    scores_array = np.array(scores)
    
    # Basic statistics
    stats = {
        'count': len(scores),
        'mean': float(np.mean(scores_array)),
        'median': float(np.median(scores_array)),
        'std': float(np.std(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
    }
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    stats['percentiles'] = {
        p: float(np.percentile(scores_array, p)) 
        for p in percentiles
    }
    
    # Histogram bins
    hist, bin_edges = np.histogram(scores_array, bins=20)
    stats['histogram'] = {
        'counts': hist.tolist(),
        'edges': bin_edges.tolist()
    }
    
    # Recommend threshold using Otsu's method (bimodal separation)
    # Sort scores and find threshold that maximizes between-class variance
    sorted_scores = np.sort(scores_array)
    best_threshold = stats['median']
    max_variance = 0
    
    # Test thresholds at every 5th percentile
    for i in range(10, 90, 5):
        threshold = np.percentile(scores_array, i)
        below = scores_array[scores_array <= threshold]
        above = scores_array[scores_array > threshold]
        
        if len(below) > 0 and len(above) > 0:
            # Between-class variance
            w_below = len(below) / len(scores_array)
            w_above = len(above) / len(scores_array)
            var_between = w_below * w_above * (np.mean(below) - np.mean(above)) ** 2
            
            if var_between > max_variance:
                max_variance = var_between
                best_threshold = threshold
    
    stats['recommended_threshold'] = float(best_threshold)
    
    # Alternative recommendations
    stats['conservative_threshold'] = float(np.percentile(scores_array, 25))  # Keep more images
    stats['aggressive_threshold'] = float(np.percentile(scores_array, 75))    # Filter more aggressively
    
    return stats


def print_distribution_stats(stats: Dict, current_threshold: float = None):
    """Pretty print score distribution statistics"""
    log.info("=" * 60)
    log.info("  SCORE DISTRIBUTION ANALYSIS")
    log.info("=" * 60)
    log.info(f"Total images: {stats['count']}")
    log.info(f"Score range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    log.info(f"Mean: {stats['mean']:.3f}")
    log.info(f"Median: {stats['median']:.3f}")
    log.info(f"Std Dev: {stats['std']:.3f}")
    log.info("")
    
    log.info("Percentiles:")
    for p, val in stats['percentiles'].items():
        log.info(f"  {p:>3}%: {val:>6.3f}")
    log.info("")
    
    # ASCII histogram
    log.info("Score Distribution (histogram):")
    hist_counts = stats['histogram']['counts']
    hist_edges = stats['histogram']['edges']
    max_count = max(hist_counts) if hist_counts else 1
    
    for i, count in enumerate(hist_counts):
        bin_start = hist_edges[i]
        bin_end = hist_edges[i + 1]
        bar_length = int(40 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_length
        log.info(f"  [{bin_start:>5.2f}, {bin_end:>5.2f}): {bar} {count}")
    log.info("")
    
    # Threshold recommendations
    log.info("=" * 60)
    log.info("  THRESHOLD RECOMMENDATIONS")
    log.info("=" * 60)
    log.info(f"🎯 Recommended (Otsu): {stats['recommended_threshold']:.3f}")
    log.info(f"   (Maximizes separation between walls/non-walls)")
    log.info("")
    log.info(f"🛡️  Conservative: {stats['conservative_threshold']:.3f}")
    log.info(f"   (Keeps more images, fewer false negatives)")
    log.info("")
    log.info(f"⚡ Aggressive: {stats['aggressive_threshold']:.3f}")
    log.info(f"   (Filters more strictly, fewer false positives)")
    log.info("")
    
    if current_threshold is not None:
        log.info(f"Current threshold: {current_threshold:.3f}")
        below_current = sum(1 for s in stats['histogram']['counts'][:10])
        log.info(f"  → Would classify ~{100 * (1 - current_threshold):.1f}% as walls")
    
    log.info("=" * 60)


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
    parser.add_argument("--stats", action="store_true",
                        help="Show score distribution and recommend threshold")

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

    # Show statistics if requested
    if args.stats:
        scores = [m.get("wall_score", 0.0) for m in updated_metadata]
        stats = analyze_score_distribution(scores)
        print_distribution_stats(stats, current_threshold=threshold)

    # Save cache
    log.info("Saving cache...")
    save_filter_cache(cache)

    # Update index (unless dry-run)
    if not args.dry_run:
        log.info("Updating gym_index.json...")
        with open(index_path, "w") as f:
            json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
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