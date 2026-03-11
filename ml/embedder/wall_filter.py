"""CLIP-based climbing wall classifier (ensemble prompts, threshold 0.272)."""
from __future__ import annotations

import torch
import clip
from PIL import Image
import numpy as np

# --- Prompts (PRD §3.3.1) ---------------------------------------------------

POSITIVE_PROMPTS = [
    "a climbing wall with colorful holds",
    "indoor bouldering wall",
    "rock climbing gym wall",
    "climbing holds on a wall",
    "a photo of a climbing route",
]

NEGATIVE_PROMPTS = [
    "a person standing",
    "a close up of a face",
    "outdoor scenery",
    "food or drink",
]

_ALL_PROMPTS = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
_N_POS = len(POSITIVE_PROMPTS)

# Lazy-loaded globals so importing the module doesn't trigger a download
_model = None
_preprocess = None
_text_features: torch.Tensor | None = None


def _load() -> None:
    global _model, _preprocess, _text_features
    if _model is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(_ALL_PROMPTS).to(device)
    with torch.no_grad():
        _text_features = _model.encode_text(text_tokens)
        _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)


def filter_wall_frames(
    frames: "list[np.ndarray | Image.Image]",
    threshold: float = 0.20,
) -> list[tuple[bool, float]]:
    """Batch CLIP wall classifier — one forward pass for all frames.

    Designed for the sparse wall-check stage: preprocess N frames (typically
    ~11 at 5s intervals), stack into a single (N, C, H, W) tensor, and run
    one model.encode_image() call instead of N separate calls.

    Args:
        frames: List of BGR numpy arrays or PIL Images.
        threshold: Same scoring formula as is_climbing_wall().

    Returns:
        List of (is_wall, score) tuples in the same order as input frames.
    """
    if not frames:
        return []

    _load()
    device = next(_model.parameters()).device

    tensors = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            pil = Image.fromarray(frame[..., ::-1])  # BGR → RGB
        else:
            pil = frame
        tensors.append(_preprocess(pil))

    batch = torch.stack(tensors).to(device)  # (N, C, H, W)

    with torch.no_grad():
        img_feats = _model.encode_image(batch)                           # (N, D)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        sims = img_feats @ _text_features.T                              # (N, P)

    pos_scores = sims[:, :_N_POS].mean(dim=1)    # (N,)
    neg_scores = sims[:, _N_POS:].mean(dim=1)    # (N,)
    finals = pos_scores - 0.5 * neg_scores        # (N,)

    return [(s.item() > threshold, s.item()) for s in finals]


def is_climbing_wall(
    frame: "np.ndarray | Image.Image",
    threshold: float = 0.20,
) -> tuple[bool, float]:
    """Classify whether a single frame shows a climbing wall.

    Delegates to filter_wall_frames() to avoid duplicating the preprocessing,
    encoding, and scoring logic.

    Args:
        frame: BGR numpy array (from cv2) or PIL Image.
        threshold: Score threshold; default 0.20 (lowered from 0.272 — real gym
            footage from the_alpine_outpost clusters in 0.19–0.25, while clearly
            non-wall content scores below 0.15).

    Returns:
        (is_wall, final_score) where final_score = pos_mean - 0.5 * neg_mean.
    """
    return filter_wall_frames([frame], threshold=threshold)[0]


def find_optimal_threshold(
    labeled_frames: "list[tuple[np.ndarray | Image.Image, bool]]",
) -> float:
    """Find F1-optimal CLIP threshold from labeled frames (PRD §3.2).

    Args:
        labeled_frames: List of (frame, is_wall) pairs. frame is a BGR ndarray
                        or PIL Image. Recommended: 200–500 frames from real gym
                        footage.

    Returns:
        Optimal threshold that maximises F1 on the provided labels.

    Usage:
        from ml.embedder.wall_filter import find_optimal_threshold
        from PIL import Image
        import os

        labeled = []
        for f in os.listdir("data/labeled/wall"):
            labeled.append((Image.open(f"data/labeled/wall/{f}"), True))
        for f in os.listdir("data/labeled/not_wall"):
            labeled.append((Image.open(f"data/labeled/not_wall/{f}"), False))

        threshold = find_optimal_threshold(labeled)
        print(f"Optimal threshold: {threshold:.3f}")
        # Then update: ml.wall_filter_threshold in accounts.yaml
    """
    from sklearn.metrics import precision_recall_curve

    frames = [f for f, _ in labeled_frames]
    labels = [int(is_wall) for _, is_wall in labeled_frames]

    # threshold=-999 ensures every frame passes — we only want the raw score
    raw_results = filter_wall_frames(frames, threshold=-999)
    scores = [score for _, score in raw_results]

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(f1.argmax())
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.20
