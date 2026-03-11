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


def is_climbing_wall(
    frame: "np.ndarray | Image.Image",
    threshold: float = 0.20,
) -> tuple[bool, float]:
    """Classify whether a frame shows a climbing wall.

    Args:
        frame: BGR numpy array (from cv2) or PIL Image.
        threshold: Score threshold; default 0.20 (lowered from 0.272 — real gym
            footage from the_alpine_outpost clusters in 0.19–0.25, while clearly
            non-wall content scores below 0.15).

    Returns:
        (is_wall, final_score) where final_score = pos_mean - 0.5 * neg_mean.
    """
    _load()
    device = next(_model.parameters()).device

    if isinstance(frame, np.ndarray):
        pil = Image.fromarray(frame[..., ::-1])  # BGR → RGB
    else:
        pil = frame

    image_input = _preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = _model.encode_image(image_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ _text_features.T)[0]

    pos_score = sims[:_N_POS].mean()
    neg_score = sims[_N_POS:].mean()  # fixed: original used len(NEGATIVE_PROMPTS) which sliced from wrong end
    final = pos_score - 0.5 * neg_score

    return final.item() > threshold, final.item()
