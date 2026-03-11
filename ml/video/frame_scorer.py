"""Score and select the best frames from a list of candidate frames."""
import cv2
import numpy as np


def _sharpness(frame: np.ndarray) -> float:
    """Laplacian variance — higher means sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _wall_coverage(frame: np.ndarray) -> float:
    """Estimate fraction covered by structured wall texture (holds, volumes).

    Uses edge density as a proxy. Returns a value in [0, 1].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean()) / 255.0


def score_frame(frame: np.ndarray) -> float:
    """Combined quality score (higher is better).

    Weights: sharpness 60% + wall coverage 40% (matches PRD §3.4).
    """
    sharp_norm = min(_sharpness(frame) / 1000.0, 1.0)
    wall = _wall_coverage(frame)
    return sharp_norm * 0.6 + wall * 0.4


def select_top_frames(frames: list[np.ndarray], n: int = 4) -> list[np.ndarray]:
    """Pick the top-n highest-quality frames, then return them in chronological order.

    Args:
        frames: Candidate BGR numpy arrays in extraction order (chronological).
        n: Number of frames to return.

    Returns:
        Up to n frames ordered chronologically (original index preserved),
        so keyframe 0 is always earlier in the video than keyframe 1, etc.
    """
    if not frames:
        return []
    # Tag with original index so we can restore chronological order after scoring
    scored = [(score_frame(f), i, f) for i, f in enumerate(frames)]
    scored.sort(key=lambda x: x[0], reverse=True)   # pick best quality
    top = scored[:n]
    top.sort(key=lambda x: x[1])                     # restore time order
    return [f for _, _, f in top]
