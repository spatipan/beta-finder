"""Extract frames from a video file at a fixed interval, skipping the intro."""
import cv2
import numpy as np


def extract_frames(video_path: str, interval_sec: float = 0.5, skip_start_sec: float = 1.0) -> list[np.ndarray]:
    """Return frames sampled every `interval_sec`, skipping the first `skip_start_sec`.

    Args:
        video_path: Path to the video file.
        interval_sec: Sampling interval in seconds (default 0.5 → 2 fps).
        skip_start_sec: Number of seconds to skip at the start (default 1.0).

    Returns:
        List of BGR numpy arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    step = max(1, int(fps * interval_sec))
    start_frame = int(fps * skip_start_sec)

    frames: list[np.ndarray] = []
    frame_idx = start_frame

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += step

    cap.release()
    return frames
