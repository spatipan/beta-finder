"""DINOv2-base embedder — produces a 768-dim CLS token per image."""
from __future__ import annotations

import torch
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image


class DinoEmbedder:
    """Wraps facebook/dinov2-base for single-image and multi-frame embedding."""

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name).eval().to(self.device)

    def embed(self, image: Image.Image) -> torch.Tensor:
        """Embed a single PIL image → 1×768 tensor (CLS token)."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    def embed_frames(self, frames: list[Image.Image]) -> torch.Tensor:
        """Batch-embed multiple frames and average → 1×768 tensor per Reel.

        Uses a single processor + model forward pass for all frames instead of
        calling embed() per frame. Functionally identical output, but avoids
        N separate forward passes (minor benefit on CPU, larger on GPU).

        Args:
            frames: List of PIL Images (e.g. top-4 keyframes for a Reel).

        Returns:
            1×768 tensor — mean CLS token across all frames.
        """
        if not frames:
            raise ValueError("frames list is empty")
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (N, 768)
        return cls_tokens.mean(dim=0, keepdim=True)       # (1, 768)

    def embed_numpy(self, frame_bgr: "np.ndarray") -> torch.Tensor:
        """Convenience: embed a BGR numpy array directly."""
        pil = Image.fromarray(frame_bgr[..., ::-1])  # BGR → RGB
        return self.embed(pil)
