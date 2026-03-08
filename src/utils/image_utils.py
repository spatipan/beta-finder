"""
Image utility functions for BetaFinder CNX
Handles image loading, validation, and metadata extraction
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

from PIL import Image


def load_image_safe(path: Path, fallback_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Load image from path with fallback for corrupt/missing files

    Args:
        path: Path to image file
        fallback_size: Size for blank fallback image (height, width)

    Returns:
        PIL Image object in RGB mode

    Raises:
        FileNotFoundError: If path doesn't exist and no fallback used
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        # Log but don't fail — return blank image
        print(f"⚠️  Could not load image {path}: {e}. Using blank fallback.")
        return Image.new("RGB", fallback_size)


def get_image_metadata(image: Image.Image) -> Dict[str, any]:
    """
    Extract metadata from PIL Image object

    Args:
        image: PIL Image object

    Returns:
        Dictionary with: width, height, size_tuple, format, mode
    """
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "size_tuple": (width, height),
        "format": image.format or "Unknown",
        "mode": image.mode,
        "size_mb": (width * height * len(image.mode)) / (1024 * 1024),
    }


def is_valid_image_path(path: Path) -> bool:
    """Check if path is a valid image file"""
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    return path.suffix.lower() in valid_extensions


if __name__ == "__main__":
    # Test image utils
    from PIL import Image
    test_img = Image.new("RGB", (224, 224))
    metadata = get_image_metadata(test_img)
    print(f"✅ Image metadata: {metadata}")
