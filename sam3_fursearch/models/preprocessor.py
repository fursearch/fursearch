from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class IsolationConfig:
    mode: str = "solid"
    background_color: tuple[int, int, int] = (128, 128, 128)
    blur_radius: int = 25

    def __post_init__(self):
        if self.mode not in ("none", "solid", "blur"):
            raise ValueError(f"Invalid isolation mode: {self.mode}")


class BackgroundIsolator:
    def __init__(self, config: Optional[IsolationConfig] = None):
        self.config = config or IsolationConfig()

    def isolate(self, crop: Image.Image, mask: np.ndarray) -> Image.Image:
        if self.config.mode == "none":
            return crop
        mask = self._resize_mask(mask, crop.size)
        if self.config.mode == "solid":
            return self._apply_solid_background(crop, mask)
        elif self.config.mode == "blur":
            return self._apply_blurred_background(crop, mask)
        return crop

    def _resize_mask(self, mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = target_size
        mask_h, mask_w = mask.shape[:2]
        if mask_w == target_w and mask_h == target_h:
            return mask
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
        return (np.array(mask_img) > 127).astype(np.uint8)

    def _apply_solid_background(self, crop: Image.Image, mask: np.ndarray) -> Image.Image:
        crop_rgba = crop.convert("RGBA")
        background = Image.new("RGBA", crop.size, self.config.background_color + (255,))
        alpha = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        return Image.composite(crop_rgba, background, alpha).convert("RGB")

    def _apply_blurred_background(self, crop: Image.Image, mask: np.ndarray) -> Image.Image:
        crop_rgba = crop.convert("RGBA")
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=self.config.blur_radius))
        blurred_rgba = blurred.convert("RGBA")
        alpha = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        return Image.composite(crop_rgba, blurred_rgba, alpha).convert("RGB")


def grayscale_preprocessor(image: Image.Image) -> Image.Image:
    """Convert image to grayscale (removes color information)."""
    return image.convert("L").convert("RGB")

grayscale_preprocessor.short_name = "gray"
