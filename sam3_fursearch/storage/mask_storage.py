"""Storage for segmentation masks."""

import re
from pathlib import Path
from typing import Optional
import struct

import numpy as np
from PIL import Image

from sam3_fursearch.config import Config, sanitize_path_component
from sam3_fursearch.models.segmentor import SegmentationResult, mask_to_bbox, create_crop_mask



def _normalize_concept(concept: str) -> str:
    """Normalize concept string for use in path (replace non-alphanumeric with _)."""
    return re.sub(r'[^a-zA-Z0-9]', '_', concept).strip('_') or "default"


class MaskStorage:
    """Handles saving and loading segmentation masks."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(Config.MASKS_DIR)

    def get_mask_dir(self, source: str, model: str, concept: str) -> Path:
        """Get directory for masks: {base}/{source}/{model}/{concept}/"""
        return self.base_dir / sanitize_path_component(source or "unknown") / sanitize_path_component(model) / _normalize_concept(concept)

    def save_mask(
        self,
        mask: np.ndarray,
        name: str,
        source: str,
        model: str,
        concept: str,
    ) -> str:
        """Save a segmentation mask as PNG.

        Args:
            mask: Binary mask array (H, W) with values 0-255 or 0-1
            name: Base name for the mask file (without extension)
            source: Ingestion source (e.g., "tgbot", "manual")
            model: Segmentor model name (e.g., "sam3")
            concept: Segmentation concept (e.g., "fursuiter head")

        Returns:
            Path to the saved mask file
        """
        target_dir = self.get_mask_dir(source, model, concept)
        target_dir.mkdir(parents=True, exist_ok=True)

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        path = target_dir / f"{sanitize_path_component(name)}.png"
        Image.fromarray(mask, mode="L").save(path, optimize=True)
        return str(path)

    def find_masks_for_post(self, post_id: str, source: str, model: str, concept: str):
        """Find all segment masks for a post_id ({post_id}_seg_*.png)."""
        safe_post_id = sanitize_path_component(post_id)
        mask_dir = self.get_mask_dir(source, model, concept)
        return sorted(mask_dir.glob(f"{safe_post_id}_seg_*.png"), key=lambda p: int(p.stem.split("_seg_")[-1]))

    def load_segs_for_post(self, post_id: str, source: str, model: str, concept: str, force_conf: bool = False):
        safe_post_id = sanitize_path_component(post_id)
        results: list[SegmentationResult] = []
        confs: list[float] = []
        mask_dir = self.get_mask_dir(source, model, concept)
        conffile = Path(mask_dir / f"{safe_post_id}.conffile")
        if conffile.exists():
            with open(conffile, 'rb') as f:
                content = f.read()
                size = len(content) // struct.calcsize('d')
                confs = list(struct.unpack(f'{size}d', content))
        masks = self.find_masks_for_post(post_id, source, model, concept)
        if len(masks) > 0 and len(confs) != len(masks):
            print(f"WARN: confidence file mismatch: {conffile}")
            if force_conf:
                return []
            confs = []
        for i, path in enumerate(masks):
            name = path.stem
            seg_idx = int(name.split("_seg_")[-1])
            assert seg_idx == i, f"Missing segment index {i} in mask files"
            mask = self.load_mask(name, source, model, concept)
            bbox = mask_to_bbox(mask)
            if mask is None or bbox is None:
                print(f"WARN: mask {i} is empty for {mask_dir}")
                results.clear()
                return results
            crop_mask = create_crop_mask(mask, bbox)
            results.append(SegmentationResult(
                    crop=None,
                    mask=mask.astype(np.uint8),
                    crop_mask=crop_mask.astype(np.uint8),
                    bbox=bbox,
                    confidence=confs[i] if confs else 1.0,
                    segmentor=model,
                ))
        return results

    def save_segs_for_post(self, post_id: str, source: str, model: str, concept: str, segs: list[SegmentationResult]) -> list[str]:
        safe_post_id = sanitize_path_component(post_id)
        paths = []
        mask_dir = self.get_mask_dir(source, model, concept)
        mask_dir.mkdir(parents=True, exist_ok=True)
        with open(mask_dir / f"{safe_post_id}.conffile", 'wb') as f:
            f.write(bytearray(struct.pack(f'{len(segs)}d', *[s.confidence for s in segs])))
        for i, mask in enumerate([s.mask for s in segs]):
            name = f"{safe_post_id}_seg_{i}"
            path = self.save_mask(mask, name, source, model, concept)
            paths.append(path)
        return paths

    def save_no_segments_marker(self, post_id: str, source: str, model: str, concept: str) -> str:
        """Save a marker indicating the segmentor found no segments for this post."""
        safe_post_id = sanitize_path_component(post_id)
        target_dir = self.get_mask_dir(source, model, concept)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{safe_post_id}.noseg"
        path.touch()
        return str(path)

    def has_no_segments_marker(self, post_id: str, source: str, model: str, concept: str) -> bool:
        """Check if a no-segments marker exists for this post."""
        return (self.get_mask_dir(source, model, concept) / f"{sanitize_path_component(post_id)}.noseg").exists()

    def load_mask(self, name: str, source: str, model: str, concept: str) -> Optional[np.ndarray]:
        path = self.get_mask_dir(source, model, concept) / f"{sanitize_path_component(name)}.png"
        if not path.exists():
            return None
        try:
            return np.array(Image.open(path).convert("L"))
        except Exception:
            print(f"WARN: corrupt mask {path}, deleting")
            path.unlink(missing_ok=True)
            return None
