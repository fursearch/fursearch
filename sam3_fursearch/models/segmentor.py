from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sam3_fursearch.config import Config


def mask_to_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Convert binary mask to bounding box (x, y, w, h)."""
    if mask is None:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


def create_crop(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


def create_crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    cropped = mask[y:y + h, x:x + w]
    return (cropped > 0).astype(np.uint8)  # Normalize to binary 0-1


@dataclass
class SegmentationResult:
    crop: Image.Image
    mask: np.ndarray
    crop_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    segmentor: str = "unknown"

    @classmethod
    def from_mask(cls, image: Image.Image, mask: np.ndarray, segmentor: str = "unknown", confidence: float = 1.0):
        bbox = mask_to_bbox(mask)
        if bbox is None:
            return None
        return cls(
            crop=create_crop(image, bbox),
            mask=mask,
            crop_mask=create_crop_mask(mask, bbox),
            bbox=bbox,
            confidence=confidence,
            segmentor=segmentor,
        )

class FullImageSegmentor:
    """A fallback segmentor that returns the full image as a single segment."""
    def __init__(self) -> None:
        self.model_name = "full"

    def segment(self, image: Image.Image, confidence: float = 1.0) -> list[SegmentationResult]:
        # print("Using FullImageSegmentor: returning full image as single segment")
        w, h = image.size
        full_mask = np.ones((h, w), dtype=np.uint8)
        return [SegmentationResult(
            crop=image.copy(),
            mask=full_mask,
            crop_mask=full_mask,
            bbox=(0, 0, w, h),
            confidence=confidence,
            segmentor=self.model_name,
        )]

class SAM3FursuitSegmentor:

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        confidence_threshold: float = Config.DETECTION_CONFIDENCE,
        max_detections: int = Config.MAX_DETECTIONS,
        concept: Optional[str] = Config.DEFAULT_CONCEPT,
    ):
        self.device = device or Config.get_segmentor_device()
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.model_name = model_name or Config.SAM3_MODEL
        self.concept = concept or Config.DEFAULT_CONCEPT
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from ultralytics.models.sam.predict import SAM3SemanticPredictor

            overrides = dict(
                conf=self.confidence_threshold,
                task="segment",
                mode="predict",
                model=f"{self.model_name}.pt",
                device=self.device,
                imgsz=Config.SAM3_IMAGE_SIZE,
                verbose=False,
                save=False,
            )
            predictor = SAM3SemanticPredictor(overrides=overrides)
            self._predictor = predictor
        return self._predictor


    def segment(self, image: Image.Image) -> list[SegmentationResult]:
        print(f"Segmenting image using {self.model_name} with concept '{self.concept}'")
        image_np = np.array(image.convert("RGB"))
        self.predictor.set_image(image_np)
        results = self.predictor(text=self.concept.split(","))
        segmentation_results = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i, mask in enumerate(masks):

                confidence = 1.0
                if boxes is not None and boxes.conf is not None and len(boxes.conf) > i:
                    confidence = float(boxes.conf[i])

                if confidence < self.confidence_threshold:
                    continue

                bbox = mask_to_bbox(mask)
                if bbox is None:
                    continue

                crop = create_crop(image, bbox)
                crop_mask = create_crop_mask(mask, bbox)
                segmentation_results.append(SegmentationResult(
                    crop=crop,
                    mask=mask.astype(np.uint8),
                    crop_mask=crop_mask,
                    bbox=bbox,
                    confidence=confidence,
                    segmentor=self.model_name,
                ))

        # Filter out small segments
        if segmentation_results:
            # Filter by absolute minimum size
            before = len(segmentation_results)
            segmentation_results = [
                s for s in segmentation_results
                if s.bbox[2] >= Config.MIN_SEGMENT_SIZE and s.bbox[3] >= Config.MIN_SEGMENT_SIZE
            ]
            if len(segmentation_results) < before:
                print(f"Filtered {before - len(segmentation_results)} segments below {Config.MIN_SEGMENT_SIZE}px minimum size")

            # Filter by ratio to average bbox area
            if len(segmentation_results) >= 2:
                areas = [s.bbox[2] * s.bbox[3] for s in segmentation_results]
                avg_area = sum(areas) / len(areas)
                before = len(segmentation_results)
                segmentation_results = [
                    s for s in segmentation_results
                    if (s.bbox[2] * s.bbox[3]) >= avg_area * Config.MIN_SEGMENT_RATIO
                ]
                if len(segmentation_results) < before:
                    print(f"Filtered {before - len(segmentation_results)} segments below {Config.MIN_SEGMENT_RATIO:.0%} of average area")

        # Sort by confidence and limit to max_detections
        segmentation_results.sort(key=lambda s: s.confidence, reverse=True)
        if len(segmentation_results) > self.max_detections:
            print(f"Limiting segments to top {self.max_detections} detections from {len(segmentation_results)} found.")
            segmentation_results = segmentation_results[:self.max_detections]

        return segmentation_results

