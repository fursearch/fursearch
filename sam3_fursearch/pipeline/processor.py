"""Processing pipeline: segmentation, isolation, embedding."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image

from sam3_fursearch.config import Config
from sam3_fursearch.models.embedder import SigLIPEmbedder
from sam3_fursearch.models.preprocessor import BackgroundIsolator, IsolationConfig
from sam3_fursearch.models.segmentor import (
    FullImageSegmentor,
    SAM3FursuitSegmentor,
    SegmentationResult,
)
from sam3_fursearch.storage.mask_storage import MaskStorage


@dataclass
class CacheKey:
    post_id: str
    source: str


@dataclass
class ProcessingResult:
    segmentation: SegmentationResult
    embedding: np.ndarray
    isolated_crop: Optional[Image.Image] = None
    segmentor_model: str = "unknown"
    segmentor_concept: Optional[str] = None
    mask_reused: bool = False
    preprocessing_info: str = ""


# Mapping from pipeline short embedder names to CLI --embedder values
SHORT_NAME_TO_CLI = {
    "dv2b": "dinov2-base",
    "dv2l": "dinov2-large",
    "dv2g": "dinov2-giant",
    "clip": "clip",
    "siglip": "siglip",
    "dv2b+chist": "dinov2-base+colorhist",
}

class CachedProcessingPipeline:
    """Segment, isolate, and embed an image."""

    def __init__(
        self,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
        mask_storage = MaskStorage(),
        embedder = None,
        preprocessors: Optional[list[Callable[[Image.Image], Image.Image]]] = None,
    ):
        self.device = Config.get_device()
        self.mask_storage = mask_storage
        segmentor_device = device or Config.get_segmentor_device()
        if segmentor_model_name == Config.SAM3_MODEL:
            self.segmentor = SAM3FursuitSegmentor(device=segmentor_device, concept=segmentor_concept)
        else:
            self.segmentor = FullImageSegmentor()
        self.segmentor_concept = segmentor_concept or ""
        if embedder is not None:
            self.embedder = embedder
            self.embedder_model_name = embedder.model_name
        else:
            self.embedder_model_name = Config.SIGLIP_MODEL
            self.embedder = SigLIPEmbedder(device=self.device, model_name=self.embedder_model_name)
        self.preprocessors = preprocessors or []
        self.isolator = BackgroundIsolator(isolation_config)
        self.isolation_config = self.isolator.config


    def get_embedder_short_name(self) -> str:
        emb = self.embedder_model_name
        if "+colorhist" in emb:
            base = emb.replace("+colorhist", "")
            return self._short_name_for(base) + "+chist"
        return self._short_name_for(emb)

    @staticmethod
    def _short_name_for(emb: str) -> str:
        # Check if any CLI name is a substring of the full model name
        for short, cli in SHORT_NAME_TO_CLI.items():
            if cli in emb:
                return short
        return emb.split("/")[-1][:8]

    def build_preprocessing_info(self) -> str:
        """Build fingerprint for pipeline."""
        parts = ["v2", f"seg:{self.segmentor.model_name}"]
        if self.segmentor.model_name != "full":
            iso = self.isolation_config
            mode_map = {"solid": "s", "blur": "b", "none": "n"}
            parts += [
                f"con:{(self.segmentor_concept or '').replace('|', '.')}",
                f"bg:{mode_map.get(iso.mode, 'n')}",
            ]
            if iso.mode == "solid":
                r, g, b = iso.background_color
                parts += [f"bgc:{r:02x}{g:02x}{b:02x}"]
            elif iso.mode == "blur":
                parts += [f"br:{iso.blur_radius}"]
        if self.preprocessors:
            pp_names = [getattr(pp, "short_name", getattr(pp, "__name__", type(pp).__name__)) for pp in self.preprocessors]
            parts += [f"pp:{'+'.join(pp_names)}"]
        parts += [f"emb:{self.get_embedder_short_name()}", f"tsz:{Config.TARGET_IMAGE_SIZE}"]
        return "|".join(parts)


    def process(self, image: Image.Image, cache_key: Optional[CacheKey] = None) -> list[ProcessingResult]:
        if max(*image.size) > Config.MAX_INPUT_IMAGE_SIZE:
            new_size = self._scale_image_size(image, Config.MAX_INPUT_IMAGE_SIZE)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        segmentations, mask_reused = self._segment(image, cache_key)
        proc_results = []
        for seg in segmentations:
            preprocessing_info = self.build_preprocessing_info()
            isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
            isolated_crop = self._resize_to_patch_multiple(isolated)
            for pp in self.preprocessors:
                isolated_crop = pp(isolated_crop)
            embedding = self.embedder.embed(isolated_crop)
            proc_results.append(ProcessingResult(
                segmentation=seg,
                embedding=embedding,
                isolated_crop=isolated_crop,
                segmentor_model=seg.segmentor,
                segmentor_concept=self.segmentor_concept,
                mask_reused=mask_reused,
                preprocessing_info=preprocessing_info,
            ))
        return proc_results

    def _scale_image_size(self, image: Image.Image, target_size: int):
        w, h = image.size
        if w >= h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        return new_w, new_h

    def _resize_to_patch_multiple(self, image: Image.Image, target_size: int = Config.TARGET_IMAGE_SIZE):
        new_w, new_h = self._scale_image_size(image, target_size)
        new_w = max(Config.PATCH_SIZE, (new_w // Config.PATCH_SIZE) * Config.PATCH_SIZE)
        new_h = max(Config.PATCH_SIZE, (new_h // Config.PATCH_SIZE) * Config.PATCH_SIZE)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _segment(self, image: Image.Image, cache_key: Optional[CacheKey] = None) -> tuple[list[SegmentationResult], bool]:
        if cache_key is not None:
            segmentations = self._load_segments_for_post(
                cache_key.post_id, cache_key.source,
                self.segmentor.model_name, self.segmentor_concept, image,
            )
            if segmentations:
                return segmentations, True
            mask_dir = self.mask_storage.get_mask_dir(cache_key.source, self.segmentor.model_name, self.segmentor_concept)
            print(f"WARN: not using pre-computed segmentations for {mask_dir / (cache_key.post_id + '_seg_*.png')}")

        segmentations = self.segmentor.segment(image)

        if cache_key is not None:
            try:
                segs = [seg for seg in segmentations if seg.mask is not None]
                if segs:
                    self.mask_storage.save_segs_for_post(cache_key.post_id, cache_key.source, self.segmentor.model_name, self.segmentor_concept, segs)
                else:
                    self.mask_storage.save_no_segments_marker(cache_key.post_id, cache_key.source, self.segmentor.model_name, self.segmentor_concept)
            except Exception as e:
                print(f"Failed to save segments for {cache_key.post_id}: {e}")

        return segmentations, False

    def _load_segments_for_post(self, post_id: str, source: str, model: str, concept: str, image: Image.Image) -> list[SegmentationResult]:
        if self.mask_storage.has_no_segments_marker(post_id, source, model, concept):
            return FullImageSegmentor().segment(image)
        segs = self.mask_storage.load_segs_for_post(post_id, source, model, concept, force_conf=True)
        for seg in segs:
            if (seg.mask.shape[1], seg.mask.shape[0]) != image.size:
                print(f"WARN: resizing cached mask {seg.mask.shape[:2][::-1]} -> {image.size} for {post_id}")
                seg.mask = np.array(Image.fromarray(seg.mask).resize(image.size, Image.Resampling.NEAREST))
        segmentations = [
            SegmentationResult.from_mask(image, seg.mask, segmentor=self.segmentor.model_name, confidence=seg.confidence) for seg in segs
        ]
        segmentations = [s for s in segmentations if s]
        return segmentations
