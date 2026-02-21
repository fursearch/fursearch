"""Image processing pipeline for fursuit recognition."""

from sam3_fursearch.pipeline.processor import (
    CacheKey,
    CachedProcessingPipeline,
    ProcessingResult,
)

__all__ = ["CachedProcessingPipeline", "ProcessingResult", "CacheKey"]
