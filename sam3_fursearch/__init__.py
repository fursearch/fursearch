"""SAM3 Fursuit Recognition System

A fursuit character recognition system using SAM3 for detection/segmentation
and DINOv2 for embedding generation.
"""

from sam3_fursearch.api.identifier import (
    FursuitIdentifier,
    IdentificationResult,
    SegmentResults,
)
from sam3_fursearch.api.ingestor import FursuitIngestor
from sam3_fursearch.config import Config
from sam3_fursearch.pipeline.processor import ProcessingResult

__version__ = "1.0.0"
__all__ = ["FursuitIdentifier", "FursuitIngestor", "IdentificationResult", "SegmentResults", "ProcessingResult", "Config"]
