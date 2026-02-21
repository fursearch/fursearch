"""API components for the SAM3 fursuit recognition system."""

from sam3_fursearch.api.annotator import annotate_image
from sam3_fursearch.api.identifier import FursuitIdentifier, IdentificationResult, detect_embedder, build_embedder_for_name, merge_multi_dataset_results
from sam3_fursearch.api.ingestor import FursuitIngestor

__all__ = ["FursuitIdentifier", "FursuitIngestor", "IdentificationResult", "annotate_image", "detect_embedder", "build_embedder_for_name", "merge_multi_dataset_results"]
