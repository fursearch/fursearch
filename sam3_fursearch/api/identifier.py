import hashlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sam3_fursearch.config import Config, sanitize_path_component
from sam3_fursearch.models.preprocessor import IsolationConfig
from sam3_fursearch.pipeline.processor import CachedProcessingPipeline, CacheKey, SHORT_NAME_TO_CLI
from sam3_fursearch.storage.database import Database
from sam3_fursearch.storage.vector_index import VectorIndex


@dataclass
class IdentificationResult:
    character_name: Optional[str]
    confidence: float
    distance: float
    post_id: str
    bbox: tuple[int, int, int, int]
    segmentor_model: str = "unknown"
    source: Optional[str] = None


@dataclass
class SegmentResults:
    segment_index: int
    segment_bbox: tuple[int, int, int, int]
    segment_confidence: float
    matches: list[IdentificationResult]


# ---------------------------------------------------------------------------
# Search strategies — how to rank matches within a single dataset
# ---------------------------------------------------------------------------

class SearchStrategy(ABC):
    """Strategy for querying the FAISS index within a single dataset."""

    @abstractmethod
    def search(
        self,
        index: VectorIndex,
        db: Database,
        embedding: np.ndarray,
        top_k: int,
    ) -> list[IdentificationResult]:
        ...


class AvgEmbeddingStrategy(SearchStrategy):
    """Average closest embeddings per (character, segmentor) group, re-rank by distance.

    Only averages embeddings that share the same segmentor_model to avoid
    mixing different preprocessing pipelines (e.g. sam3 crops vs full images).
    Far matches (> FAISS_MAX_DISTANCE) are excluded from averaging.
    """

    def search(
        self,
        index: VectorIndex,
        db: Database,
        embedding: np.ndarray,
        top_k: int,
    ) -> list[IdentificationResult]:
        fetch_k = min(top_k * 6, index.size)
        distances, raw_indices = index.search(embedding, fetch_k)
        # Discard far matches to avoid polluting averages with unrelated embeddings
        indices = [np.where(distances[0] <= Config.FAISS_MAX_DISTANCE, raw_indices[0], -1)]

        # Group FAISS indices by (character, segmentor_model)
        group_indices: dict[tuple[str, str], list[int]] = {}
        group_detection: dict[tuple[str, str], object] = {}
        for idx in indices[0]:
            if idx == -1:
                continue
            detection = db.get_detection_by_embedding_id(int(idx))
            if detection is None:
                continue
            key = ((detection.character_name or "").lower(), detection.segmentor_model)
            group_indices.setdefault(key, []).append(int(idx))
            if key not in group_detection:
                group_detection[key] = detection

        if not group_indices:
            return []

        query = embedding.reshape(1, -1).astype(np.float32)
        results: list[IdentificationResult] = []
        for key, faiss_ids in group_indices.items():
            embs = np.stack([index.reconstruct(i) for i in faiss_ids])
            avg_emb = embs.mean(axis=0, keepdims=True).astype(np.float32)
            dist = float(np.sum((query - avg_emb) ** 2))
            confidence = max(0.0, 1.0 - dist / 2.0)
            det = group_detection[key]
            results.append(IdentificationResult(
                character_name=det.character_name,
                confidence=confidence,
                distance=dist,
                post_id=det.post_id,
                bbox=(det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height),
                segmentor_model=det.segmentor_model,
                source=det.source,
            ))

        # Keep best match per character across segmentor groups
        results.sort(key=lambda x: x.confidence, reverse=True)
        seen: set[str] = set()
        deduped: list[IdentificationResult] = []
        for r in results:
            char_key = (r.character_name or "").lower()
            if char_key not in seen:
                seen.add(char_key)
                deduped.append(r)
        return deduped[:top_k]


# ---------------------------------------------------------------------------
# Merge strategies — how to combine results across multiple datasets
# ---------------------------------------------------------------------------

class MergeStrategy(ABC):
    """Strategy for merging SegmentResults from multiple datasets."""

    @abstractmethod
    def merge(
        self,
        all_results: list[list[SegmentResults]],
        top_k: int,
    ) -> list[SegmentResults]:
        ...


class ConfidenceMerge(MergeStrategy):
    """Keep the best-confidence match per character across all datasets."""

    def merge(
        self,
        all_results: list[list[SegmentResults]],
        top_k: int,
    ) -> list[SegmentResults]:
        base = next((r for r in all_results if r), None)
        if base is None:
            return []

        merged = []
        for seg_idx, base_seg in enumerate(base):
            char_best: dict[str, IdentificationResult] = {}
            for dataset_results in all_results:
                if seg_idx < len(dataset_results):
                    for m in dataset_results[seg_idx].matches:
                        key = (m.character_name or "").lower()
                        if key not in char_best or m.confidence > char_best[key].confidence:
                            char_best[key] = m
            matches = sorted(char_best.values(), key=lambda m: m.confidence, reverse=True)[:top_k]
            merged.append(SegmentResults(
                segment_index=base_seg.segment_index,
                segment_bbox=base_seg.segment_bbox,
                segment_confidence=base_seg.segment_confidence,
                matches=matches,
            ))
        return merged


_SEARCH_STRATEGIES: dict[str, type[SearchStrategy]] = {
    "avg_embedding": AvgEmbeddingStrategy,
}

_MERGE_STRATEGIES: dict[str, type[MergeStrategy]] = {
    "confidence": ConfidenceMerge,
}

_DEFAULT_SEARCH_STRATEGY: SearchStrategy = _SEARCH_STRATEGIES[Config.SEARCH_STRATEGY]()
_DEFAULT_MERGE_STRATEGY: MergeStrategy = _MERGE_STRATEGIES[Config.MERGE_STRATEGY]()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_debug_crop(
    image: Image.Image,
    name: str,
    source: Optional[str] = None,
) -> None:
    """Save crop image for debugging (optional, use --save-crops)."""
    crops_dir = Path(Config.CROPS_INGEST_DIR) / sanitize_path_component(source or "unknown")
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crops_dir / f"{sanitize_path_component(name)}.jpg"
    image.convert("RGB").save(crop_path, quality=90)


def discover_datasets(base_dir: str = Config.BASE_DIR) -> list[tuple[str, str]]:
    """Find all (db_path, index_path) pairs in base_dir.

    Matches *.db files that have a corresponding *.index file.
    Ignores backup files (*.backup.*, *.bak*).
    """
    base = Path(base_dir)
    if not base.is_dir():
        return []
    datasets = []
    for db_file in sorted(base.glob("*.db")):
        name = db_file.name
        if ".backup." in name or ".bak" in name:
            continue
        index_file = db_file.with_suffix(".index")
        if index_file.exists() and ".backup." not in index_file.name and ".bak" not in index_file.name:
            datasets.append((str(db_file), str(index_file)))
    return datasets


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

class FursuitIdentifier:
    """Read-only identification across a dataset."""

    def __init__(
        self,
        db_path: str,
        index_path: str,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
        embedder=None,
        preprocessors: Optional[list] = None,
        search_strategy: Optional[SearchStrategy] = None,
    ):
        if not embedder:
            emb = detect_embedder(db_path=db_path)
            embedder = build_embedder_for_name(short_name=emb, device=device)

        embedding_dim = embedder.embedding_dim if embedder else Config.EMBEDDING_DIM

        self.db = Database(db_path)
        self.index = VectorIndex(index_path, embedding_dim=embedding_dim)
        self.search_strategy = search_strategy or _DEFAULT_SEARCH_STRATEGY

        # Non-cached pipeline for query processing only
        self.pipeline = CachedProcessingPipeline(
            device=device,
            isolation_config=isolation_config,
            segmentor_model_name=segmentor_model_name,
            segmentor_concept=segmentor_concept,
            embedder=embedder,
            preprocessors=preprocessors,
        )

        self.fallback_pipeline = CachedProcessingPipeline(
            device=device,
            isolation_config=isolation_config,
            segmentor_model_name="full",
            segmentor_concept="",
            embedder=embedder,
            preprocessors=preprocessors,
        )

    @property
    def total_index_size(self) -> int:
        return self.index.size

    def identify(
        self,
        image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        save_crops: bool = False,
        crop_prefix: str = "query",
    ) -> list[SegmentResults]:
        if self.total_index_size == 0:
            print("WARN: Index is empty, no matches possible")
            return []

        # Generate cache key from image content so segmentation masks are
        # reused when the same image is processed by multiple identifiers.
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        cache_key = CacheKey(post_id=image_hash, source="query")

        proc_results = self.pipeline.process(image, cache_key=cache_key)
        if not proc_results:
            proc_results = self.fallback_pipeline.process(image, cache_key=cache_key)

        segment_results = []
        for i, proc_result in enumerate(proc_results):
            if save_crops and proc_result.isolated_crop:
                _save_debug_crop(proc_result.isolated_crop, f"{crop_prefix}_{i}", source="search")
            matches = self._search_embedding(proc_result.embedding, top_k)
            segment_results.append(SegmentResults(
                segment_index=i,
                segment_bbox=proc_result.segmentation.bbox,
                segment_confidence=proc_result.segmentation.confidence,
                matches=matches,
            ))
        return segment_results

    def _search_embedding(self, embedding: np.ndarray, top_k: int) -> list[IdentificationResult]:
        if self.total_index_size == 0:
            return []
        return self.search_strategy.search(self.index, self.db, embedding, top_k)

    def search_text(self, text: str, top_k: int = Config.DEFAULT_TOP_K) -> list[IdentificationResult]:
        """Search for characters by text description. Requires CLIP or SigLIP embedder."""
        embedder = self.pipeline.embedder
        if not hasattr(embedder, "embed_text"):
            raise ValueError(
                f"Text search requires a CLIP or SigLIP embedder. "
                f"This dataset uses {self.pipeline.get_embedder_short_name()}."
            )
        if self.total_index_size == 0:
            print("Warning: All indexes are empty, no matches possible")
            return []
        embedding = embedder.embed_text(text)
        results = self._search_embedding(embedding, top_k)
        # Use embedder-native confidence if available (e.g. SigLIP sigmoid scaling)
        if hasattr(embedder, "text_confidence"):
            for r in results:
                r.confidence = embedder.text_confidence(r.distance)
            results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def get_stats(self) -> dict:
        stats = self.db.get_stats()
        stats["index_size"] = self.total_index_size
        return stats

    @staticmethod
    def get_combined_stats(stats_list: list[dict]):
        from collections import Counter
        ret = {
            "total_detections": 0,
            "unique_characters": 0,
            "unique_posts": 0,
            "top_characters": {},
            "segmentor_breakdown": {},
            "preprocessing_breakdown": {},
            "git_version_breakdown": {},
            "source_breakdown": {},
            "index_size": 0,
            "num_datasets": 0,
        }
        for k, v in ret.items():
            if isinstance(v, dict):
                cnt = Counter(v)
                for stats in stats_list:
                    val = stats.get(k, {})
                    if isinstance(val, list):
                        val = dict(val)
                    cnt.update(val)
                ret[k] = dict(cnt)
            else:
                ret[k] = sum([stats.get(k, 0) for stats in stats_list])
        top = ret["top_characters"]
        ret["num_datasets"] = len(stats_list)
        ret["top_characters"] = sorted(top.items(), key=lambda x: x[1], reverse=True)[:10]
        return ret


# ---------------------------------------------------------------------------
# Multi-dataset merge
# ---------------------------------------------------------------------------

def merge_multi_dataset_results(
    all_results: list[list[SegmentResults]],
    top_k: int = Config.DEFAULT_TOP_K,
    strategy: Optional[MergeStrategy] = None,
) -> list[SegmentResults]:
    """Merge SegmentResults from multiple datasets using the given strategy."""
    return (strategy or _DEFAULT_MERGE_STRATEGY).merge(all_results, top_k)


def merge_with_preferred(
    all_results: list[list[SegmentResults]],
    dataset_names: list[str],
    preferred_datasets: set[str],
    num_preferred: int,
    top_k: int = Config.DEFAULT_TOP_K,
    strategy: Optional[MergeStrategy] = None,
) -> list[SegmentResults]:
    """Merge results, reserving the first num_preferred slots for preferred_datasets.

    Falls back to regular merge if preferred_datasets is empty or none match.
    """
    if not preferred_datasets or num_preferred <= 0:
        return merge_multi_dataset_results(all_results, top_k, strategy)
    pref_idx = [i for i, n in enumerate(dataset_names) if n in preferred_datasets]
    if not pref_idx:
        return merge_multi_dataset_results(all_results, top_k, strategy)

    pref_merged = merge_multi_dataset_results([all_results[i] for i in pref_idx], num_preferred, strategy)
    all_merged = merge_multi_dataset_results(all_results, top_k, strategy)

    base = next((r for r in all_results if r), None)
    if base is None:
        return []
    result = []
    for seg_idx, base_seg in enumerate(base):
        pref_matches = pref_merged[seg_idx].matches if seg_idx < len(pref_merged) else []
        all_matches = all_merged[seg_idx].matches if seg_idx < len(all_merged) else []
        seen = {(m.character_name or "").lower() for m in pref_matches}
        combined = list(pref_matches)
        for m in all_matches:
            if len(combined) >= top_k:
                break
            if (m.character_name or "").lower() not in seen:
                combined.append(m)
                seen.add((m.character_name or "").lower())
        result.append(SegmentResults(
            segment_index=base_seg.segment_index,
            segment_bbox=base_seg.segment_bbox,
            segment_confidence=base_seg.segment_confidence,
            matches=combined,
        ))
    return result


# ---------------------------------------------------------------------------
# Embedder detection / construction
# ---------------------------------------------------------------------------

def detect_embedder(db_path: str, default: str = Config.DEFAULT_EMBEDDER):
    """Detects the short name of the embedder from the dataset."""
    emb = Database.read_metadata_lightweight(db_path, Config.METADATA_KEY_EMBEDDER)
    if emb:
        print(f"Auto-detected embedder for {db_path}: {emb} ({SHORT_NAME_TO_CLI.get(emb, emb)})", file=sys.stderr)
        return emb
    print(f"WARN: Could not find embedder for {db_path}, using default: {default}", file=sys.stderr)
    return default


def build_embedder_for_name(short_name: str, device: Optional[str] = None):
    """Instantiate an embedder from its short name (e.g. 'siglip', 'dv2b', 'clip')."""

    cli_name = SHORT_NAME_TO_CLI.get(short_name, short_name)
    if cli_name in ("siglip", "google/siglip-base-patch16-224"):
        from sam3_fursearch.models.embedder import SigLIPEmbedder
        return SigLIPEmbedder(device=device)
    if cli_name == "clip":
        from sam3_fursearch.models.embedder import CLIPEmbedder
        return CLIPEmbedder(device=device)
    if cli_name == "dinov2-base":
        from sam3_fursearch.models.embedder import DINOv2Embedder
        return DINOv2Embedder(model_name=Config.DINOV2_MODEL, device=device)
    if cli_name == "dinov2-large":
        from sam3_fursearch.models.embedder import DINOv2Embedder
        return DINOv2Embedder(model_name=Config.DINOV2_LARGE_MODEL, device=device)
    if cli_name == "dinov2-base+colorhist":
        from sam3_fursearch.models.embedder import DINOv2Embedder, ColorHistogramEmbedder
        return ColorHistogramEmbedder(DINOv2Embedder(device=device))
    # Fallback: default SigLIP
    from sam3_fursearch.models.embedder import SigLIPEmbedder
    return SigLIPEmbedder(device=device)
