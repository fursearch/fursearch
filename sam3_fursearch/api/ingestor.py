import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

from sam3_fursearch.api.identifier import _save_debug_crop, build_embedder_for_name, detect_embedder
from sam3_fursearch.config import Config
from sam3_fursearch.models.preprocessor import IsolationConfig
from sam3_fursearch.pipeline.processor import CacheKey, CachedProcessingPipeline, SHORT_NAME_TO_CLI
from sam3_fursearch.storage.database import Database, Detection
from sam3_fursearch.storage.mask_storage import MaskStorage
from sam3_fursearch.storage.vector_index import VectorIndex


class FursuitIngestor:
    def __init__(
        self,
        db_path: str = Config.DB_PATH,
        index_path: str = Config.INDEX_PATH,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
        mask_storage=MaskStorage(),
        embedder=None,
        preprocessors: Optional[list] = None,
    ):
        if not embedder:
            emb = detect_embedder(db_path=db_path)
            embedder = build_embedder_for_name(short_name = emb, device = device)
        self.embedder = embedder
        self.db = Database(db_path)
        embedding_dim = embedder.embedding_dim if embedder else Config.EMBEDDING_DIM
        self.index = VectorIndex(index_path, embedding_dim=embedding_dim)
        self._sync_index_and_db()
        self.pipeline = CachedProcessingPipeline(
            device=device,
            isolation_config=isolation_config,
            segmentor_model_name=segmentor_model_name,
            segmentor_concept=segmentor_concept,
            mask_storage=mask_storage,
            embedder=embedder,
            preprocessors=preprocessors,
        )
        # Store/validate embedder metadata
        self._validate_or_store_embedder()

        self.fallback_pipeline = CachedProcessingPipeline(
            device=device,
            isolation_config=isolation_config,
            segmentor_model_name="full",
            segmentor_concept="",
            mask_storage=mask_storage,
            embedder=embedder,
            preprocessors=preprocessors,
        )

    def _validate_or_store_embedder(self):
        """Validate embedder matches dataset, or store it on first use."""
        current = self.pipeline.get_embedder_short_name()
        embedder_dim = self.pipeline.embedder.embedding_dim
        stored = self.db.get_metadata(Config.METADATA_KEY_EMBEDDER)
        if stored is not None:
            if stored != current:
                raise ValueError(
                    f"Dataset was built with embedder '{stored}', "
                    f"but current embedder is '{current}'. "
                    f"Use --embedder {stored} to match."
                )
        elif self.index.size > 0 and self.index.embedding_dim != embedder_dim:
            raise ValueError(
                f"Index has {self.index.embedding_dim}D embeddings but current embedder "
                f"'{current}' produces {embedder_dim}D. "
                f"Use --embedder to select the matching embedder."
            )
        else:
            self.db.set_metadata(Config.METADATA_KEY_EMBEDDER, current)

    def _sync_index_and_db(self):
        """Ensure FAISS index and database are in sync (crash recovery)."""
        max_valid_id = self.index.size - 1
        max_db_id = self.db.get_next_embedding_id() - 1
        if max_db_id > max_valid_id:
            deleted = self.db.delete_orphaned_detections(max_valid_id)
            if deleted > 0:
                print(f"Sync: deleted {deleted} orphaned detections (embedding_id > {max_valid_id})")

    def regenerate_mask_cache(self, image_paths: list[str], source: str) -> int:
        total = len(image_paths)
        regenerated_count = 0

        for i, img_path in enumerate(image_paths):
            post_id = self._extract_post_id(img_path)
            filename = os.path.basename(img_path)

            try:
                image = self._load_image(img_path)
            except Exception as e:
                print(f"[{i+1}/{total}] Failed to load {filename}: {e}")
                continue

            cache_key = CacheKey(post_id=post_id, source=source)
            segmentations, mask_reused = self.pipeline._segment(image, cache_key=cache_key)

            if not mask_reused:
                regenerated_count += 1
                print(f"[{i+1}/{total}] Regenerated mask cache for {filename} ({len(segmentations)} segments)")
            else:
                print(f"[{i+1}/{total}] No segments found for {filename}, nothing to cache")

        print(f"Mask cache regeneration complete: {regenerated_count} posts processed")
        return regenerated_count

    def add_images(
        self,
        character_names: list[str],
        image_paths: list[str],
        source: str,
        save_crops: bool = False,
        uploaded_by: Optional[str] = None,
        add_full_image: bool = True,
        batch_size: int = 100,
        skip_non_fursuit: bool = False,
        classify_threshold: float = Config.DEFAULT_CLASSIFY_THRESHOLD,
        post_ids: Optional[list[str]] = None,
    ) -> int:

        assert len(character_names) == len(image_paths)
        character_names = [name.lower().replace(" ", "_") for name in character_names]

        if post_ids is not None:
            assert len(post_ids) == len(image_paths)
        else:
            post_ids = [self._extract_post_id(p) for p in image_paths]

        posts_need_full = self.db.get_posts_needing_update(post_ids, self.fallback_pipeline.build_preprocessing_info(), source)
        posts_need_seg = self.db.get_posts_needing_update(post_ids, self.pipeline.build_preprocessing_info(), source)
        posts_to_process = posts_need_seg if not add_full_image else posts_need_full | posts_need_seg

        print(f"Processing {len(posts_to_process)} posts ({len(posts_need_full)} need full, {len(posts_need_seg)} need seg)")
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            return 0

        classifier = None
        if skip_non_fursuit:
            from sam3_fursearch.models.classifier import ImageClassifier
            classifier = ImageClassifier(device=self.pipeline.device)
            print(f"Using classifier to skip non-fursuit images (threshold: {classify_threshold})")

        total = len(filtered_indices)
        added_count = 0
        skipped_count = 0
        masks_reused_count = 0
        masks_generated_count = 0
        pending_embeddings: list[np.ndarray] = []
        pending_detections: list[Detection] = []

        def new_detection(post_id, character_name, bbox, confidence, segmentor_model, filename, preproc_info):
            return Detection(
                id=None, post_id=post_id, character_name=character_name, embedding_id=-1,
                bbox_x=bbox[0], bbox_y=bbox[1], bbox_width=bbox[2], bbox_height=bbox[3],
                confidence=confidence, segmentor_model=segmentor_model,
                source=source, uploaded_by=uploaded_by, source_filename=filename,
                preprocessing_info=preproc_info,
            )

        def flush_batch():
            """Commit DB then save FAISS. If interrupted, _sync_index_and_db cleans up orphans."""
            nonlocal added_count
            if not pending_embeddings:
                return
            print(f"  Saving batch of {len(pending_detections)} embeddings to database and index...")
            start_id = self.index.add(np.vstack(pending_embeddings).astype(np.float32))
            for i, detection in enumerate(pending_detections):
                detection.embedding_id = start_id + i
            self.db.add_detections_batch(pending_detections)
            self.index.save(backup=True)
            added_count += len(pending_detections)
            print(f"  Batch saved: {len(pending_detections)} embeddings (index: {self.index.size})")
            pending_embeddings.clear()
            pending_detections.clear()

        for i, idx in enumerate(filtered_indices):
            character_name = character_names[idx]
            img_path = image_paths[idx]
            post_id = self._extract_post_id(img_path)
            filename = os.path.basename(img_path)

            try:
                image = self._load_image(img_path)
            except Exception as e:
                print(f"[{i+1}/{total}] Failed to load {filename}: {e}")
                continue

            if classifier and not classifier.is_fursuit(image, threshold=classify_threshold):
                skipped_count += 1
                print(f"[{i+1}/{total}] Skipped {filename} (not fursuit)")
                continue

            cache_key = CacheKey(post_id=post_id, source=source)
            proc_results = self.pipeline.process(image, cache_key=cache_key)

            # Only keep the largest segment bbox
            if len(proc_results) > 1:
                proc_results = [max(proc_results, key=lambda r: r.segmentation.bbox[2] * r.segmentation.bbox[3])]

            if proc_results and proc_results[0].mask_reused:
                mask_reused = True
                masks_reused_count += len(proc_results)
            else:
                mask_reused = False
                masks_generated_count += len(proc_results)

            full_msg = ""
            if not proc_results:
                print(f"[{i+1}/{total}] No segments found for {filename}, adding full image as fallback")
            if not proc_results or add_full_image:
                proc_results += self.fallback_pipeline.process(image)
                full_msg = " +full"

            for j, result in enumerate(proc_results):
                if save_crops and result.isolated_crop:
                    seg_name = f"{post_id}_seg_{j}" if result.segmentor_model != "full" else f"{post_id}_full"
                    _save_debug_crop(result.isolated_crop, seg_name, source=source)
                pending_embeddings.append(result.embedding.reshape(1, -1))
                pending_detections.append(new_detection(
                    post_id, character_name, result.segmentation.bbox,
                    result.segmentation.confidence, result.segmentor_model,
                    filename, result.preprocessing_info))

            mask_msg = " (masks reused)" if mask_reused else ""
            print(f"[{i+1}/{total}] {character_name}: {len(proc_results)} segments{mask_msg}{full_msg}")

            if len(pending_embeddings) >= batch_size:
                flush_batch()

        flush_batch()

        skip_msg = f", {skipped_count} skipped (not fursuit)" if skipped_count else ""
        mask_msg = f", masks: {masks_reused_count} reused/{masks_generated_count} generated" if masks_reused_count or masks_generated_count else ""
        print(f"Ingestion complete: {added_count} embeddings added{skip_msg}{mask_msg} (index: {self.index.size})")
        return added_count

    def _load_image(self, img_path: str) -> Image.Image:
        img_path = str(img_path)
        if img_path.startswith(('http://', 'https://')):
            response = requests.get(img_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            if not Path(img_path).exists():
                raise FileNotFoundError()
            img = Image.open(img_path)
        if img is None:
            raise ValueError()
        return img

    @staticmethod
    def _extract_post_id(img_path: str) -> str:
        basename = os.path.basename(img_path)
        return os.path.splitext(basename)[0]

    def get_stats(self) -> dict:
        db_stats = self.db.get_stats()
        db_stats["index_size"] = self.index.size
        return db_stats
