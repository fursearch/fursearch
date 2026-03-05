"""Tests for merge strategies and related logic in identifier.py."""

import numpy as np
import pytest

from sam3_fursearch.api.identifier import (
    AvgEmbeddingStrategy,
    ConfidenceMerge,
    IdentificationResult,
    SegmentResults,
    merge_multi_dataset_results,
)
from sam3_fursearch.storage.database import Database, Detection
from sam3_fursearch.storage.vector_index import VectorIndex

DIM = 4  # small dimension for fast tests


# ---------------------------------------------------------------------------
# E2E fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    index = VectorIndex(str(tmp_path / "test.index"), embedding_dim=DIM)
    db = Database(str(tmp_path / "test.db"))
    return index, db


def add_embedding(index, db, character_name, embedding, segmentor_model="sam3", source="test"):
    emb = np.array(embedding, dtype=np.float32)
    emb_id = index.add(emb)
    db.add_detection(Detection(
        id=None,
        post_id=f"post_{emb_id}",
        character_name=character_name,
        embedding_id=emb_id,
        bbox_x=0, bbox_y=0, bbox_width=100, bbox_height=100,
        confidence=0.9,
        segmentor_model=segmentor_model,
        source=source,
    ))
    return emb_id


# ---------------------------------------------------------------------------
# AvgEmbeddingStrategy — end-to-end
# ---------------------------------------------------------------------------

class TestAvgEmbeddingStrategyE2E:

    @pytest.fixture(autouse=True)
    def setup(self, store):
        self.index, self.db = store
        self.strategy = AvgEmbeddingStrategy()

    def search(self, query, top_k=5):
        q = np.array(query, dtype=np.float32).reshape(1, -1)
        return self.strategy.search(self.index, self.db, q, top_k)

    def test_empty_index_returns_empty(self):
        assert self.search([1, 0, 0, 0]) == []

    def test_returns_nearest_character(self):
        add_embedding(self.index, self.db, "alice", [1, 0, 0, 0])
        add_embedding(self.index, self.db, "bob",   [0, 1, 0, 0])
        results = self.search([1, 0.1, 0, 0])
        assert results[0].character_name == "alice"

    def test_result_has_correct_metadata(self):
        add_embedding(self.index, self.db, "alice", [1, 0, 0, 0], segmentor_model="sam3", source="furtrack")
        results = self.search([1, 0, 0, 0])
        r = results[0]
        assert r.character_name == "alice"
        assert r.segmentor_model == "sam3"
        assert r.source == "furtrack"
        assert r.confidence > 0
        assert r.distance >= 0

    def test_multiple_embeddings_per_character_deduped(self):
        add_embedding(self.index, self.db, "alice", [1.0, 0.0, 0, 0])
        add_embedding(self.index, self.db, "alice", [0.9, 0.1, 0, 0])
        add_embedding(self.index, self.db, "alice", [0.8, 0.2, 0, 0])
        add_embedding(self.index, self.db, "bob",   [0.0, 1.0, 0, 0])
        results = self.search([1, 0, 0, 0], top_k=5)
        names = [r.character_name for r in results]
        assert names.count("alice") == 1
        assert names.count("bob") == 1

    def test_averaging_changes_ranking(self):
        # alice has two embeddings; their average centroid is farther from the
        # query than bob's single point — so averaging makes bob rank above alice,
        # whereas simple nearest-neighbour dedup would rank alice first (emb at 0.02).
        #
        # alice emb1=[0.9,0.1,0,0]  sq_dist to query = 0.02
        # alice emb2=[0.7,0.5,0,0]  sq_dist to query = 0.34
        # alice avg =[0.8,0.3,0,0]  sq_dist to query = 0.04+0.09 = 0.13
        # bob        =[0.75,0, 0,0]  sq_dist to query = 0.0625
        query = [1, 0, 0, 0]
        add_embedding(self.index, self.db, "alice", [0.9, 0.1, 0, 0])
        add_embedding(self.index, self.db, "alice", [0.7, 0.5, 0, 0])
        add_embedding(self.index, self.db, "bob",   [0.75, 0, 0, 0])
        results = self.search(query, top_k=2)
        assert results[0].character_name == "bob"
        assert results[1].character_name == "alice"

    def test_distance_filter_excludes_far_embeddings(self):
        # sq_dist([1,0,0,0], [0,0,0,1]) = 2.0 > FAISS_MAX_DISTANCE (0.7)
        add_embedding(self.index, self.db, "close", [1, 0, 0, 0])
        add_embedding(self.index, self.db, "far",   [0, 0, 0, 1])
        results = self.search([1, 0, 0, 0], top_k=5)
        names = {r.character_name for r in results}
        assert "close" in names
        assert "far" not in names

    def test_confidence_inversely_proportional_to_distance(self):
        add_embedding(self.index, self.db, "near", [0.99, 0,   0, 0])
        add_embedding(self.index, self.db, "far",  [0.6,  0.5, 0, 0])
        results = self.search([1, 0, 0, 0], top_k=5)
        near = next(r for r in results if r.character_name == "near")
        far  = next(r for r in results if r.character_name == "far")
        assert near.confidence > far.confidence
        assert near.distance < far.distance

    def test_top_k_limits_output(self):
        for i in range(4):
            emb = [0.1] * DIM
            emb[i] = 0.9
            add_embedding(self.index, self.db, f"char{i}", emb)
        results = self.search([1, 0, 0, 0], top_k=2)
        assert len(results) <= 2

    def test_different_segmentors_same_character_deduped(self):
        # Same character name with two different segmentor models — appears only once
        add_embedding(self.index, self.db, "alice", [1,   0,   0, 0], segmentor_model="sam3")
        add_embedding(self.index, self.db, "alice", [0.9, 0.1, 0, 0], segmentor_model="full")
        results = self.search([1, 0, 0, 0], top_k=5)
        assert sum(1 for r in results if r.character_name == "alice") == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_match(name: str, confidence: float, distance: float = 0.1) -> IdentificationResult:
    return IdentificationResult(
        character_name=name,
        confidence=confidence,
        distance=distance,
        post_id="post1",
        bbox=(0, 0, 100, 100),
        segmentor_model="sam3",
        source="test",
    )


def make_seg(matches: list[IdentificationResult], seg_idx: int = 0) -> SegmentResults:
    return SegmentResults(
        segment_index=seg_idx,
        segment_bbox=(0, 0, 100, 100),
        segment_confidence=0.9,
        matches=matches,
    )


# ---------------------------------------------------------------------------
# ConfidenceMerge
# ---------------------------------------------------------------------------

class TestConfidenceMerge:
    def setup_method(self):
        self.strategy = ConfidenceMerge()

    def test_empty_input(self):
        assert self.strategy.merge([], top_k=5) == []

    def test_all_empty_datasets(self):
        assert self.strategy.merge([[], []], top_k=5) == []

    def test_single_dataset(self):
        matches = [make_match("alice", 0.9), make_match("bob", 0.7)]
        result = self.strategy.merge([[make_seg(matches)]], top_k=5)
        assert len(result) == 1
        assert [m.character_name for m in result[0].matches] == ["alice", "bob"]

    def test_best_confidence_wins_across_datasets(self):
        # Dataset A has alice at 0.9, Dataset B has alice at 0.95
        ds_a = [make_seg([make_match("alice", 0.9)])]
        ds_b = [make_seg([make_match("alice", 0.95)])]
        result = self.strategy.merge([ds_a, ds_b], top_k=5)
        assert len(result[0].matches) == 1
        assert result[0].matches[0].confidence == 0.95

    def test_characters_merged_across_datasets(self):
        ds_a = [make_seg([make_match("alice", 0.9)])]
        ds_b = [make_seg([make_match("bob", 0.8)])]
        result = self.strategy.merge([ds_a, ds_b], top_k=5)
        names = {m.character_name for m in result[0].matches}
        assert names == {"alice", "bob"}

    def test_top_k_limits_results(self):
        matches = [make_match(f"char{i}", 1.0 - i * 0.1) for i in range(6)]
        result = self.strategy.merge([[make_seg(matches)]], top_k=3)
        assert len(result[0].matches) == 3

    def test_results_sorted_by_confidence_descending(self):
        ds_a = [make_seg([make_match("alice", 0.6)])]
        ds_b = [make_seg([make_match("bob", 0.9), make_match("carol", 0.7)])]
        result = self.strategy.merge([ds_a, ds_b], top_k=5)
        confidences = [m.confidence for m in result[0].matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_case_insensitive_dedup(self):
        # "Alice" and "alice" from different datasets should be deduplicated
        ds_a = [make_seg([make_match("Alice", 0.9)])]
        ds_b = [make_seg([make_match("alice", 0.8)])]
        result = self.strategy.merge([ds_a, ds_b], top_k=5)
        assert len(result[0].matches) == 1
        assert result[0].matches[0].confidence == 0.9

    def test_none_character_name(self):
        matches = [make_match(None, 0.5)]
        result = self.strategy.merge([[make_seg(matches)]], top_k=5)
        assert len(result[0].matches) == 1

    def test_multiple_segments(self):
        seg0 = make_seg([make_match("alice", 0.9)], seg_idx=0)
        seg1 = make_seg([make_match("bob", 0.8)], seg_idx=1)
        result = self.strategy.merge([[seg0, seg1]], top_k=5)
        assert len(result) == 2
        assert result[0].matches[0].character_name == "alice"
        assert result[1].matches[0].character_name == "bob"

    def test_segment_metadata_taken_from_base(self):
        seg = make_seg([make_match("alice", 0.9)], seg_idx=0)
        seg.segment_bbox = (10, 20, 30, 40)
        seg.segment_confidence = 0.77
        result = self.strategy.merge([[seg]], top_k=5)
        assert result[0].segment_bbox == (10, 20, 30, 40)
        assert result[0].segment_confidence == 0.77

    def test_skips_empty_leading_datasets(self):
        # First dataset is empty — base should come from second
        ds_b = [make_seg([make_match("alice", 0.9)])]
        result = self.strategy.merge([[], ds_b], top_k=5)
        assert len(result) == 1
        assert result[0].matches[0].character_name == "alice"


# ---------------------------------------------------------------------------
# Env var wiring
# ---------------------------------------------------------------------------

class TestEnvVarWiring:
    def test_search_default_is_avg_embedding(self, monkeypatch):
        monkeypatch.delenv("SEARCH_STRATEGY", raising=False)
        import importlib
        import sam3_fursearch.config as cfg_mod
        importlib.reload(cfg_mod)
        assert cfg_mod.Config.SEARCH_STRATEGY == "avg_embedding"

    def test_merge_default_is_confidence(self, monkeypatch):
        monkeypatch.delenv("MERGE_STRATEGY", raising=False)
        import importlib
        import sam3_fursearch.config as cfg_mod
        importlib.reload(cfg_mod)
        assert cfg_mod.Config.MERGE_STRATEGY == "confidence"

    def test_merge_multi_dataset_uses_default(self):
        ds_a = [make_seg([make_match("alice", 0.9)])]
        ds_b = [make_seg([make_match("bob", 0.8)])]
        result = merge_multi_dataset_results([ds_a, ds_b], top_k=5)
        names = {m.character_name for m in result[0].matches}
        assert names == {"alice", "bob"}

    def test_explicit_strategy_overrides_default(self):
        ds_a = [make_seg([make_match("alice", 0.9)])]
        result = merge_multi_dataset_results([ds_a], top_k=5, strategy=ConfidenceMerge())
        assert result[0].matches[0].character_name == "alice"
