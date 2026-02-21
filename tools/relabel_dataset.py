#!/usr/bin/env python3
"""Relabel a dataset by re-identifying every detection using avg_embedding.

Creates a copy of the database with character_name updated to the top-1
identification result. The FAISS index is read-only (shared, not copied).

Usage:
    # Dry run (default) - prints proposed changes
    python tools/relabel_dataset.py

    # Actually write changes
    python tools/relabel_dataset.py --apply

    # Custom dataset
    python tools/relabel_dataset.py --dataset fursearch --output fursearch_relabeled
"""

import argparse
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sam3_fursearch.config import Config
from sam3_fursearch.api.cli import _open_dataset, _get_dataset_paths
from sam3_fursearch.storage.database import Database, Detection
from sam3_fursearch.storage.vector_index import VectorIndex


def load_dataset(db_path: str):
    # Load all detections
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, embedding_id, character_name FROM detections ORDER BY id"
    ).fetchall()
    conn.close()
    print(f"Loaded {len(rows)} detections from {db_path}")
    return rows


def relabel(dataset: str, output_db_path: str, top_k: int, apply: bool):
    db, index = _open_dataset(dataset)
    total = index.size
    if total == 0:
        print("Index is empty, nothing to relabel.")
        return

    db_path = db.db_path
    rows = load_dataset(db_path)

    # Build embedding_id -> character_name lookup for exclusion
    emb_id_to_char: dict[int, str] = {}
    for row_id, emb_id, char_name in rows:
        emb_id_to_char[emb_id] = (char_name or "").lower()

    # For each detection, find top-1 match (excluding self)
    fetch_k = min(top_k * 6 + 1, total)  # +1 for self
    updates: list[tuple[str, int]] = []  # (new_name, detection_id)
    changed = 0
    unchanged = 0
    no_match = 0
    change_counts: Counter = Counter()

    for i, (row_id, emb_id, old_name) in enumerate(rows):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(rows)}...")

        query = index.reconstruct(emb_id).reshape(1, -1).astype(np.float32)
        distances, indices = index.search(query, fetch_k)

        # Group by character, excluding self embedding
        char_indices: dict[str, list[int]] = {}
        char_name_cased: dict[str, str] = {}
        for idx in indices[0]:
            if idx == -1 or idx == emb_id:
                continue
            char = emb_id_to_char.get(int(idx))
            if char is None:
                continue
            char_indices.setdefault(char, []).append(int(idx))
            # Keep first cased version we see
            if char not in char_name_cased:
                det = db.get_detection_by_embedding_id(int(idx))
                if det:
                    char_name_cased[char] = det.character_name or ""

        if not char_indices:
            no_match += 1
            continue

        # Average embeddings per character, pick closest
        best_char = None
        best_dist = float("inf")
        for key, faiss_ids in char_indices.items():
            embs = np.stack([index.reconstruct(i) for i in faiss_ids])
            avg = embs.mean(axis=0, keepdims=True).astype(np.float32)
            dist = float(np.sum((query - avg) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_char = key

        new_name = char_name_cased.get(best_char, best_char)
        old_lower = (old_name or "").lower()

        if old_lower != best_char:
            changed += 1
            change_counts[(old_name or "(none)", new_name)] += 1
            updates.append((new_name, row_id))
        else:
            unchanged += 1

    print(f"\nResults: {changed} changed, {unchanged} unchanged, {no_match} no match")

    if change_counts:
        print(f"\nTop relabeling changes:")
        for (old, new), count in change_counts.most_common(30):
            print(f"  {old} -> {new}  ({count}x)")

    if not apply:
        print(f"\nDry run complete. Use --apply to write {output_db_path}")
        return

    # Copy DB and apply updates
    shutil.copy2(db_path, output_db_path)
    out_conn = sqlite3.connect(output_db_path)
    out_conn.executemany(
        "UPDATE detections SET character_name = ? WHERE id = ?", updates
    )
    out_conn.commit()
    out_conn.close()
    print(f"\nWrote relabeled database to {output_db_path}")


def save_average(dataset: str, out_dataset: str, apply: bool):
    db, index = _open_dataset(dataset)
    out_db_path, out_db_index = _get_dataset_paths(out_dataset)
    out_index = VectorIndex(out_db_index)
    out_index.reset()
    out_db = Database(out_db_path)

    char_names = db.get_all_character_names()
    avg_dets = []
    for i, char in enumerate(char_names):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(char_names)}...")
        dets = db.get_detections_by_character(char)
        bx, by, bh, bw = 0,0,0,0
        conf = 0
        embs = []
        for det in dets:
            bx, by, bh, bw = bx + det.bbox_x, by + det.bbox_y, bh + det.bbox_height, bw + det.bbox_width
            query = index.reconstruct(det.embedding_id).reshape(1, -1).astype(np.float32)
            embs.append(query)
            conf += det.confidence
        avgemb = np.average(embs, axis=0).astype(np.float32)
        avgconf = conf / len(dets)
        bx, by, bh, bw = [t // len(dets) for t in [bx, by, bh, bw]]
        char_name = (char or "").lower()
        out_index.add(avgemb)
        avg_dets.append(Detection(None, str(i), char_name, i, bx, by, bw, bh, avgconf))

    if not apply:
        print(f"\nDry run complete. Use --apply to write {out_db_path}")
        return

    out_db.add_detections_batch(avg_dets)
    out_db.close()
    out_index.save()
    print(out_db.get_stats())
    print(f"\nWrote relabeled database to {out_db_path}")


def main():
    parser = argparse.ArgumentParser(description="Relabel dataset using avg_embedding identification")
    parser.add_argument("--dataset", "-d", default="fursearch", help="Source dataset name")
    parser.add_argument("--output", "-o", default=None, help="Output dataset name (default: {dataset}_relabeled)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k neighbors per character for averaging")
    parser.add_argument("--average", action="store_true", help="Squeeze all embeddings for characters into one")
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry run)")
    args = parser.parse_args()

    base = Path(Config.BASE_DIR)
    output_name = args.output or f"{args.dataset}_relabeled"
    output_db_path = str(base / f"{output_name}.db")

    if args.average:
        save_average(args.dataset, output_name, args.apply)
        return

    relabel(args.dataset, output_db_path, args.top_k, args.apply)


if __name__ == "__main__":
    main()
