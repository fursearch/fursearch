#!/usr/bin/env python3
"""Hotfix: Create .noseg marker files for posts where SAM3 found no segments.

Detects fallback detections from the database (segmentor_model='full' but
preprocessing_info contains 'seg:sam3') and creates .noseg markers so future
ingests skip SAM3 for these posts.

Usage:
    python tools/hotfix_create_noseg_markers.py [--dry-run] [--db fursearch.db]
"""

import argparse
import sqlite3
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_fursearch.config import Config
from sam3_fursearch.storage.mask_storage import MaskStorage


def get_fallback_posts(db_path: str) -> list[tuple[str, str, str, str]]:
    """Find posts where SAM3 was used but fell back to full image.

    Returns list of (post_id, source, model, concept) tuples.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Find detections where:
    # - segmentor_model is 'full' (fallback was used)
    # - preprocessing_info contains 'seg:sam3' (SAM3 was the intended segmentor)
    c.execute("""
        SELECT DISTINCT post_id, source, preprocessing_info
        FROM detections
        WHERE segmentor_model = 'full'
          AND preprocessing_info LIKE '%seg:sam3%'
    """)

    results = []
    for post_id, source, preproc in c.fetchall():
        # Parse concept from preprocessing_info: "v2|seg:sam3|con:fursuiter head|..."
        concept = ""
        for part in preproc.split("|"):
            if part.startswith("con:"):
                concept = part[4:].replace(".", " ")  # Restore spaces
                break
        results.append((post_id, source or "unknown", "sam3", concept))

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Create .noseg markers for fallback detections")
    parser.add_argument("--db", default=Config.DB_PATH, help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without creating files")
    parser.add_argument("--masks-dir", default=Config.MASKS_DIR, help="Masks directory")
    args = parser.parse_args()

    print(f"Scanning database: {args.db}")
    fallback_posts = get_fallback_posts(args.db)
    print(f"Found {len(fallback_posts)} posts with SAM3 fallback to full image")

    if not fallback_posts:
        print("Nothing to do.")
        return

    mask_storage = MaskStorage(base_dir=args.masks_dir)

    created = 0
    skipped = 0

    for post_id, source, model, concept in fallback_posts:
        if mask_storage.has_no_segments_marker(post_id, source, model, concept):
            skipped += 1
            continue

        if args.dry_run:
            marker_path = mask_storage.get_mask_dir(source, model, concept) / f"{post_id}.noseg"
            print(f"Would create: {marker_path}")
        else:
            mask_storage.save_no_segments_marker(post_id, source, model, concept)
        created += 1

    if args.dry_run:
        print(f"\nDry run: would create {created} markers ({skipped} already exist)")
    else:
        print(f"\nCreated {created} .noseg markers ({skipped} already existed)")


if __name__ == "__main__":
    main()
