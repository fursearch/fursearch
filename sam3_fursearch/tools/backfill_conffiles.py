"""Backfill conffiles for mask directories using confidence values from the database.

For posts where masks exist on disk but conffiles are missing, this tool reads
the segment confidence from the database and writes the corresponding .conffile.

Usage:
    python tools/backfill_conffiles.py                    # default dataset (fursearch)
    python tools/backfill_conffiles.py --dataset validation
    python tools/backfill_conffiles.py --dry-run           # preview without writing
"""

import argparse
import re
import struct
import sqlite3
from collections import defaultdict
from pathlib import Path


def _normalize_concept(concept: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '_', concept).strip('_') or "default"


def parse_preprocessing_info(preproc: str) -> dict:
    """Extract seg model and concept from preprocessing_info string."""
    result = {}
    for part in preproc.split("|"):
        if part.startswith("seg:"):
            result["model"] = part[4:]
        elif part.startswith("con:"):
            result["concept"] = part[4:]
    return result


def main():
    parser = argparse.ArgumentParser(description="Backfill conffiles from database confidence values")
    parser.add_argument("--dataset", "-ds", default="fursearch", help="Dataset name (default: fursearch)")
    parser.add_argument("--masks-dir", help="Override masks directory (default: fursearch_masks)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    base_dir = Path('.').resolve() #.parent
    db_path = base_dir / f"{args.dataset}.db"
    masks_dir = Path(args.masks_dir) if args.masks_dir else base_dir / f"fursearch_masks"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    if not masks_dir.exists():
        print(f"Masks directory not found: {masks_dir}")
        return

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # Get all SAM3 detections grouped by (post_id, source, preprocessing_info), ordered by embedding_id
    c.execute("""
        SELECT post_id, source, segmentor_model, preprocessing_info, confidence
        FROM detections
        WHERE segmentor_model != 'full'
          AND confidence IS NOT NULL
        ORDER BY post_id, source, preprocessing_info, embedding_id
    """)

    # Group detections: (post_id, source, model, concept) -> [confidence, ...]
    groups = defaultdict(list)
    for post_id, source, model, preproc, confidence in c.fetchall():
        parsed = parse_preprocessing_info(preproc or "")
        concept = parsed.get("concept", "")
        model_name = parsed.get("model", model)
        key = (post_id, source or "unknown", model_name, concept)
        groups[key].append(confidence)

    conn.close()
    print(f'Found {len(groups)} confidence values')

    written = 0
    skipped_existing = 0
    skipped_no_masks = 0
    mismatched = 0
    i = -1

    for (post_id, source, model, concept), confidences in groups.items():
        i += 1
        # print(f'[{i}/{len(groups)}]')
        normalized_concept = _normalize_concept(concept)
        mask_dir = masks_dir / source / model / normalized_concept
        conffile = mask_dir / f"{post_id}.conffile"

        # Skip if conffile already exists
        if conffile.exists():
            skipped_existing += 1
            continue

        # # Check that masks actually exist on disk
        # masks = sorted(mask_dir.glob(f"{post_id}_seg_*.png"), key=lambda p: int(p.stem.split("_seg_")[-1]))
        # if not masks:
        #     skipped_no_masks += 1
        #     continue

        # # Validate segment count matches
        # if len(masks) != len(confidences):
        #     mismatched += 1
        #     print(f"WARN: mismatch for {post_id} in {mask_dir}: {len(masks)} masks vs {len(confidences)} DB entries")
        #     continue

        if args.dry_run:
            print(f"Would write: {conffile} ({len(confidences)} segments: {[f'{c:.4f}' for c in confidences]})")
            written += 1
            continue

        mask_dir.mkdir(parents=True, exist_ok=True)
        with open(conffile, 'wb') as f:
            f.write(struct.pack(f'{len(confidences)}d', *confidences))
        written += 1

    action = "Would write" if args.dry_run else "Written"
    print(f"\n{action}: {written} conffiles")
    print(f"Skipped (already exist): {skipped_existing}")
    print(f"Skipped (no masks on disk): {skipped_no_masks}")
    print(f"Skipped (count mismatch): {mismatched}")
    print(f"Total DB groups processed: {len(groups)}")


if __name__ == "__main__":
    main()
