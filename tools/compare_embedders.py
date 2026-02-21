#!/usr/bin/env python3
"""Compare identification results across datasets with different embedders.

Loads one dataset at a time to avoid GPU OOM on low-VRAM machines.

Usage:
    # Single image, no ground truth
    python tools/compare_embedders.py --image path/to/img.jpg

    # Single image with known character
    python tools/compare_embedders.py --image path/to/img.jpg --character "eon_(gryphon)"

    # Ground truth CSV file (image_path,char1,char2,...)
    python tools/compare_embedders.py --ground-truth tests/ground_truth.csv

    # Directory of test images (character inferred from parent dir name)
    python tools/compare_embedders.py --image-dir path/to/test_images/

    # Specify output file
    python tools/compare_embedders.py --image img.jpg -o results.txt

    # Only compare specific datasets
    python tools/compare_embedders.py --image img.jpg --datasets dino,manual

    # Exclude nfc25-sourced images from results (avoid training/validation overlap)
    python tools/compare_embedders.py --ground-truth tests/ground_truth_nfc25.csv --exclude nfc25
"""

import argparse
import csv
import gc
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from sam3_fursearch.api.identifier import (
    FursuitIdentifier,
    SegmentResults,
    discover_datasets,
    detect_embedder,
    merge_multi_dataset_results,
)
from sam3_fursearch.config import Config
from sam3_fursearch.storage.database import Database
from sam3_fursearch.storage.vector_index import VectorIndex


def load_aliases(alias_path: str | None = None) -> dict[str, dict[str, str]]:
    """Load character alias mapping from CSV.

    Returns {canonical_name_lower: {dataset_name: alias_in_that_dataset}}.
    The CSV has columns: fursearch, dino, validation, ...
    The 'fursearch' column is the canonical name.
    """
    if alias_path is None:
        alias_path = str(Path(__file__).resolve().parent.parent / "tests" / "character_aliases.csv")
    p = Path(alias_path)
    if not p.exists():
        return {}

    aliases: dict[str, dict[str, str]] = {}
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = row.get("fursearch", "").strip()
            if not canonical:
                continue
            ds_map = {}
            for col, val in row.items():
                if col == "fursearch" or not val.strip():
                    continue
                ds_map[col] = val.strip()
            # Also include the canonical name itself under "fursearch"
            ds_map["fursearch"] = canonical
            aliases[canonical.lower()] = ds_map
    return aliases


def resolve_gt_for_dataset(
    gt_name: str,
    ds_name: str,
    aliases: dict[str, dict[str, str]],
) -> str | None:
    """Resolve a ground truth character name to its alias in a specific dataset.

    Returns the dataset-specific name, or the original name if no alias found.
    """
    entry = aliases.get(gt_name.lower())
    if entry and ds_name in entry:
        return entry[ds_name]
    return gt_name


def parse_args():
    parser = argparse.ArgumentParser(description="Compare embedder performance across datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single test image")
    group.add_argument("--image-dir", type=str, help="Directory of test images")
    group.add_argument("--ground-truth", type=str, help="Ground truth CSV: image_path,char1[,char2,...]")
    parser.add_argument("--character", "-c", type=str, help="Ground truth character name (for single image)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results per dataset (default: 10)")
    parser.add_argument("--output", "-o", type=str, default="tools/embedder_comparison_results.txt",
                        help="Output file path")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of dataset names to compare (default: all)")
    parser.add_argument("--exclude", "-e", type=str,
                        help="Comma-separated list of sources to exclude from results (e.g. nfc25,manual)")
    parser.add_argument("--device", type=str, help="Device override (cuda, cpu, mps)")
    parser.add_argument("--aliases", type=str, default="tests/character_aliases.csv",
                        help="Character alias CSV file (default: tests/character_aliases.csv)")
    return parser.parse_args()


def dataset_name(db_path: str) -> str:
    """Extract dataset name from db path: /path/to/foo.db -> foo"""
    return Path(db_path).stem


def load_test_images(args) -> list[tuple[str, list[str] | None]]:
    """Return list of (image_path, ground_truth_characters) tuples.

    Ground truth is a list of character names, one per segment (ordered by
    segment confidence). None if no ground truth is available.
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []

    if args.image:
        gt = [args.character] if args.character else None
        images.append((args.image, gt))
    elif args.ground_truth:
        gt_path = Path(args.ground_truth)
        if not gt_path.exists():
            # Try relative to project root
            gt_path = Path(Config.BASE_DIR) / args.ground_truth
        if not gt_path.exists():
            print(f"Error: ground truth file not found: {args.ground_truth}")
            sys.exit(1)
        with open(gt_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if not row or not row[0].strip():
                    continue
                img_path = row[0].strip()
                # Resolve relative paths against project root
                if not Path(img_path).is_absolute():
                    img_path = str(Path(Config.BASE_DIR) / img_path)
                chars = [c.strip() for c in row[1:] if c.strip()]
                images.append((img_path, chars if chars else None))
        if not images:
            print(f"Error: no entries in ground truth file: {args.ground_truth}")
            sys.exit(1)
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        if not img_dir.is_dir():
            print(f"Error: {args.image_dir} is not a directory")
            sys.exit(1)
        for f in sorted(img_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                gt = [f.parent.name] if f.parent != img_dir else None
                images.append((str(f), gt))
        if not images:
            print(f"Error: no images found in {args.image_dir}")
            sys.exit(1)
    return images


def get_dataset_list(args) -> list[tuple[str, str, str]]:
    """Return list of (name, db_path, index_path) for datasets to compare."""
    all_datasets = discover_datasets()
    if not all_datasets:
        print("Error: no datasets found")
        sys.exit(1)

    filter_names = None
    if args.datasets:
        filter_names = set(args.datasets.split(","))

    result = []
    for db_path, index_path in all_datasets:
        name = dataset_name(db_path)
        if filter_names and name not in filter_names:
            continue
        result.append((name, db_path, index_path))

    if not result:
        print("Error: no matching datasets found")
        sys.exit(1)

    return result


def get_dataset_overview(datasets: list[tuple[str, str, str]]) -> tuple[str, list[dict]]:
    """Format dataset overview table. Lightweight: only reads DB + index, no model loading."""
    lines = ["DATASET OVERVIEW", "=" * 70]
    header = f"{'Dataset':<15} {'Embedder':<12} {'Entries':>10} {'Characters':>12} {'Avg/char':>10}"
    lines.append(header)
    lines.append("-" * 70)

    ds_infos = []
    for name, db_path, index_path in datasets:
        db = Database(db_path)
        stats = db.get_stats()
        all_chars = {c.lower() for c in db.get_all_character_names()}
        emb_name = detect_embedder(db_path, default="unknown")
        try:
            idx = VectorIndex(index_path, embedding_dim=768)
            index_size = idx.size
        except Exception:
            index_size = 0
        total = stats.get("total_detections", 0)
        chars = stats.get("unique_characters", 0)
        avg = total / chars if chars else 0
        lines.append(f"{name:<15} {emb_name:<12} {total:>10,} {chars:>12,} {avg:>10.1f}")
        ds_infos.append({
            "name": name, "embedder": emb_name, "total": total,
            "chars": chars, "all_characters": all_chars,
        })

    lines.append("")
    return "\n".join(lines), ds_infos


def format_segment_results(
    name: str,
    emb_name: str,
    total_entries: int,
    segment_results: list[SegmentResults],
    ground_truth: list[str] | None,
    ds_characters: set[str] | None,
    aliases: dict[str, dict[str, str]],
    top_k: int,
) -> tuple[str, list[dict], str | None]:
    """Format results for one dataset + one image.

    Returns (text, list_of_per_segment_metrics, top1_char_for_segment_0).
    Each metric dict includes 'in_dataset' to indicate if the ground truth
    character exists in this dataset.
    """
    lines = []
    seg_metrics = []
    top1_char = None

    lines.append(f"  Dataset: {name} ({emb_name}, {total_entries:,} entries)")

    if not segment_results:
        lines.append("    (no segments found)")
        return "\n".join(lines), seg_metrics, top1_char

    for seg in segment_results:
        # Determine ground truth for this segment, resolved to dataset-specific alias
        seg_gt_canonical = None
        seg_gt = None
        if ground_truth and seg.segment_index < len(ground_truth):
            seg_gt_canonical = ground_truth[seg.segment_index]
            seg_gt = resolve_gt_for_dataset(seg_gt_canonical, name, aliases)

        in_dataset = True
        if seg_gt and ds_characters is not None:
            in_dataset = seg_gt.lower() in ds_characters

        alias_note = ""
        if seg_gt and seg_gt_canonical and seg_gt.lower() != seg_gt_canonical.lower():
            alias_note = f" (alias: {seg_gt})"

        metrics = {
            "dataset": name,
            "embedder": emb_name,
            "segment_index": seg.segment_index,
            "ground_truth": seg_gt_canonical,
            "ground_truth_resolved": seg_gt,
            "in_dataset": in_dataset,
            "top1_correct": False,
            "topk_correct": False,
            "correct_rank": None,
            "top1_confidence": 0.0,
            "top2_confidence": 0.0,
            "confidence_gap": 0.0,
        }

        if len(segment_results) > 1:
            gt_label = f", gt={seg_gt_canonical}{alias_note}" if seg_gt_canonical else ""
            in_ds_label = "" if in_dataset else " [NOT IN DATASET]"
            lines.append(f"    Segment {seg.segment_index} (bbox={seg.segment_bbox}, conf={seg.segment_confidence:.2f}{gt_label}){in_ds_label}:")
        elif seg_gt_canonical and not in_dataset:
            lines.append(f"    [NOT IN DATASET: {seg_gt_canonical}{alias_note}]")
        elif alias_note:
            lines.append(f"    [gt={seg_gt_canonical}{alias_note}]")

        if not seg.matches:
            lines.append("    (no matches)")
            seg_metrics.append(metrics)
            continue

        for rank, match in enumerate(seg.matches[:top_k], 1):
            marker = ""
            if seg_gt and match.character_name and match.character_name.lower() == seg_gt.lower():
                marker = " <-- CORRECT"
            lines.append(f"    #{rank}: {match.character_name} ({match.confidence:.1%}){marker}")

        if seg.matches:
            top1 = seg.matches[0]
            if seg.segment_index == 0:
                top1_char = top1.character_name
            metrics["top1_confidence"] = top1.confidence
            if len(seg.matches) > 1:
                metrics["top2_confidence"] = seg.matches[1].confidence
                metrics["confidence_gap"] = top1.confidence - seg.matches[1].confidence

            if seg_gt and in_dataset:
                for rank, match in enumerate(seg.matches[:top_k], 1):
                    if match.character_name and match.character_name.lower() == seg_gt.lower():
                        if rank == 1:
                            metrics["top1_correct"] = True
                        metrics["topk_correct"] = True
                        metrics["correct_rank"] = rank
                        break

        seg_metrics.append(metrics)

    return "\n".join(lines), seg_metrics, top1_char


def format_summary(
    all_metrics: list[list[dict]],
    top1_chars: list[dict[str, str | None]],
    test_images: list[tuple[str, list[str] | None]],
    ds_names: list[str],
    top_k: int,
) -> str:
    """Format summary statistics across all images."""
    lines = ["SUMMARY", "=" * 70]

    # Flatten: collect per-segment metrics grouped by dataset
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for image_metrics in all_metrics:
        for m in image_metrics:
            by_dataset[m["dataset"]].append(m)

    has_gt = any(gt for _, gt in test_images)

    if has_gt:
        lines.append("")
        lines.append(f"Accuracy (top-k={top_k}, only counting images where character is in dataset):")
        lines.append(f"{'Dataset':<15} {'Embedder':<12} {'Tested':>7} {'Skip':>5} {'Top-1':>8} {'Top-k':>8} {'Avg Conf':>9} {'Avg Gap':>9}")
        lines.append("-" * 80)

        for ds_name in ds_names:
            metrics = by_dataset.get(ds_name, [])
            if not metrics:
                continue
            # Only count segments that have ground truth AND character is in dataset
            with_gt = [m for m in metrics if m.get("ground_truth")]
            in_ds = [m for m in with_gt if m.get("in_dataset", True)]
            not_in_ds = len(with_gt) - len(in_ds)

            emb = metrics[0]["embedder"]
            n = len(in_ds)
            if n == 0:
                lines.append(f"{ds_name:<15} {emb:<12} {'0':>7} {not_in_ds:>5} {'N/A':>8} {'N/A':>8} {'N/A':>9} {'N/A':>9}")
                continue
            top1_acc = sum(1 for m in in_ds if m["top1_correct"]) / n * 100
            topk_acc = sum(1 for m in in_ds if m["topk_correct"]) / n * 100
            avg_conf = sum(m["top1_confidence"] for m in in_ds) / n
            avg_gap = sum(m["confidence_gap"] for m in in_ds) / n
            lines.append(f"{ds_name:<15} {emb:<12} {n:>7} {not_in_ds:>5} {top1_acc:>7.1f}% {topk_acc:>7.1f}% {avg_conf:>8.1%} {avg_gap:>8.1%}")

        # Head-to-head (only for segment 0, only where both datasets have the character)
        if len(ds_names) >= 2:
            lines.append("")
            lines.append("Head-to-head (segment 0, top-1, only where both datasets have character):")
            for i in range(len(ds_names)):
                for j in range(i + 1, len(ds_names)):
                    a_name, b_name = ds_names[i], ds_names[j]
                    a_wins, b_wins, ties, both_wrong = 0, 0, 0, 0
                    for image_metrics in all_metrics:
                        a_m = next((m for m in image_metrics if m["dataset"] == a_name and m.get("segment_index", 0) == 0), None)
                        b_m = next((m for m in image_metrics if m["dataset"] == b_name and m.get("segment_index", 0) == 0), None)
                        if not a_m or not b_m:
                            continue
                        if not a_m.get("ground_truth") or not b_m.get("ground_truth"):
                            continue
                        if not a_m.get("in_dataset", True) or not b_m.get("in_dataset", True):
                            continue
                        a_ok = a_m["top1_correct"]
                        b_ok = b_m["top1_correct"]
                        if a_ok and b_ok:
                            ties += 1
                        elif a_ok:
                            a_wins += 1
                        elif b_ok:
                            b_wins += 1
                        else:
                            both_wrong += 1
                    total = a_wins + b_wins + ties + both_wrong
                    if total:
                        lines.append(f"  {a_name} vs {b_name}: "
                                     f"{a_name} wins {a_wins}, {b_name} wins {b_wins}, "
                                     f"both correct {ties}, both wrong {both_wrong} (n={total})")
    else:
        lines.append("")
        lines.append("No ground truth provided. Confidence statistics only:")
        lines.append(f"{'Dataset':<15} {'Embedder':<12} {'Avg Top-1 Conf':>15} {'Avg Conf Gap':>15}")
        lines.append("-" * 60)

        for ds_name in ds_names:
            metrics = by_dataset.get(ds_name, [])
            if not metrics:
                continue
            emb = metrics[0]["embedder"]
            n = len(metrics)
            avg_conf = sum(m["top1_confidence"] for m in metrics) / n
            avg_gap = sum(m["confidence_gap"] for m in metrics) / n
            lines.append(f"{ds_name:<15} {emb:<12} {avg_conf:>14.1%} {avg_gap:>14.1%}")

    # Agreement analysis using tracked top-1 characters
    if len(ds_names) >= 2 and top1_chars:
        lines.append("")
        lines.append("Top-1 Agreement (segment 0):")
        lines.append("-" * 40)
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                a, b = ds_names[i], ds_names[j]
                agree = sum(
                    1 for tc in top1_chars
                    if tc.get(a) and tc.get(b) and tc[a].lower() == tc[b].lower()
                )
                disagree = sum(
                    1 for tc in top1_chars
                    if tc.get(a) and tc.get(b) and tc[a].lower() != tc[b].lower()
                )
                total = agree + disagree
                pct = agree / total * 100 if total else 0
                lines.append(f"  {a} vs {b}: {agree}/{total} agree ({pct:.1f}%), {disagree} disagree")

                # if disagree > 0:
                #     lines.append(f"    Disagreements:")
                #     for k, tc in enumerate(top1_chars):
                #         if tc.get(a) and tc.get(b) and tc[a].lower() != tc[b].lower():
                #             img_name = Path(test_images[k][0]).name
                #             gt_chars = test_images[k][1]
                #             gt = gt_chars[0] if gt_chars else "?"
                #             lines.append(
                #                 f"      {img_name} (gt={gt}): {a}={tc[a]}, {b}={tc[b]}"
                #             )

    lines.append("")
    return "\n".join(lines)


def filter_excluded_sources(
    segment_results: list[SegmentResults],
    exclude_sources: set[str],
) -> list[SegmentResults]:
    """Remove matches whose source is in the exclusion set."""
    for seg in segment_results:
        seg.matches = [m for m in seg.matches if m.source not in exclude_sources]
    return segment_results


def free_gpu_memory():
    """Release GPU memory between dataset loads."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    args = parse_args()

    exclude_sources: set[str] = set()
    if args.exclude:
        exclude_sources = {s.strip() for s in args.exclude.split(",")}

    test_images = load_test_images(args)
    print(f"\nLoaded {len(test_images)} test image(s)")
    if exclude_sources:
        print(f"Excluding sources: {', '.join(sorted(exclude_sources))}")

    # Load character aliases
    alias_path = Config.get_absolute_path(args.aliases) if args.aliases else None
    aliases = load_aliases(alias_path)
    if aliases:
        print(f"Loaded {len(aliases)} character alias entries")

    datasets = get_dataset_list(args)
    ds_names = [name for name, _, _ in datasets]
    print(f"Found {len(datasets)} dataset(s): {', '.join(ds_names)}\n")

    output_lines = []
    output_lines.append("EMBEDDER COMPARISON ANALYSIS")
    output_lines.append("=" * 70)
    output_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Test images: {len(test_images)}")
    output_lines.append(f"Datasets: {', '.join(ds_names)}")
    if exclude_sources:
        output_lines.append(f"Excluded sources: {', '.join(sorted(exclude_sources))}")
    output_lines.append("")

    # Dataset overview (lightweight, no model loading)
    overview_text, ds_infos = get_dataset_overview(datasets)
    output_lines.append(overview_text)

    # Build per-dataset character sets for existence checking
    ds_characters: dict[str, set[str]] = {}
    for info in ds_infos:
        ds_characters[info["name"]] = info["all_characters"]

    # Pre-load all test images
    loaded_images: list[tuple[Image.Image | None, str, list[str] | None]] = []
    for img_path, ground_truth in test_images:
        try:
            image = Image.open(img_path).convert("RGB")
            loaded_images.append((image, img_path, ground_truth))
        except Exception as e:
            print(f"  WARNING: Could not load {img_path}: {e}")
            loaded_images.append((None, img_path, ground_truth))

    # Results storage: per-image, per-dataset
    # image_results[img_idx][ds_name] = (text, seg_metrics, top1_char)
    image_results: list[dict[str, tuple[str, list[dict], str | None]]] = [
        {} for _ in loaded_images
    ]
    # Raw segment results for RRF merge
    image_segments: list[dict[str, list[SegmentResults]]] = [
        {} for _ in loaded_images
    ]

    # Process one dataset at a time to avoid GPU OOM
    for ds_idx, (name, db_path, index_path) in enumerate(datasets):
        emb_name = ds_infos[ds_idx]["embedder"]
        total_entries = ds_infos[ds_idx]["total"]
        chars = ds_characters.get(name, set())

        print(f"\n--- Loading dataset: {name} ({emb_name}) ---")
        try:
            ident = FursuitIdentifier(
                db_path=db_path,
                index_path=index_path,
                device=args.device,
                segmentor_model_name=Config.SAM3_MODEL,
                segmentor_concept=Config.DEFAULT_CONCEPT,
            )
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            continue

        for img_idx, (image, img_path, ground_truth) in enumerate(loaded_images):
            if image is None:
                continue
            print(f"  [{img_idx + 1}/{len(loaded_images)}] {Path(img_path).name} -> {name}")
            try:
                # Request extra results when excluding sources, so we still
                # have enough matches after filtering.
                fetch_k = args.top_k * 3 if exclude_sources else args.top_k
                segment_results = ident.identify(image, top_k=fetch_k)
                if exclude_sources:
                    filter_excluded_sources(segment_results, exclude_sources)
                    # Trim back to requested top_k
                    for seg in segment_results:
                        seg.matches = seg.matches[:args.top_k]
            except Exception as e:
                text = f"  Dataset: {name} ({emb_name}, {total_entries:,} entries)\n    ERROR: {e}"
                image_results[img_idx][name] = (text, [], None)
                continue

            text, seg_metrics, top1_char = format_segment_results(
                name, emb_name, total_entries, segment_results,
                ground_truth, chars, aliases, args.top_k,
            )
            image_results[img_idx][name] = (text, seg_metrics, top1_char)
            image_segments[img_idx][name] = segment_results

        # Free GPU memory before loading next dataset
        del ident
        free_gpu_memory()
        print(f"--- Done with {name}, freed GPU memory ---")

    # Assemble per-image output
    output_lines.append("PER-IMAGE RESULTS")
    output_lines.append("=" * 70)

    all_metrics: list[list[dict]] = []
    top1_chars: list[dict[str, str | None]] = []

    for img_idx, (image, img_path, ground_truth) in enumerate(loaded_images):
        output_lines.append(f"\nImage: {Path(img_path).name}")
        output_lines.append(f"Path: {img_path}")
        if ground_truth:
            output_lines.append(f"Ground truth: {', '.join(ground_truth)}")
        output_lines.append("")

        if image is None:
            output_lines.append("  ERROR: Could not load image")
            output_lines.append("")
            continue

        img_metrics = []
        img_top1 = {}

        for name in ds_names:
            if name not in image_results[img_idx]:
                continue
            text, seg_metrics, top1_char = image_results[img_idx][name]
            output_lines.append(text)
            output_lines.append("")
            img_metrics.extend(seg_metrics)
            img_top1[name] = top1_char

        all_metrics.append(img_metrics)
        top1_chars.append(img_top1)

    # Combined (RRF) results across datasets
    if len(ds_names) >= 2:
        output_lines.append("")
        output_lines.append("COMBINED (Reciprocal Rank Fusion)")
        output_lines.append("=" * 70)

        for img_idx, (image, img_path, ground_truth) in enumerate(loaded_images):
            if image is None:
                continue
            segments_by_ds = image_segments[img_idx]
            if len(segments_by_ds) < 2:
                continue

            all_seg_results = [segments_by_ds[name] for name in ds_names if name in segments_by_ds]
            merged = merge_multi_dataset_results(all_seg_results, top_k=args.top_k)

            output_lines.append(f"\nImage: {Path(img_path).name}")
            if ground_truth:
                output_lines.append(f"Ground truth: {', '.join(ground_truth)}")
            output_lines.append(f"  Combined ({' + '.join(name for name in ds_names if name in segments_by_ds)}):")

            for seg in merged:
                seg_gt_canonical = None
                if ground_truth and seg.segment_index < len(ground_truth):
                    seg_gt_canonical = ground_truth[seg.segment_index]

                # For RRF combined results, check all aliases for this character
                gt_aliases_lower = set()
                if seg_gt_canonical:
                    gt_aliases_lower.add(seg_gt_canonical.lower())
                    entry = aliases.get(seg_gt_canonical.lower(), {})
                    for alias in entry.values():
                        gt_aliases_lower.add(alias.lower())

                if len(merged) > 1:
                    gt_label = f", gt={seg_gt_canonical}" if seg_gt_canonical else ""
                    output_lines.append(f"    Segment {seg.segment_index}{gt_label}:")
                for rank, m in enumerate(seg.matches[:args.top_k], 1):
                    marker = ""
                    if gt_aliases_lower and m.character_name and m.character_name.lower() in gt_aliases_lower:
                        marker = " <-- CORRECT"
                    output_lines.append(f"    #{rank}: {m.character_name} ({m.confidence:.1%}){marker}")
            output_lines.append("")

    # Summary
    output_lines.append("")
    output_lines.append(format_summary(all_metrics, top1_chars, test_images, ds_names, args.top_k))

    # Write output
    output_path = Config.get_absolute_path(args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nResults written to: {output_path}")

    # Also print to stdout
    print("\n" + "=" * 70)
    print("\n".join(output_lines))


if __name__ == "__main__":
    main()
