#!/usr/bin/env python3
"""Benchmark runner for fursuit identification strategies.

Usage:
    python benchmark/run.py                     # run all test cases
    python benchmark/run.py --case maple_nfc    # run one test case
    python benchmark/run.py --list              # list available test cases
    python benchmark/run.py --compare           # run + compare to best previous result
    python benchmark/run.py --explore           # print distance matrices only (no eval)
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import yaml

BASE = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
CONFIG_PATH = BENCHMARK_DIR / "config.yaml"

sys.path.insert(0, str(BASE))
from sam3_fursearch.storage.database import get_git_version
from sam3_fursearch.api.identifier import discover_datasets

DATASET_DB = {
    Path(db).stem: (Path(db), Path(index))
    for (db, index) in discover_datasets()
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_posts(db_path: Path, index_path: Path, character: str | None) -> dict[str, dict]:
    """Load all embeddings for a character (or all characters if None) from a dataset.

    Returns {post_id: {centroid, embeddings, source, segmentor_models}}.
    """
    if not db_path.exists() or not index_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    if character is None:
        rows = conn.execute(
            "SELECT post_id, embedding_id, source, segmentor_model, character_name "
            "FROM detections WHERE character_name IS NOT NULL ORDER BY post_id",
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT post_id, embedding_id, source, segmentor_model, ? "
            "FROM detections WHERE character_name = ? ORDER BY post_id",
            (character, character),
        ).fetchall()
    conn.close()
    if not rows:
        return {}
    index = faiss.read_index(str(index_path))
    posts: dict[str, dict] = {}
    for post_id, emb_id, source, seg_model, char_name in rows:
        if post_id not in posts:
            posts[post_id] = {"embeddings": [], "source": source, "segmentor_models": [], "character_name": char_name}
        posts[post_id]["embeddings"].append(index.reconstruct(int(emb_id)).astype(np.float32))
        posts[post_id]["segmentor_models"].append(seg_model)
    for v in posts.values():
        v["centroid"] = np.mean(v["embeddings"], axis=0)
    return posts


def load_cross_dataset_anchors(
    cross_dataset_cfg: dict,
    primary_posts: dict[str, dict],
) -> dict[str, list[dict]]:
    """Load anchor posts from cross-dataset config.

    Returns {dataset_name: [post_dict, ...]} where each post_dict has
    centroid, embeddings, cluster_id (the ground-truth cluster it belongs to).
    Includes both known_posts and all posts for aliased character names.
    """
    anchors: dict[str, list[dict]] = {}
    for dataset_name, cfg in cross_dataset_cfg.items():
        db_path, index_path = DATASET_DB.get(dataset_name, (None, None))
        if not db_path or not db_path.exists():
            continue
        index = faiss.read_index(str(index_path))
        conn = sqlite3.connect(str(db_path))
        dataset_anchors: list[dict] = []

        # Known individual post_ids
        for post_id, cluster_id in (cfg.get("known_posts") or {}).items():
            rows = conn.execute(
                "SELECT embedding_id, source FROM detections WHERE post_id = ?",
                (post_id,),
            ).fetchall()
            if not rows:
                continue
            embs = [index.reconstruct(int(r[0])).astype(np.float32) for r in rows]
            dataset_anchors.append({
                "post_id": post_id,
                "source": rows[0][1],
                "embeddings": embs,
                "centroid": np.mean(embs, axis=0),
                "cluster_id": cluster_id,
                "origin": f"{dataset_name}:known_post",
            })

        # Aliased character names (all posts for that name → cluster)
        for cluster_id_str, names in (cfg.get("aliases") or {}).items():
            cluster_id = int(cluster_id_str)
            for char_name in (names or []):
                char_posts = load_posts(db_path, index_path, char_name)
                for pid, pdata in char_posts.items():
                    dataset_anchors.append({
                        "post_id": pid,
                        "source": pdata["source"],
                        "embeddings": pdata["embeddings"],
                        "centroid": pdata["centroid"],
                        "cluster_id": cluster_id,
                        "origin": f"{dataset_name}:alias:{char_name}",
                    })
        conn.close()
        if dataset_anchors:
            anchors[dataset_name] = dataset_anchors
    return anchors


# ── Math helpers ──────────────────────────────────────────────────────────────

def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))


def sam3_quantiles(posts: dict) -> dict | None:
    """Pairwise squared-L2 distance quantiles across all sam3 embeddings in posts."""
    embs = [
        emb
        for pdata in posts.values()
        for emb, seg in zip(pdata["embeddings"], pdata["segmentor_models"])
        if seg == "sam3"
    ]
    if len(embs) < 2:
        return None
    dists = np.array([
        sq_dist(embs[i], embs[j])
        for i in range(len(embs))
        for j in range(i + 1, len(embs))
    ])
    return {
        "n_embeddings": len(embs),
        "n_pairs": len(dists),
        "p10": float(np.percentile(dists, 10)),
        "p25": float(np.percentile(dists, 25)),
        "p50": float(np.percentile(dists, 50)),
        "p75": float(np.percentile(dists, 75)),
        "p90": float(np.percentile(dists, 90)),
        "max": float(np.max(dists)),
    }


def intrachar_medians(posts: dict) -> list[float]:
    """Per-character median sam3 pairwise distance for every character in posts."""
    char_groups: dict[str, list] = {}
    for pdata in posts.values():
        cname = pdata.get("character_name") or ""
        if cname:
            char_groups.setdefault(cname, []).append(pdata)
    medians = []
    for char_posts in char_groups.values():
        q = sam3_quantiles({i: p for i, p in enumerate(char_posts)})
        if q is not None:
            medians.append(q["p50"])
    return medians


def intrachar_stats(medians: list[float]) -> dict | None:
    """Distribution of per-character median intra-distances."""
    if len(medians) < 2:
        return None
    arr = np.array(medians)
    return {
        "n_chars": len(medians),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def kmeans_pp(data: np.ndarray, k: int, seed: int = 42, n_iter: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [data[rng.integers(len(data))].copy()]
    for _ in range(k - 1):
        d2 = np.array([min(sq_dist(x, c) for c in centers) for x in data])
        if d2.sum() == 0:
            break
        centers.append(data[rng.choice(len(data), p=d2 / d2.sum())].copy())
    centers = np.array(centers, dtype=np.float32)
    labels = np.zeros(len(data), dtype=int)
    for _ in range(n_iter):
        new_labels = np.array([min(range(len(centers)), key=lambda j: sq_dist(x, centers[j])) for x in data])
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(len(centers)):
            if np.any(labels == j):
                centers[j] = data[labels == j].mean(axis=0)
    return labels


# ── Strategy implementations ──────────────────────────────────────────────────
# Each strategy fn signature:
#   (query_emb, db_posts, anchors, params) -> predicted_cluster_id (int)
# db_posts: list of {post_id, centroid, embeddings, cluster_id}
# anchors: {dataset_name: [post_dict, ...]} from load_cross_dataset_anchors

def _nearest(query_emb, posts):
    dists = [sq_dist(query_emb, p["centroid"]) for p in posts]
    return posts[int(np.argmin(dists))]["cluster_id"]


def strategy_avg_all(query_emb, db_posts, anchors, params):
    all_embs = np.vstack([e for p in db_posts for e in p["embeddings"]])
    global_avg = all_embs.mean(axis=0)
    dists = [sq_dist(global_avg, p["centroid"]) for p in db_posts]
    return db_posts[int(np.argmin(dists))]["cluster_id"]


def strategy_nearest_centroid(query_emb, db_posts, anchors, params):
    return _nearest(query_emb, db_posts)


def strategy_nearest_subcluster(query_emb, db_posts, anchors, params):
    threshold = params.get("threshold", 0.35)
    all_embs = np.vstack([e for p in db_posts for e in p["embeddings"]])
    mask = np.array([sq_dist(query_emb, e) for e in all_embs]) <= threshold
    filtered = all_embs[mask] if mask.any() else all_embs
    avg = filtered.mean(axis=0)
    dists = [sq_dist(avg, p["centroid"]) for p in db_posts]
    return db_posts[int(np.argmin(dists))]["cluster_id"]


def strategy_kmeans(query_emb, db_posts, anchors, params):
    k = params.get("k") or len(set(p["cluster_id"] for p in db_posts))
    k = min(k, len(db_posts))
    cents = np.array([p["centroid"] for p in db_posts])
    labels = kmeans_pp(cents, k=k)
    cluster_cents = [cents[labels == j].mean(axis=0) for j in range(k) if (labels == j).any()]
    nearest = int(np.argmin([sq_dist(query_emb, cc) for cc in cluster_cents]))
    cluster_ids = [db_posts[i]["cluster_id"] for i in range(len(db_posts)) if labels[i] == nearest]
    return max(set(cluster_ids), key=cluster_ids.count)


def strategy_anchor_filter(query_emb, db_posts, anchors, params):
    anchor_dataset = params.get("anchor_dataset", "furtrack")
    threshold = params.get("threshold", 0.35)
    anchor_list = anchors.get(anchor_dataset, [])
    if not anchor_list:
        return strategy_nearest_centroid(query_emb, db_posts, anchors, params)
    anchor_cents = np.array([a["centroid"] for a in anchor_list])
    nearest_anchor_idx = int(np.argmin([sq_dist(query_emb, ac) for ac in anchor_cents]))
    nearest_anchor_cent = anchor_cents[nearest_anchor_idx]
    filtered = [p for p in db_posts if sq_dist(p["centroid"], nearest_anchor_cent) <= threshold]
    if not filtered:
        filtered = db_posts
    return _nearest(query_emb, filtered)


STRATEGY_FNS = {
    "avg_all": strategy_avg_all,
    "nearest_centroid": strategy_nearest_centroid,
    "nearest_subcluster": strategy_nearest_subcluster,
    "kmeans": strategy_kmeans,
    "anchor_filter": strategy_anchor_filter,
}


# ── LOO evaluation ────────────────────────────────────────────────────────────

def run_loocv(
    primary_posts: dict[str, dict],
    ground_truth: dict[str, int],
    strategy_cfg: dict,
    anchors: dict[str, list[dict]],
    num_clusters: int,
) -> dict:
    """Leave-one-post-out cross-validation for a single strategy."""
    fn = STRATEGY_FNS.get(strategy_cfg["type"])
    if fn is None:
        return {"error": f"Unknown strategy type: {strategy_cfg['type']}"}

    params = dict(strategy_cfg.get("params") or {})
    if params.get("k") is None and strategy_cfg["type"] == "kmeans":
        params["k"] = num_clusters

    details = []
    for query_pid in primary_posts:
        if query_pid not in ground_truth:
            continue
        expected = ground_truth[query_pid]
        query_emb = primary_posts[query_pid]["centroid"]

        db_posts = [
            {**primary_posts[p], "cluster_id": ground_truth[p]}
            for p in primary_posts
            if p != query_pid and p in ground_truth
        ]
        if not db_posts:
            continue

        predicted = fn(query_emb, db_posts, anchors, params)
        details.append({
            "post_id": query_pid,
            "source": primary_posts[query_pid]["source"],
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
        })

    correct = sum(d["correct"] for d in details)
    total = len(details)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "details": details,
    }


def run_test_case(tc: dict, strategies: list[dict], explore: bool = False) -> dict:
    """Run all strategies on one test case. Returns result dict."""
    character = tc["character"]
    primary_ds = tc["primary_dataset"]
    ground_truth = {str(k): int(v) for k, v in (tc.get("ground_truth") or {}).items()}
    num_clusters = tc.get("num_clusters", len(set(ground_truth.values())) or 2)

    db_path, index_path = DATASET_DB.get(primary_ds, (None, None))
    if not db_path:
        return {"error": f"Unknown dataset: {primary_ds}"}

    primary_posts = load_posts(db_path, index_path, character)
    if not primary_posts:
        return {"error": f"No posts found for '{character}' in {primary_ds}"}

    anchors = load_cross_dataset_anchors(tc.get("cross_dataset") or {}, primary_posts)

    # Cluster quality metrics (requires ground truth)
    cluster_quality = {}
    if ground_truth:
        intra, inter = [], []
        pids = [p for p in primary_posts if p in ground_truth]
        for i, pi in enumerate(pids):
            for j, pj in enumerate(pids):
                if i >= j:
                    continue
                d = sq_dist(primary_posts[pi]["centroid"], primary_posts[pj]["centroid"])
                if ground_truth[pi] == ground_truth[pj]:
                    intra.append(d)
                else:
                    inter.append(d)
        cluster_quality = {
            "intra_mean": float(np.mean(intra)) if intra else None,
            "intra_max": float(np.max(intra)) if intra else None,
            "inter_mean": float(np.mean(inter)) if inter else None,
            "inter_min": float(np.min(inter)) if inter else None,
            "separation_ratio": (
                float(np.min(inter) / np.max(intra))
                if intra and inter and np.max(intra) > 0
                else None
            ),
            "theoretical_max_accuracy": (
                sum(1 for p in ground_truth if sum(1 for q in ground_truth if ground_truth[q] == ground_truth[p]) > 1)
                / len(ground_truth)
                if ground_truth else None
            ),
        }

    sam3_emb_quantiles = sam3_quantiles(primary_posts)

    if explore or not ground_truth:
        return {
            "id": tc["id"],
            "character": character,
            "primary_dataset": primary_ds,
            "num_posts": len(primary_posts),
            "num_posts_with_gt": len([p for p in primary_posts if p in ground_truth]),
            "cluster_quality": cluster_quality,
            "sam3_emb_quantiles": sam3_emb_quantiles,
            "strategies": {},
        }

    strategy_results = {}
    for sc in strategies:
        strategy_results[sc["id"]] = run_loocv(primary_posts, ground_truth, sc, anchors, num_clusters)

    return {
        "id": tc["id"],
        "character": character,
        "primary_dataset": primary_ds,
        "num_posts": len(primary_posts),
        "num_posts_with_gt": len([p for p in primary_posts if p in ground_truth]),
        "cluster_quality": cluster_quality,
        "strategies": strategy_results,
    }


# ── Results storage ───────────────────────────────────────────────────────────

def save_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def load_all_results() -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def best_result(results: list[dict]) -> dict | None:
    """Find the result with the best mean accuracy across all cases and strategies."""
    best, best_score = None, -1.0
    for r in results:
        accs = [
            case_res["accuracy"]
            for case_data in r.get("cases", {}).values()
            for case_res in case_data.get("strategies", {}).values()
            if case_res.get("accuracy") is not None
        ]
        if accs and (score := sum(accs) / len(accs)) > best_score:
            best, best_score = r, score
    return best


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_case_result(case_result: dict, best_strategies: dict | None = None) -> None:
    cid = case_result["id"]
    char = case_result["character"]
    ds = case_result["primary_dataset"]
    n = case_result["num_posts_with_gt"]
    print(f"\n  [{cid}] '{char}' in {ds}  ({n} posts with ground truth)")

    cq = case_result.get("cluster_quality", {})
    if cq:
        sep = cq.get("separation_ratio")
        theo = cq.get("theoretical_max_accuracy")
        sep_str = f"{sep:.3f}" if sep is not None else "n/a"
        theo_str = f"{theo*100:.0f}%" if theo is not None else "n/a"
        print(f"    separation_ratio={sep_str}  theoretical_max={theo_str}")
        if sep is not None and sep < 1.0:
            print(f"    ⚠  Clusters overlap (ratio<1) — geometry-based strategies have a ceiling")

    q = case_result.get("sam3_emb_quantiles")
    if q:
        print(f"    sam3 pairwise dists  ('{char}')  n={q['n_embeddings']} embs  {q['n_pairs']} pairs")
        print(f"      p10={q['p10']:.3f}  p25={q['p25']:.3f}  p50={q['p50']:.3f}  p75={q['p75']:.3f}  p90={q['p90']:.3f}  max={q['max']:.3f}")

    strats = case_result.get("strategies", {})
    if not strats:
        print("    (explore mode — no ground truth set)")
        return

    print(f"    {'strategy':<42}  acc      vs best")
    print(f"    {'-'*65}")
    for sid, res in strats.items():
        acc = res.get("accuracy")
        acc_str = f"{acc*100:.0f}%" if acc is not None else " n/a"
        best_acc = (best_strategies or {}).get(sid)
        if best_acc is not None and acc is not None:
            delta = acc - best_acc
            delta_str = f"{delta*100:+.0f}%"
            marker = " ▲" if delta > 0 else (" ▼" if delta < 0 else "  =")
        else:
            delta_str, marker = "  new", ""
        print(f"    {sid:<42}  {acc_str:<6}  {delta_str}{marker}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fursuit benchmark runner")
    parser.add_argument("--case", help="Run a specific test case by id")
    parser.add_argument("--list", action="store_true", help="List available test cases")
    parser.add_argument("--explore", action="store_true", help="Print distance info only, skip eval")
    parser.add_argument("--compare", action="store_true", help="Compare to best previous result")
    parser.add_argument("--no-save", action="store_true", help="Don't write results to disk")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    strategies = cfg.get("strategies", [])
    test_cases = cfg.get("test_cases", [])

    if args.list:
        print("Available test cases:")
        for tc in test_cases:
            gt_count = len(tc.get("ground_truth") or {})
            print(f"  {tc['id']:<30}  {tc['description']}  ({gt_count} gt posts)")
        return

    active_cases = test_cases
    if args.case:
        active_cases = [tc for tc in test_cases if tc["id"] == args.case]
        if not active_cases:
            print(f"ERROR: No test case with id '{args.case}'")
            sys.exit(1)

    # Load best previous for comparison
    prev_results = load_all_results()
    best = best_result(prev_results) if args.compare else None
    best_by_case_strategy: dict[str, dict[str, float]] = {}
    if best:
        for cid, cdata in best.get("cases", {}).items():
            best_by_case_strategy[cid] = {
                sid: res["accuracy"]
                for sid, res in cdata.get("strategies", {}).items()
                if res.get("accuracy") is not None
            }

    print("=" * 70)
    print("  FURSUIT BENCHMARK")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  git={get_git_version()}")
    print("=" * 70)

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_version": get_git_version(),
        "cases": {},
    }

    for tc in active_cases:
        result = run_test_case(tc, strategies, explore=args.explore)
        run_record["cases"][tc["id"]] = result
        print_case_result(result, best_by_case_strategy.get(tc["id"]))

    # Dataset-wide intra-character sam3 stats (explore mode only)
    if args.explore:
        print("\n── Dataset-wide intra-character sam3 distance stats ──")
        col = f"  {'dataset':<14}  {'chars':>6}  p10    p25    p50    p75    p90"
        print(col)
        print(f"  {'-' * (len(col) - 2)}")
        merged_medians: list[float] = []
        for ds_name, (db_path, index_path) in sorted(DATASET_DB.items()):
            all_posts = load_posts(db_path, index_path, None)
            medians = intrachar_medians(all_posts)
            merged_medians.extend(medians)
            stats = intrachar_stats(medians)
            if stats:
                print(
                    f"  {ds_name:<14}  {stats['n_chars']:>6}"
                    f"  {stats['p10']:.3f}  {stats['p25']:.3f}  {stats['p50']:.3f}"
                    f"  {stats['p75']:.3f}  {stats['p90']:.3f}"
                )
        merged = intrachar_stats(merged_medians)
        if merged:
            print(f"  {'-' * (len(col) - 2)}")
            print(
                f"  {'[merged]':<14}  {merged['n_chars']:>6}"
                f"  {merged['p10']:.3f}  {merged['p25']:.3f}  {merged['p50']:.3f}"
                f"  {merged['p75']:.3f}  {merged['p90']:.3f}"
            )

    # Overall summary
    all_accs = [
        res["accuracy"]
        for cdata in run_record["cases"].values()
        for res in cdata.get("strategies", {}).values()
        if res.get("accuracy") is not None
    ]
    if all_accs:
        print(f"\n  Overall mean accuracy: {sum(all_accs)/len(all_accs)*100:.1f}%  ({len(all_accs)} evaluations)")

    if not args.no_save and not args.explore:
        has_eval = any(cdata.get("strategies") for cdata in run_record["cases"].values())
        if has_eval:
            path = save_result(run_record)
            print(f"\n  Results saved to {path.relative_to(BASE)}")

    if args.compare and best:
        best_ts = best.get("timestamp", "?")
        print(f"\n  Compared against best run: {best_ts}  git={best.get('git_version', '?')}")


if __name__ == "__main__":
    main()
