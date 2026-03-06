"""Fursuit benchmark Streamlit app.

Usage:
    streamlit run benchmark/app.py

Tabs:
  Explorer  — PCA scatter of any character's embeddings, click to view images,
              lasso-select to assign ground truth cluster IDs
  Benchmark — Run benchmark + browse historical results
"""

import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

BASE = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
CONFIG_PATH = BENCHMARK_DIR / "config.yaml"

sys.path.insert(0, str(BASE))
from sam3_fursearch.storage.database import get_source_image_url
from sam3_fursearch.api.identifier import discover_datasets


DATASET_NAMES = [Path(d).stem for (d,_) in discover_datasets()]


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_character_posts(dataset: str, character: str) -> list[dict]:
    """Load all posts for a character from a dataset. Cached for 5 min."""
    db_path = BASE / f"{dataset}.db"
    index_path = BASE / f"{dataset}.index"
    if not db_path.exists() or not index_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT post_id, embedding_id, source, segmentor_model, character_name "
        "FROM detections WHERE character_name = ? ORDER BY post_id",
        (character,),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    index = faiss.read_index(str(index_path))
    posts: dict[str, dict] = {}
    for post_id, emb_id, source, seg_model, char_name in rows:
        if post_id not in posts:
            posts[post_id] = {
                "post_id": post_id,
                "dataset": dataset,
                "source": source,
                "character": char_name,
                "embeddings": [],
                "segmentor_models": [],
            }
        posts[post_id]["embeddings"].append(index.reconstruct(int(emb_id)).astype(np.float32))
        posts[post_id]["segmentor_models"].append(seg_model)
    for v in posts.values():
        v["centroid"] = np.mean(v["embeddings"], axis=0).tolist()
        v["num_embeddings"] = len(v["embeddings"])
    return list(posts.values())


@st.cache_data(ttl=300)
def get_all_characters(dataset: str) -> list[str]:
    db_path = BASE / f"{dataset}.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT DISTINCT character_name FROM detections WHERE character_name IS NOT NULL "
        "ORDER BY character_name"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """2-component PCA via SVD — no sklearn needed."""
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2))
    X = vectors - vectors.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return (X @ Vt[:2].T).astype(float)


def image_url(post_id: str, source: str, dataset: str) -> str | None:
    direct = get_source_image_url(source, post_id)
    if direct:
        return direct
    if source == "furtrack" or dataset == "furtrack":
        return f"https://furtrack.com/p/{post_id}"  # page link, not image
    return None


def thumbnail_html(url: str | None, post_id: str, width: int = 200) -> str:
    if url and url.startswith("http") and not "furtrack.com/p/" in url:
        return f'<img src="{url}" width="{width}" style="border-radius:6px"/>'
    if url:
        return f'<a href="{url}" target="_blank">{post_id[:12]}</a>'
    return f'<span style="color:#888">{post_id[:12]}</span>'


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))


def load_results() -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            r = json.loads(f.read_text())
            r["_file"] = f.name
            results.append(r)
        except Exception:
            pass
    return results


# ── Explorer tab ──────────────────────────────────────────────────────────────

def tab_explorer() -> None:
    st.header("Embedding Explorer")

    cfg = load_config()
    test_cases = {tc["id"]: tc for tc in cfg.get("test_cases", [])}

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        dataset = st.selectbox("Dataset", DATASET_NAMES, key="exp_dataset")
    with col2:
        all_chars = get_all_characters(dataset)
        char_input = st.text_input("Character name", key="exp_char",
                                   help="Start typing — matches are case-sensitive")
        matching = [c for c in all_chars if char_input.lower() in c.lower()] if char_input else all_chars[:100]
        character = st.selectbox("Select character", matching, key="exp_char_sel") if matching else None
    with col3:
        test_case_ids = ["(none)"] + list(test_cases.keys())
        tc_sel = st.selectbox("Link to test case (for ground truth)", test_case_ids, key="exp_tc")

    if not character:
        st.info("Enter a character name to explore.")
        return

    posts = load_character_posts(dataset, character)
    if not posts:
        st.warning(f"No posts found for '{character}' in {dataset}.")
        return

    # Load ground truth if a test case is selected
    tc = test_cases.get(tc_sel) if tc_sel != "(none)" else None
    gt: dict[str, int] = {str(k): int(v) for k, v in (tc or {}).get("ground_truth", {}).items()}

    # Build scatter data
    centroids = np.array([p["centroid"] for p in posts])
    coords = pca_2d(centroids)

    df_rows = []
    for i, p in enumerate(posts):
        cluster = str(gt.get(p["post_id"], "?"))
        df_rows.append({
            "post_id": p["post_id"],
            "source": p["source"],
            "pc1": float(coords[i, 0]),
            "pc2": float(coords[i, 1]),
            "cluster": cluster,
            "n_emb": p["num_embeddings"],
            "label": f"{p['post_id'][:8]} [{p['source']}] cluster={cluster}",
        })

    import pandas as pd
    df = pd.DataFrame(df_rows)

    color_seq = px.colors.qualitative.Set2
    fig = px.scatter(
        df, x="pc1", y="pc2",
        color="cluster",
        symbol="source",
        hover_data={"post_id": True, "source": True, "n_emb": True, "pc1": False, "pc2": False},
        text="label" if len(posts) <= 20 else None,
        color_discrete_sequence=color_seq,
        title=f"'{character}' in {dataset}  ({len(posts)} posts)",
        labels={"pc1": "PC1", "pc2": "PC2"},
        height=500,
    )
    fig.update_traces(marker_size=12, textposition="top center")
    fig.update_layout(legend_title_text="cluster / source")

    # Pairwise distance matrix (small datasets only)
    show_matrix = len(posts) <= 15
    if show_matrix:
        n = len(posts)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = float(np.sum((centroids[i] - centroids[j]) ** 2))
        labels_short = [p["post_id"][:8] + f"\n[{p['source']}]" for p in posts]
        heat = go.Figure(go.Heatmap(
            z=dist_matrix, x=labels_short, y=labels_short,
            colorscale="RdYlGn_r", zmin=0,
            text=[[f"{dist_matrix[i,j]:.3f}" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",
        ))
        heat.update_layout(title="Pairwise squared-L2 distance", height=400)

    # Render scatter + handle selection
    event = st.plotly_chart(fig, on_select="rerun", key="scatter_chart", use_container_width=True)
    if show_matrix:
        with st.expander("Pairwise distance matrix"):
            st.plotly_chart(heat, use_container_width=True)

    # Selected point detail
    selected_indices = []
    if event and hasattr(event, "selection") and event.selection:
        selected_indices = [pt.get("point_index") for pt in event.selection.get("points", [])]
        selected_indices = [i for i in selected_indices if i is not None]

    if selected_indices:
        st.subheader(f"Selected {len(selected_indices)} post(s)")
        cols = st.columns(min(len(selected_indices), 4))
        for col_i, pt_i in enumerate(selected_indices[:8]):
            p = posts[pt_i]
            with cols[col_i % 4]:
                url = image_url(p["post_id"], p["source"], dataset)
                st.markdown(thumbnail_html(url, p["post_id"]), unsafe_allow_html=True)
                st.caption(f"`{p['post_id'][:16]}`  [{p['source']}]  {p['num_embeddings']} embs")
                if tc:
                    cur = gt.get(p["post_id"], "")
                    st.caption(f"cluster: {cur or '(unset)'}")
    else:
        # Show all posts as a grid
        st.subheader("All posts")
        cols = st.columns(4)
        for i, p in enumerate(posts):
            with cols[i % 4]:
                url = image_url(p["post_id"], p["source"], dataset)
                st.markdown(thumbnail_html(url, p["post_id"]), unsafe_allow_html=True)
                st.caption(f"`{p['post_id'][:16]}`  [{p['source']}]\n{p['num_embeddings']} embs  cluster={gt.get(p['post_id'], '?')}")

    # Ground truth editor
    if tc:
        st.divider()
        st.subheader("Ground truth editor")
        st.caption(f"Editing test case: **{tc['id']}**")
        updated_gt = dict(gt)
        changed = False

        edit_cols = st.columns([3, 1, 3])
        with edit_cols[0]:
            for p in posts:
                pid = p["post_id"]
                cur_val = str(gt.get(pid, ""))
                url = image_url(pid, p["source"], dataset)
                st.markdown(thumbnail_html(url, pid, width=80), unsafe_allow_html=True)
                new_val = st.text_input(
                    f"{pid[:12]} [{p['source']}]",
                    value=cur_val,
                    key=f"gt_{pid}",
                    placeholder="cluster id (integer)",
                )
                if new_val and new_val.isdigit() and int(new_val) != gt.get(pid):
                    updated_gt[pid] = int(new_val)
                    changed = True
                elif not new_val and pid in updated_gt:
                    del updated_gt[pid]
                    changed = True

        if changed:
            if st.button("💾 Save ground truth to config.yaml"):
                tc["ground_truth"] = updated_gt
                save_config(cfg)
                st.success("Saved!")
                st.cache_data.clear()
                st.rerun()


# ── Benchmark tab ─────────────────────────────────────────────────────────────

def tab_benchmark() -> None:
    st.header("Benchmark")

    cfg = load_config()
    test_cases = cfg.get("test_cases", [])
    strategies = cfg.get("strategies", [])

    col1, col2 = st.columns([3, 1])
    with col1:
        run_cases = st.multiselect(
            "Test cases to run",
            [tc["id"] for tc in test_cases],
            default=[tc["id"] for tc in test_cases],
        )
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("▶ Run benchmark", type="primary")

    if run_btn:
        with st.spinner("Running benchmark…"):
            import subprocess
            case_args = []
            if len(run_cases) == 1:
                case_args = ["--case", run_cases[0]]
            result = subprocess.run(
                [sys.executable, str(BENCHMARK_DIR / "run.py")] + case_args,
                capture_output=True, text=True, cwd=str(BASE),
            )
        st.code(result.stdout + (result.stderr or ""), language="")
        st.cache_data.clear()

    # Historical results
    st.subheader("Historical results")
    results = load_results()
    if not results:
        st.info("No results yet. Run the benchmark above.")
        return

    # Build comparison table
    import pandas as pd

    all_strategy_ids = [s["id"] for s in strategies]
    rows = []
    for r in results:
        row = {
            "timestamp": r.get("timestamp", "?")[:19].replace("T", " "),
            "git": r.get("git_version", "?"),
        }
        case_accs = {}
        for cid, cdata in r.get("cases", {}).items():
            for sid, sdata in cdata.get("strategies", {}).items():
                key = f"{cid}/{sid}"
                case_accs[key] = sdata.get("accuracy")
        row.update(case_accs)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        acc_cols = [c for c in df.columns if c not in ("timestamp", "git")]
        # Highlight max per column
        def highlight_max(s):
            is_max = s == s.max()
            return ["background-color: #d4edda" if v else "" for v in is_max]

        st.dataframe(
            df.style.apply(highlight_max, subset=acc_cols),
            use_container_width=True,
            height=300,
        )

        # Line chart: mean accuracy per run
        df["mean_acc"] = df[acc_cols].mean(axis=1)
        fig = px.line(df, x="timestamp", y="mean_acc", markers=True,
                      title="Mean accuracy over runs", labels={"mean_acc": "Mean accuracy", "timestamp": ""})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Per-run detail
    with st.expander("Per-run detail"):
        run_labels = [f"{r.get('timestamp','?')[:19]}  git={r.get('git_version','?')}" for r in results]
        sel_run_label = st.selectbox("Select run", run_labels)
        sel_run = results[run_labels.index(sel_run_label)]
        for cid, cdata in sel_run.get("cases", {}).items():
            st.markdown(f"**{cid}**")
            cq = cdata.get("cluster_quality", {})
            if cq:
                sep = cq.get("separation_ratio")
                st.caption(f"separation_ratio={sep:.3f}" if sep else "separation_ratio=n/a")
            strat_rows = []
            for sid, sdata in cdata.get("strategies", {}).items():
                acc = sdata.get("accuracy")
                strat_rows.append({
                    "strategy": sid,
                    "correct": sdata.get("correct"),
                    "total": sdata.get("total"),
                    "accuracy": f"{acc*100:.0f}%" if acc is not None else "n/a",
                })
            st.table(strat_rows)


# ── Search tab ────────────────────────────────────────────────────────────────

def tab_search() -> None:
    st.header("Search Explorer")
    st.caption("Compare how different strategies rank results for a query. Requires models to be loaded.")

    cfg = load_config()
    strategies = cfg.get("strategies", [])

    query_mode = st.radio("Query", ["Character name lookup", "Upload image"], horizontal=True)

    if query_mode == "Character name lookup":
        col1, col2 = st.columns(2)
        with col1:
            ds = st.selectbox("Dataset", DATASET_NAMES, key="search_ds")
        with col2:
            char = st.text_input("Character name", key="search_char")

        if not char:
            return

        posts = load_character_posts(ds, char)
        if not posts:
            st.warning(f"No posts found for '{char}' in {ds}.")
            return

        st.write(f"**{len(posts)} posts** for '{char}' in {ds}")

        # Show centroid-level nearest neighbours across ALL datasets
        centroids = np.array([p["centroid"] for p in posts])
        char_centroid = centroids.mean(axis=0)

        results_rows = []
        for ds2 in DATASET_NAMES:
            db_path = BASE / f"{ds2}.db"
            if not db_path.exists():
                continue
            conn = sqlite3.connect(str(db_path))
            chars = conn.execute(
                "SELECT DISTINCT character_name FROM detections WHERE character_name IS NOT NULL"
            ).fetchall()
            conn.close()
            for (cname,) in chars:
                other_posts = load_character_posts(ds2, cname)
                if not other_posts:
                    continue
                other_cents = np.array([p["centroid"] for p in other_posts])
                other_cent = other_cents.mean(axis=0)
                dist = float(np.sum((char_centroid - other_cent) ** 2))
                results_rows.append({"dataset": ds2, "character": cname, "distance": dist, "posts": len(other_posts)})

        import pandas as pd
        df = pd.DataFrame(results_rows).sort_values("distance").head(20)
        df["distance"] = df["distance"].round(4)
        df = df[df["character"] != char] if char in df["character"].values else df

        st.subheader("Nearest characters (by centroid distance)")
        for _, row in df.iterrows():
            nearest_posts = load_character_posts(row["dataset"], row["character"])
            url_samples = [
                image_url(p["post_id"], p["source"], row["dataset"])
                for p in nearest_posts[:2]
            ]
            cols = st.columns([1, 1, 4])
            with cols[0]:
                st.metric(row["character"], f"d={row['distance']:.3f}", delta=row["dataset"])
            with cols[1]:
                for u in url_samples:
                    if u:
                        st.markdown(thumbnail_html(u, "", width=60), unsafe_allow_html=True)

    else:  # Upload image
        uploaded = st.file_uploader("Upload a fursuit photo", type=["jpg", "jpeg", "png", "webp"])
        if not uploaded:
            return
        st.image(uploaded, width=300)

        st.info("Full image-based search requires loading the embedding model. "
                "Run `python sam3_fursearch/api/cli.py identify --help` for CLI-based search.")

        # Show raw embedding similarity after embedding (if models available)
        if st.button("Run search (loads SigLIP + SAM3 models)"):
            with st.spinner("Loading models and embedding image…"):
                try:
                    from PIL import Image
                    from sam3_fursearch.api.identifier import FursuitIdentifier, discover_datasets, merge_multi_dataset_results
                    img = Image.open(uploaded)
                    datasets = discover_datasets(str(BASE))
                    identifiers = [FursuitIdentifier(db, idx) for db, idx in datasets]
                    all_results = [ident.identify(img, top_k=5) for ident in identifiers]
                    merged = merge_multi_dataset_results(all_results, top_k=10)
                    if merged:
                        st.subheader("Top matches")
                        for seg in merged:
                            st.write(f"Segment {seg.segment_index} (conf={seg.segment_confidence:.2f})")
                            for m in seg.matches:
                                url = image_url(m.post_id, m.source or "", m.source or "")
                                cols = st.columns([2, 1, 1])
                                with cols[0]:
                                    st.write(f"**{m.character_name}** ({m.source})")
                                    st.markdown(thumbnail_html(url, m.post_id, width=100), unsafe_allow_html=True)
                                with cols[1]:
                                    st.metric("Confidence", f"{m.confidence:.2%}")
                                with cols[2]:
                                    st.metric("Distance", f"{m.distance:.3f}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Fursearch Benchmark",
        page_icon="🐾",
        layout="wide",
    )
    st.title("🐾 Fursearch Benchmark")

    tab_exp, tab_bench, tab_search_t = st.tabs(["Explorer", "Benchmark", "Search"])
    with tab_exp:
        tab_explorer()
    with tab_bench:
        tab_benchmark()
    with tab_search_t:
        tab_search()


if __name__ == "__main__":
    main()
