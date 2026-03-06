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
def load_all_posts(dataset: str, limit: int = 500) -> list[dict]:
    """Load a random sample of posts across all characters from a dataset."""
    db_path = BASE / f"{dataset}.db"
    index_path = BASE / f"{dataset}.index"
    if not db_path.exists() or not index_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    post_rows = conn.execute(
        "SELECT DISTINCT post_id, source, character_name FROM detections "
        "WHERE character_name IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (limit,),
    ).fetchall()
    if not post_rows:
        conn.close()
        return []
    post_ids = [r[0] for r in post_rows]
    post_meta = {r[0]: (r[1], r[2]) for r in post_rows}
    placeholders = ",".join("?" * len(post_ids))
    emb_rows = conn.execute(
        f"SELECT post_id, embedding_id FROM detections WHERE post_id IN ({placeholders})",
        post_ids,
    ).fetchall()
    conn.close()
    index = faiss.read_index(str(index_path))
    posts_dict: dict[str, dict] = {}
    for post_id, emb_id in emb_rows:
        source, char_name = post_meta[post_id]
        if post_id not in posts_dict:
            posts_dict[post_id] = {
                "post_id": post_id,
                "dataset": dataset,
                "source": source,
                "character": char_name,
                "embeddings": [],
            }
        posts_dict[post_id]["embeddings"].append(
            index.reconstruct(int(emb_id)).astype(np.float32)
        )
    for v in posts_dict.values():
        v["centroid"] = np.mean(v["embeddings"], axis=0).tolist()
        v["num_embeddings"] = len(v["embeddings"])
    return list(posts_dict.values())


def search_nearest_posts(
    query_emb: np.ndarray, dataset_names: list[str], limit: int = 500
) -> list[dict]:
    """Search FAISS indices for posts nearest to query_emb across multiple datasets."""
    all_posts: list[dict] = []
    for ds in dataset_names:
        db_path = BASE / f"{ds}.db"
        index_path = BASE / f"{ds}.index"
        if not db_path.exists() or not index_path.exists():
            continue
        index = faiss.read_index(str(index_path))
        if index.ntotal == 0:
            continue
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = limit
        k = min(limit, index.ntotal)
        D, I = index.search(query_emb.reshape(1, -1).astype(np.float32), k)
        id_to_dist = {int(i): float(d) for i, d in zip(I[0], D[0]) if i >= 0}
        if not id_to_dist:
            continue
        emb_ids = list(id_to_dist.keys())
        conn = sqlite3.connect(str(db_path))
        placeholders = ",".join("?" * len(emb_ids))
        rows = conn.execute(
            f"SELECT post_id, embedding_id, source, character_name FROM detections "
            f"WHERE embedding_id IN ({placeholders})",
            emb_ids,
        ).fetchall()
        conn.close()
        posts_dict: dict[str, dict] = {}
        for post_id, emb_id, source, char_name in rows:
            if post_id not in posts_dict:
                posts_dict[post_id] = {
                    "post_id": post_id,
                    "dataset": ds,
                    "source": source,
                    "character": char_name or "unknown",
                    "embeddings": [],
                    "min_dist": float("inf"),
                }
            posts_dict[post_id]["embeddings"].append(
                index.reconstruct(int(emb_id)).astype(np.float32)
            )
            posts_dict[post_id]["min_dist"] = min(
                posts_dict[post_id]["min_dist"], id_to_dist.get(int(emb_id), float("inf"))
            )
        for v in posts_dict.values():
            v["centroid"] = np.mean(v["embeddings"], axis=0).tolist()
            v["num_embeddings"] = len(v["embeddings"])
        all_posts.extend(posts_dict.values())
    all_posts.sort(key=lambda p: p["min_dist"])
    return all_posts[:limit]


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


def find_local_image(post_id: str, source: str, dataset: str) -> Path | None:
    """Return the local file path for a post image, if it exists."""
    # fursearch: tg_download/{post_id}.jpg
    if dataset == "fursearch" or source == "tgbot":
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = BASE / "datasets" / "fursearch" / "tg_download" / f"{post_id}{ext}"
            if p.exists():
                return p
    return None



def render_image(post_id: str, source: str, dataset: str, width: int = 200) -> None:
    """Render a post image using st.image (local file or URL) or a fallback caption."""
    local = find_local_image(post_id, source, dataset)
    if local:
        st.image(str(local), width=width)
        return
    url = get_source_image_url(source, post_id)
    if url:
        st.image(url, width=width)
        return
    # furtrack: just show a link
    if source == "furtrack" or dataset == "furtrack":
        st.markdown(f"[furtrack.com/p/{post_id}](https://furtrack.com/p/{post_id})", unsafe_allow_html=False)
        return
    st.caption(f"no image · `{post_id[:14]}`")


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

def _tab_explorer_upload(uploaded, dataset_names: list[str], limit: int) -> None:
    """Render the upload-image vicinity explorer."""
    import hashlib
    from collections import Counter
    from PIL import Image as PILImage

    file_bytes = uploaded.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
    uploaded.seek(0)
    img = PILImage.open(uploaded)

    col_img, col_btn = st.columns([1, 3])
    with col_img:
        st.image(img, width=200)

    cache_key = f"upload_vicinity_{file_hash}"
    if cache_key not in st.session_state:
        with col_btn:
            if not st.button("Embed & search vicinity (loads SigLIP)", key="embed_btn"):
                return
        with st.spinner("Loading SigLIP and embedding image…"):
            try:
                from sam3_fursearch.models.embedder import SigLIPEmbedder
                embedder = SigLIPEmbedder()
                query_emb = embedder.embed(img)
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                return
        with st.spinner(f"Searching {limit} nearest posts…"):
            posts = search_nearest_posts(query_emb, dataset_names, limit=limit)
        st.session_state[cache_key] = {"emb": query_emb, "posts": posts}
        st.rerun()

    state = st.session_state[cache_key]
    query_emb: np.ndarray = state["emb"]
    posts: list[dict] = state["posts"]

    if not posts:
        st.warning("No nearby posts found.")
        return

    centroids = np.array([p["centroid"] for p in posts])
    all_vecs = np.vstack([centroids, query_emb.reshape(1, -1)])
    coords = pca_2d(all_vecs)
    query_coord = coords[-1]
    post_coords = coords[:-1]

    import pandas as pd
    char_counts = Counter(p["character"] for p in posts)
    top_chars = {c for c, _ in char_counts.most_common(20)}

    df_rows = []
    for i, p in enumerate(posts):
        char = p["character"]
        df_rows.append({
            "post_id": p["post_id"],
            "source": p["source"],
            "dataset": p["dataset"],
            "character": char if char in top_chars else "other",
            "pc1": float(post_coords[i, 0]),
            "pc2": float(post_coords[i, 1]),
            "n_emb": p["num_embeddings"],
            "dist": round(p.get("min_dist", 0.0), 4),
        })
    df = pd.DataFrame(df_rows)

    fig = px.scatter(
        df, x="pc1", y="pc2",
        color="character",
        symbol="dataset",
        custom_data=["post_id", "source", "dataset", "character", "dist", "n_emb"],
        title=f"Vicinity of uploaded image — {len(posts)} nearest posts",
        labels={"pc1": "PC1", "pc2": "PC2"},
        height=550,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    fig.update_traces(
        marker_size=10,
        hovertemplate=(
            "<b>%{customdata[3]}</b>  [%{customdata[2]}]<br>"
            "post: %{customdata[0]}<br>"
            "source: %{customdata[1]}  dist: %{customdata[4]}<br>"
            "<i>click to view image</i>"
            "<extra></extra>"
        ),
    )
    fig.add_trace(go.Scatter(
        x=[float(query_coord[0])], y=[float(query_coord[1])],
        mode="markers+text",
        marker=dict(symbol="star", size=22, color="red", line=dict(width=1, color="darkred")),
        text=["[query]"],
        textposition="top center",
        name="★ query",
        hovertemplate="<b>Uploaded image (query)</b><extra></extra>",
    ))
    fig.update_layout(legend_title_text="character / dataset")

    event = st.plotly_chart(fig, on_select="rerun", key="scatter_upload", width="stretch")

    selected_indices = []
    if event and hasattr(event, "selection") and event.selection:
        selected_indices = [pt.get("point_index") for pt in event.selection.get("points", [])]
        selected_indices = [i for i in selected_indices if i is not None and i < len(posts)]

    if selected_indices:
        st.subheader(f"Selected {len(selected_indices)} post(s)")
        sel_cols = st.columns(4)
        for col_i, pt_i in enumerate(selected_indices[:20]):
            p = posts[pt_i]
            with sel_cols[col_i % 4]:
                render_image(p["post_id"], p["source"], p["dataset"], width=220)
                st.caption(
                    f"`{p['post_id'][:16]}`  [{p['source']}]  "
                    f"{p['character']}  d={p.get('min_dist', 0):.4f}"
                )
    else:
        st.info("Click or lasso-select points to view their images.")


def tab_explorer() -> None:
    st.header("Embedding Explorer")

    cfg = load_config()
    test_cases = {tc["id"]: tc for tc in cfg.get("test_cases", [])}

    MAX_POINTS = 500

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        dataset = st.selectbox("Dataset", DATASET_NAMES, key="exp_dataset")
    with col2:
        all_chars = get_all_characters(dataset)
        char_input = st.text_input("Character (blank = all)", key="exp_char",
                                   help="Leave blank to sample all characters")
        if char_input:
            matching = [c for c in all_chars if char_input.lower() in c.lower()]
            character = st.selectbox("Select character", matching, key="exp_char_sel") if matching else None
        else:
            character = None
    with col3:
        test_case_ids = ["(none)"] + list(test_cases.keys())
        tc_sel = st.selectbox("Link to test case (for ground truth)", test_case_ids, key="exp_tc")

    uploaded = st.file_uploader(
        "Upload image to explore vicinity (searches all datasets)",
        type=["jpg", "jpeg", "png", "webp"], key="exp_upload",
    )
    if uploaded is not None:
        _tab_explorer_upload(uploaded, DATASET_NAMES, MAX_POINTS)
        return

    # Load posts
    if character:
        posts = load_character_posts(dataset, character)
        color_col = "cluster"
        title = f"'{character}' in {dataset}  ({len(posts)} posts)"
    else:
        posts = load_all_posts(dataset, limit=MAX_POINTS)
        color_col = "character"
        title = f"All characters in {dataset}  ({len(posts)} sampled posts)"

    if not posts:
        st.warning(f"No posts found in {dataset}.")
        return

    # Load ground truth if a test case is selected
    tc = test_cases.get(tc_sel) if tc_sel != "(none)" else None
    gt: dict[str, int] = {str(k): int(v) for k, v in (tc or {}).get("ground_truth", {}).items()}

    # Build scatter data
    centroids = np.array([p["centroid"] for p in posts])
    coords = pca_2d(centroids)

    import pandas as pd
    from collections import Counter
    if color_col == "character":
        top_chars = {c for c, _ in Counter(p["character"] for p in posts).most_common(20)}
    else:
        top_chars = None

    df_rows = []
    for i, p in enumerate(posts):
        cluster = str(gt.get(p["post_id"], "?"))
        char = p["character"]
        char_display = (char if char in top_chars else "other") if top_chars else char
        df_rows.append({
            "post_id": p["post_id"],
            "source": p["source"],
            "character": char_display,
            "pc1": float(coords[i, 0]),
            "pc2": float(coords[i, 1]),
            "cluster": cluster,
            "n_emb": p["num_embeddings"],
            "label": f"{p['post_id'][:8]} [{p['source']}]",
        })

    df = pd.DataFrame(df_rows)

    color_seq = (px.colors.qualitative.Alphabet if color_col == "character"
                 else px.colors.qualitative.Set2)
    fig = px.scatter(
        df, x="pc1", y="pc2",
        color=color_col,
        symbol="source",
        custom_data=["post_id", "source", "cluster", "n_emb", "character"],
        text="label" if len(posts) <= 20 else None,
        color_discrete_sequence=color_seq,
        title=title,
        labels={"pc1": "PC1", "pc2": "PC2"},
        height=520,
    )
    fig.update_traces(
        marker_size=10 if len(posts) > 50 else 14,
        textposition="top center",
        hovertemplate=(
            "<b>%{customdata[4]}</b>  cluster: %{customdata[2]}<br>"
            "post: %{customdata[0]}<br>"
            "source: %{customdata[1]}  embeddings: %{customdata[3]}<br>"
            "<i>click to view image</i>"
            "<extra></extra>"
        ),
    )
    fig.update_layout(legend_title_text=color_col)

    # Pairwise distance matrix (only in single-character mode with few posts)
    show_matrix = character is not None and len(posts) <= 15
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
    event = st.plotly_chart(fig, on_select="rerun", key="scatter_chart", width="stretch")
    if show_matrix:
        with st.expander("Pairwise distance matrix"):
            st.plotly_chart(heat, width="stretch")

    # Selected point detail
    selected_indices = []
    if event and hasattr(event, "selection") and event.selection:
        selected_indices = [pt.get("point_index") for pt in event.selection.get("points", [])]
        selected_indices = [i for i in selected_indices if i is not None and i < len(posts)]

    if selected_indices:
        st.subheader(f"Selected {len(selected_indices)} post(s)")
        sel_cols = st.columns(4)
        for col_i, pt_i in enumerate(selected_indices[:20]):
            p = posts[pt_i]
            with sel_cols[col_i % 4]:
                render_image(p["post_id"], p["source"], p.get("dataset", dataset), width=220)
                st.caption(
                    f"`{p['post_id'][:16]}`  [{p['source']}]  {p['num_embeddings']} embs"
                    + (f"  cluster={gt[p['post_id']]}" if p["post_id"] in gt else "")
                )
    else:
        if len(posts) <= 50:
            st.subheader("All posts")
            grid_cols = st.columns(4)
            for i, p in enumerate(posts):
                with grid_cols[i % 4]:
                    render_image(p["post_id"], p["source"], p.get("dataset", dataset), width=180)
                    st.caption(
                        f"`{p['post_id'][:16]}`  [{p['source']}]  "
                        f"{p['num_embeddings']} embs  cluster={gt.get(p['post_id'], '?')}"
                    )
        else:
            st.info("Click or lasso-select points on the chart to view images.")

    # Ground truth editor — only in single-character mode
    if tc and character:
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
                render_image(pid, p["source"], dataset, width=100)
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
            width="stretch",
            height=300,
        )

        # Line chart: mean accuracy per run
        df["mean_acc"] = df[acc_cols].mean(axis=1)
        fig = px.line(df, x="timestamp", y="mean_acc", markers=True,
                      title="Mean accuracy over runs", labels={"mean_acc": "Mean accuracy", "timestamp": ""})
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

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
            cols = st.columns([1, 1, 4])
            with cols[0]:
                st.metric(row["character"], f"d={row['distance']:.3f}", delta=row["dataset"])
            with cols[1]:
                for p in nearest_posts[:2]:
                    render_image(p["post_id"], p["source"], row["dataset"], width=60)

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
                                cols = st.columns([2, 1, 1])
                                with cols[0]:
                                    st.write(f"**{m.character_name}** ({m.source})")
                                    render_image(m.post_id, m.source or "", m.source or "", width=100)
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
