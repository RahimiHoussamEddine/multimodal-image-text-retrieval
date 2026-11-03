"""Interactive local cross-modal retrieval and benchmark review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA

from multimodal_retrieval.embeddings import (
    encode_image_query,
    encode_text_query,
    load_encoder,
)

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data/processed/corpus.json"
INDEX_PATH = ROOT / "data/processed/embedding_index.npz"
EVALUATION_PATH = ROOT / "reports/evaluation.json"

st.set_page_config(page_title="Image–Text Retrieval Workbench", page_icon="🔎", layout="wide")
st.title("Multimodal Image–Text Retrieval Workbench")
st.caption(
    "Search a locally cached, license-filtered COCO sample with a frozen SigLIP model. "
    "The app sends no query or image to a hosted inference API."
)


@st.cache_data(show_spinner=False)
def load_assets() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, dict]:
    if not all(path.exists() for path in (CORPUS_PATH, INDEX_PATH, EVALUATION_PATH)):
        raise FileNotFoundError("Run `make fetch`, `make encode`, and `make evaluate` first")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    with np.load(INDEX_PATH, allow_pickle=False) as index:
        image_embeddings = np.asarray(index["image_embeddings"], dtype=np.float32)
        text_embeddings = np.asarray(index["text_embeddings"], dtype=np.float32)
        caption_image_indices = np.asarray(index["caption_image_indices"], dtype=np.int32)
    return (
        corpus["images"],
        image_embeddings,
        text_embeddings,
        caption_image_indices,
        evaluation,
    )


@st.cache_resource(show_spinner="Loading the pinned SigLIP model into local memory...")
def get_encoder():
    return load_encoder()


@st.cache_data(show_spinner=False)
def pca_projection(image_embeddings: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=2026).fit_transform(image_embeddings)


try:
    image_rows, image_embeddings, text_embeddings, caption_image_indices, evaluation = load_assets()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

metrics = evaluation["metrics"]["siglip"]
top_columns = st.columns(3)
top_columns[0].metric("Text → image R@1", f"{metrics['caption_to_image']['r@1']:.1%}")
top_columns[1].metric("Text → image R@5", f"{metrics['caption_to_image']['r@5']:.1%}")
top_columns[2].metric("Image → caption R@5", f"{metrics['image_to_caption']['r@5']:.1%}")

st.warning(
    "This is a deterministic 500-image sample from COCO val2017, restricted to per-image "
    "CC BY 2.0 metadata. It is not a representative estimate of all images or query traffic. "
    "The pretrained model may have seen COCO-like or overlapping web images during WebLI training."
)

search_tab, benchmark_tab, map_tab = st.tabs(
    ["Semantic search", "Retrieval evaluation", "Embedding atlas"]
)


def render_matches(query_embedding: np.ndarray, direction: str, top_k: int) -> None:
    if direction == "text":
        scores = image_embeddings @ query_embedding
    else:
        scores = image_embeddings @ query_embedding
    result_indices = np.argsort(-scores, kind="stable")[:top_k]
    if direction == "text":
        caption_scores = text_embeddings @ query_embedding
    else:
        caption_scores = None
    st.markdown(f"#### Top {len(result_indices)} matching COCO images")
    for rank, image_index in enumerate(result_indices, start=1):
        row = image_rows[int(image_index)]
        image_path = ROOT / "data/raw/images" / row["file_name"]
        with st.container(border=True):
            left, right = st.columns([1, 2])
            with left:
                st.image(str(image_path), width="stretch")
            with right:
                st.markdown(f"**#{rank} · image {row['image_id']}**")
                st.caption(f"SigLIP cosine similarity: {float(scores[image_index]):.4f}")
                caption_start = int(np.flatnonzero(caption_image_indices == image_index)[0])
                caption_end = int(np.flatnonzero(caption_image_indices == image_index)[-1]) + 1
                if caption_scores is not None:
                    best_caption = caption_start + int(
                        np.argmax(caption_scores[caption_start:caption_end])
                    )
                    st.write(row["captions"][best_caption - caption_start])
                with st.expander("All five COCO reference captions"):
                    for caption in row["captions"]:
                        st.write(f"• {caption}")
                source, license_link = st.columns(2)
                source.link_button("Photo source / creator", row["flickr_url"])
                license_url = row["license_url"].replace("http://", "https://", 1)
                license_link.link_button("CC BY 2.0 terms", license_url)
                st.caption(
                    "COCO caption annotations: CC BY 4.0. Photo source attribution: linked page."
                )


with search_tab:
    st.subheader("Search images with natural language")
    st.write(
        "Use a scene description or phrase. The query is encoded locally and compared with the "
        "precomputed image vectors."
    )
    text_query = st.text_input(
        "Text query", placeholder="A person riding a bicycle beside a bus", key="text_query"
    )
    top_k = st.slider("Number of images to show", 1, 12, 5, key="top_k")
    if st.button("Search images", key="search_text"):
        if not text_query.strip():
            st.warning("Enter a text query first.")
        else:
            with st.spinner("Encoding the text query on CPU..."):
                processor, model, _ = get_encoder()
                query_embedding = encode_text_query(text_query, processor, model)
            st.markdown(f"**Search results for:** {text_query.strip()}")
            render_matches(query_embedding, "text", top_k)

    st.divider()
    st.subheader("Search captions with an image")
    uploaded = st.file_uploader(
        "Choose a local image", type=["jpg", "jpeg", "png", "webp"], key="image_query"
    )
    if st.button("Find similar COCO images", key="search_image"):
        if uploaded is None:
            st.warning("Upload an image first.")
        else:
            with st.spinner("Encoding the uploaded image on CPU..."):
                processor, model, _ = get_encoder()
                with Image.open(uploaded) as query_image:
                    query_embedding = encode_image_query(query_image, processor, model)
            st.markdown("**Image-to-image results from the cross-modal image space**")
            render_matches(query_embedding, "image", top_k)

with benchmark_tab:
    st.subheader("COCO validation-sample retrieval results")
    st.caption(
        "R@k asks whether the paired image or any of the five reference captions appears in the "
        "first k results. Query-level uncertainty resamples image IDs, keeping five captions from "
        "each image together."
    )
    report_metrics = evaluation["metrics"]
    metric_rows = []
    for method, label in (
        ("siglip", "Frozen SigLIP (raw pixels/text)"),
        ("category_tag_tfidf", "TF-IDF (COCO ground-truth object tags)"),
        ("random_ranking_reference", "Random ranking reference"),
    ):
        for direction, direction_label in (
            ("caption_to_image", "Text → image"),
            ("image_to_caption", "Image → text"),
        ):
            values = report_metrics[method][direction]
            metric_rows.append(
                {
                    "Method": label,
                    "Direction": direction_label,
                    "R@1": values["r@1"],
                    "R@5": values["r@5"],
                    "R@10": values["r@10"],
                    "Median rank": values.get("median_rank"),
                }
            )
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")
    st.markdown("##### Paired cluster-bootstrap difference in recall: SigLIP minus tag TF-IDF")
    delta_rows = []
    difference_report = evaluation[
        "paired_siglip_minus_category_tfidf_recall_delta_cluster_bootstrap_95ci"
    ]
    for direction, direction_label in (
        ("caption_to_image", "Text → image"),
        ("image_to_caption", "Image → text"),
    ):
        for metric, result in difference_report[direction].items():
            delta_rows.append(
                {
                    "Direction": direction_label,
                    "Metric": metric,
                    "Delta": result["delta"],
                    "95% lower": result["ci_95_lower"],
                    "95% upper": result["ci_95_upper"],
                }
            )
    st.dataframe(pd.DataFrame(delta_rows), hide_index=True, width="stretch")
    st.info(
        "The TF-IDF comparator has COCO object-category annotations, while SigLIP receives only "
        "image pixels and caption text. It is a privileged metadata baseline, not an equally "
        "informed competing model. SigLIP is frozen; no fine-tuning or model selection was run."
    )

with map_tab:
    st.subheader("Image embedding atlas")
    st.caption(
        "PCA projects the frozen image embeddings to two dimensions for browsing. Distances in "
        "this plot are not retrieval metrics."
    )
    projection = pca_projection(image_embeddings)
    category_options = ["All images"] + sorted(
        {category for row in image_rows for category in row["categories"]}
    )
    selected_category = st.selectbox("Filter by COCO object category", category_options)
    points = []
    for i, row in enumerate(image_rows):
        if selected_category != "All images" and selected_category not in row["categories"]:
            continue
        points.append(
            {
                "x": float(projection[i, 0]),
                "y": float(projection[i, 1]),
                "image_id": row["image_id"],
                "categories": ", ".join(row["categories"]),
                "caption": row["captions"][0],
            }
        )
    figure = px.scatter(
        pd.DataFrame(points),
        x="x",
        y="y",
        hover_name="image_id",
        hover_data={"categories": True, "caption": True, "x": False, "y": False},
        title=f"Frozen SigLIP image embeddings · {len(points)} images",
    )
    figure.update_layout(height=620, margin={"l": 20, "r": 20, "t": 55, "b": 20})
    st.plotly_chart(figure, width="stretch")
