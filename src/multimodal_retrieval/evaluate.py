"""Evaluate frozen cross-modal retrieval against a transparent tag-based baseline."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np

from multimodal_retrieval.embeddings import INDEX_PATH, METADATA_PATH, ROOT
from multimodal_retrieval.retrieval import (
    RANK_CUTS,
    category_tfidf_scores,
    clustered_bootstrap_delta,
    image_query_hits,
    text_query_hits,
)

CORPUS_PATH = ROOT / "data/processed/corpus.json"
EVALUATION_PATH = ROOT / "reports/evaluation.json"
DATA_QUALITY_PATH = ROOT / "reports/data_quality.json"
CAPTIONS_PER_IMAGE = 5


def _random_recall(candidates: int, positives: int, cut: int) -> float:
    take = min(int(cut), int(candidates))
    if candidates <= positives or candidates - positives < take:
        return 1.0
    miss_probability = math.comb(candidates - positives, take) / math.comb(candidates, take)
    return float(1.0 - miss_probability)


def _cluster_caption_hits(hits: np.ndarray, image_count: int) -> np.ndarray:
    if len(hits) != image_count * CAPTIONS_PER_IMAGE:
        raise ValueError("Expected exactly five caption queries per image for clustered inference")
    return hits.reshape(image_count, CAPTIONS_PER_IMAGE).mean(axis=1)


def _category_recall(
    image_rows: list[dict[str, Any]],
    image_hit_arrays: dict[int, np.ndarray],
    text_hit_arrays: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for category, members in sorted(
        (
            (name, [i for i, row in enumerate(image_rows) if name in row["categories"]])
            for name in sorted({name for row in image_rows for name in row["categories"]})
        ),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        if len(members) < 10:
            continue
        member_set = set(members)
        caption_indices = [
            index
            for index, image_index in enumerate(
                np.repeat(np.arange(len(image_rows)), CAPTIONS_PER_IMAGE)
            )
            if int(image_index) in member_set
        ]
        output.append(
            {
                "category": category,
                "images": len(members),
                "siglip_image_to_caption_r@5": float(image_hit_arrays[5][members].mean()),
                "siglip_text_to_image_r@5": float(text_hit_arrays[5][caption_indices].mean()),
            }
        )
    return output


def evaluate() -> dict[str, Any]:
    if not CORPUS_PATH.exists() or not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Run `make fetch` and `make encode` before `make evaluate`")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    model_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    rows = corpus["images"]
    captions = [caption for row in rows for caption in row["captions"]]
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    caption_owner = np.repeat(np.arange(len(rows)), CAPTIONS_PER_IMAGE).astype(np.int32)
    if any(len(row["captions"]) != CAPTIONS_PER_IMAGE for row in rows):
        raise ValueError("The evaluation corpus must have exactly five captions per image")

    with np.load(INDEX_PATH, allow_pickle=False) as index:
        image_embeddings = np.asarray(index["image_embeddings"], dtype=np.float32)
        text_embeddings = np.asarray(index["text_embeddings"], dtype=np.float32)
        indexed_image_ids = np.asarray(index["image_ids"], dtype=np.int64)
        indexed_caption_owner = np.asarray(index["caption_image_indices"], dtype=np.int32)
    if not np.array_equal(image_ids, indexed_image_ids):
        raise ValueError("Embedding image IDs are not aligned with the prepared COCO sample")
    if not np.array_equal(caption_owner, indexed_caption_owner):
        raise ValueError("Embedding captions are not aligned with their COCO images")
    if image_embeddings.shape[0] != len(rows) or text_embeddings.shape[0] != len(captions):
        raise ValueError("Embedding counts do not match the evaluation corpus")

    siglip_image_to_text = image_embeddings @ text_embeddings.T
    siglip_text_to_image = siglip_image_to_text.T.copy()
    tag_image_to_text, tag_text_to_image = category_tfidf_scores(
        [row["categories"] for row in rows], captions
    )

    siglip_i2t = image_query_hits(siglip_image_to_text, caption_owner)
    tag_i2t = image_query_hits(tag_image_to_text, caption_owner)
    siglip_t2i = text_query_hits(siglip_text_to_image, caption_owner)
    tag_t2i = text_query_hits(tag_text_to_image, caption_owner)
    siglip_i2t_hits = siglip_i2t["per_image"]
    tag_i2t_hits = tag_i2t["per_image"]
    siglip_t2i_hits = siglip_t2i["per_query"]
    tag_t2i_hits = tag_t2i["per_query"]
    image_count = len(rows)
    caption_count = len(captions)

    paired_differences: dict[str, Any] = {"image_to_caption": {}, "caption_to_image": {}}
    for cut in RANK_CUTS:
        paired_differences["image_to_caption"][f"r@{cut}"] = clustered_bootstrap_delta(
            siglip_i2t_hits[cut], tag_i2t_hits[cut], seed=2026 + cut
        )
        paired_differences["caption_to_image"][f"r@{cut}"] = clustered_bootstrap_delta(
            _cluster_caption_hits(siglip_t2i_hits[cut], image_count),
            _cluster_caption_hits(tag_t2i_hits[cut], image_count),
            seed=3026 + cut,
        )

    selected_license_counts = Counter(row["license_name"] for row in rows)
    categories = sorted({category for row in rows for category in row["categories"]})
    report = {
        "dataset": {
            "name": corpus["dataset"],
            "split": "val2017 deterministic license-filtered sample",
            "images": image_count,
            "captions": caption_count,
            "captions_per_image": CAPTIONS_PER_IMAGE,
            "eligible_images_before_sampling": int(
                corpus["candidate_images_after_license_and_data_checks"]
            ),
            "sample_seed": int(corpus["sample_seed"]),
            "source_image_licenses": dict(sorted(selected_license_counts.items())),
            "unique_object_categories_in_sample": len(categories),
        },
        "protocol": {
            "model_is_frozen_pretrained": True,
            "fine_tuning": False,
            "model_selection_on_this_sample": False,
            "model_pretraining_overlap_with_coco_possible": True,
            "image_to_caption_positive_rule": (
                "any of the five COCO captions for the image in top-k"
            ),
            "caption_to_image_positive_rule": "paired image in top-k",
            "caption_queries_bootstrapped_by_image_id": True,
            "category_baseline_note": (
                "TF-IDF fits on COCO instance-category tags and retrieves caption text; it uses "
                "ground-truth object metadata rather than raw pixels and is a privileged, "
                "interpretable reference."
            ),
        },
        "model": model_metadata,
        "metrics": {
            "siglip": {
                "image_to_caption": siglip_i2t["metrics"],
                "caption_to_image": siglip_t2i["metrics"],
            },
            "category_tag_tfidf": {
                "image_to_caption": tag_i2t["metrics"],
                "caption_to_image": tag_t2i["metrics"],
            },
            "random_ranking_reference": {
                "image_to_caption": {
                    f"r@{cut}": _random_recall(caption_count, CAPTIONS_PER_IMAGE, cut)
                    for cut in RANK_CUTS
                },
                "caption_to_image": {
                    f"r@{cut}": _random_recall(image_count, 1, cut) for cut in RANK_CUTS
                },
            },
        },
        "paired_siglip_minus_category_tfidf_recall_delta_cluster_bootstrap_95ci": (
            paired_differences
        ),
        "category_recall_r_at_5_descriptive": _category_recall(
            rows, siglip_i2t_hits, siglip_t2i_hits
        ),
    }
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    quality = {
        "dataset": corpus["dataset"],
        "sample_images": image_count,
        "sample_captions": caption_count,
        "selected_licenses": dict(sorted(selected_license_counts.items())),
        "eligible_before_sample": int(corpus["candidate_images_after_license_and_data_checks"]),
        "unique_categories": len(categories),
        "raw_images_committed": False,
        "caption_rows_committed": False,
    }
    DATA_QUALITY_PATH.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    evaluate()


if __name__ == "__main__":
    evaluate()
