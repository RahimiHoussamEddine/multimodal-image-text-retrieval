"""Ranking metrics and a category-tag TF-IDF retrieval baseline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

RANK_CUTS = (1, 5, 10)


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    positions = np.empty_like(order)
    row_numbers = np.arange(scores.shape[0])[:, None]
    positions[row_numbers, order] = np.arange(scores.shape[1])[None, :]
    return positions


def image_query_hits(
    scores: np.ndarray, caption_image_indices: np.ndarray, cuts: Iterable[int] = RANK_CUTS
) -> dict[str, Any]:
    """Per-image hit indicators when images retrieve their five reference captions."""
    if scores.ndim != 2 or scores.shape[1] != len(caption_image_indices):
        raise ValueError("Image-to-caption score and target dimensions do not match")
    positions = _rank_positions(scores)
    results: dict[str, Any] = {"per_image": {}}
    for cut in cuts:
        hits = np.empty(scores.shape[0], dtype=np.float64)
        for image_index in range(scores.shape[0]):
            positives = np.flatnonzero(caption_image_indices == image_index)
            if positives.size == 0:
                raise ValueError(f"Image {image_index} has no reference captions")
            hits[image_index] = float(positions[image_index, positives].min() < cut)
        results["per_image"][int(cut)] = hits
    results["metrics"] = {
        f"r@{cut}": float(np.mean(hits)) for cut, hits in results["per_image"].items()
    }
    return results


def text_query_hits(
    scores: np.ndarray, true_image_indices: np.ndarray, cuts: Iterable[int] = RANK_CUTS
) -> dict[str, Any]:
    """Per-caption hit indicators when captions retrieve their paired image."""
    if scores.ndim != 2 or scores.shape[0] != len(true_image_indices):
        raise ValueError("Text-to-image score and target dimensions do not match")
    if np.any(true_image_indices < 0) or np.any(true_image_indices >= scores.shape[1]):
        raise ValueError("Text-to-image targets are outside the gallery")
    positions = _rank_positions(scores)
    true_ranks = positions[np.arange(scores.shape[0]), true_image_indices]
    results: dict[str, Any] = {"per_query": {}}
    for cut in cuts:
        results["per_query"][int(cut)] = (true_ranks < cut).astype(np.float64)
    results["metrics"] = {
        f"r@{cut}": float(np.mean(hits)) for cut, hits in results["per_query"].items()
    }
    results["metrics"]["median_rank"] = float(np.median(true_ranks) + 1)
    return results


def category_tfidf_scores(
    categories: list[list[str]], captions: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Score captions against object-tag documents; vocabulary/IDF fit only on image tags."""
    tag_documents = [" ".join(sorted(set(row))) for row in categories]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", norm="l2")
    tag_matrix = vectorizer.fit_transform(tag_documents)
    caption_matrix = vectorizer.transform(captions)
    image_to_caption = (tag_matrix @ caption_matrix.T).toarray().astype(np.float32)
    text_to_image = image_to_caption.T.copy()
    return image_to_caption, text_to_image


def clustered_bootstrap_delta(
    model_hits: np.ndarray,
    baseline_hits: np.ndarray,
    seed: int = 2026,
    resamples: int = 1000,
) -> dict[str, float]:
    """Paired image-cluster percentile interval for a recall difference."""
    model_values = np.asarray(model_hits, dtype=np.float64)
    baseline_values = np.asarray(baseline_hits, dtype=np.float64)
    if model_values.shape != baseline_values.shape or model_values.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be equal-length image-level arrays")
    if model_values.size == 0:
        raise ValueError("Cannot bootstrap an empty retrieval sample")
    difference = model_values - baseline_values
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(resamples, len(difference)))
    estimates = difference[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "delta": float(difference.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_resamples": int(resamples),
        "bootstrap_unit": "image ID cluster (five captions remain together)",
    }
