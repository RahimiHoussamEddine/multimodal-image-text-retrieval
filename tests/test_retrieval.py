import numpy as np

from multimodal_retrieval.retrieval import (
    category_tfidf_scores,
    clustered_bootstrap_delta,
    image_query_hits,
    text_query_hits,
)


def test_image_to_caption_recall_accepts_any_of_five_positives() -> None:
    scores = np.array([[0.9, 0.8, 0.1, 0.0], [0.1, 0.0, 0.9, 0.8]])
    owners = np.array([0, 0, 1, 1])
    result = image_query_hits(scores, owners)
    assert result["metrics"]["r@1"] == 1.0
    assert result["metrics"]["r@5"] == 1.0


def test_caption_to_image_recall_uses_paired_image_rank() -> None:
    scores = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.0], [0.0, 0.1, 0.9]])
    result = text_query_hits(scores, np.array([0, 1, 2]), cuts=(1, 2))
    assert result["metrics"]["r@1"] == 1 / 3
    assert result["metrics"]["r@2"] == 1.0
    assert result["metrics"]["median_rank"] == 2.0


def test_category_tfidf_is_a_tag_to_caption_retrieval_baseline() -> None:
    image_scores, text_scores = category_tfidf_scores(
        [["dog", "person"], ["bus", "road"]],
        ["a dog running beside a person", "a bus on the road"],
    )
    assert image_scores.shape == (2, 2)
    assert text_scores.shape == (2, 2)
    assert image_scores[0, 0] > image_scores[0, 1]
    assert image_scores[1, 1] > image_scores[1, 0]


def test_clustered_paired_bootstrap_is_reproducible() -> None:
    model = np.array([1.0, 1.0, 0.0, 1.0])
    baseline = np.array([0.0, 1.0, 0.0, 0.0])
    result1 = clustered_bootstrap_delta(model, baseline, resamples=100)
    result2 = clustered_bootstrap_delta(model, baseline, resamples=100)
    assert result1 == result2
    assert result1["delta"] == 0.5
    assert result1["ci_95_lower"] <= result1["delta"] <= result1["ci_95_upper"]
