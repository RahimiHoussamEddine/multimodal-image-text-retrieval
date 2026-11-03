"""Create a local, frozen SigLIP image/caption embedding index."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, SiglipProcessor

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
CORPUS_PATH = ROOT / "data/processed/corpus.json"
INDEX_PATH = ROOT / "data/processed/embedding_index.npz"
METADATA_PATH = ROOT / "data/processed/embedding_metadata.json"
MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
EXPECTED_WEIGHTS_SHA256 = "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"
IMAGE_BATCH_SIZE = 12
TEXT_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_snapshot(*, local_files_only: bool) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=ROOT / ".cache/huggingface/hub",
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ],
            local_files_only=local_files_only,
        )
    )
    weights_path = snapshot / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError("Pinned SigLIP model.safetensors is missing from the cache")
    actual_hash = _sha256(weights_path)
    if actual_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(f"SigLIP safetensors hash mismatch: {actual_hash}")
    return snapshot


def prepare_model_cache() -> str:
    """Cache and verify the exact model revision before offline app use."""
    snapshot = _model_snapshot(local_files_only=False)
    return _sha256(snapshot / "model.safetensors")


def _load_encoder(*, local_files_only: bool) -> tuple[Any, Any, str]:
    torch.set_num_threads(2)
    snapshot = _model_snapshot(local_files_only=local_files_only)
    weights_path = snapshot / "model.safetensors"
    actual_hash = _sha256(weights_path)
    image_processor = AutoImageProcessor.from_pretrained(
        snapshot, local_files_only=True, use_fast=False
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=False, local_files_only=True)
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = AutoModel.from_pretrained(
        snapshot,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model, actual_hash


def _feature_tensor(features: Any) -> torch.Tensor:
    if torch.is_tensor(features):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if isinstance(features, (tuple, list)) and features:
        return features[0]
    raise TypeError(f"Cannot extract SigLIP features from {type(features).__name__}")


def load_encoder() -> tuple[Any, Any, str]:
    """Load the pinned processor and safe-tensor model on CPU."""
    return _load_encoder(local_files_only=True)


def encode_text_query(text: str, processor: Any, model: Any) -> np.ndarray:
    clean_text = str(text).strip()
    if not clean_text:
        raise ValueError("Text query cannot be empty")
    inputs = processor(
        text=[clean_text],
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt",
    )
    with torch.inference_mode():
        features = _feature_tensor(
            model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        )
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
    return features.cpu().numpy().astype(np.float32, copy=False)[0]


def encode_image_query(image: Image.Image, processor: Any, model: Any) -> np.ndarray:
    inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
    with torch.inference_mode():
        features = _feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
    return features.cpu().numpy().astype(np.float32, copy=False)[0]


def _encode_images(processor: Any, model: Any, image_rows: list[dict[str, Any]]) -> np.ndarray:
    image_features: list[np.ndarray] = []
    image_paths = [ROOT / "data/raw/images" / row["file_name"] for row in image_rows]
    for start in range(0, len(image_paths), IMAGE_BATCH_SIZE):
        batch_paths = image_paths[start : start + IMAGE_BATCH_SIZE]
        images: list[Image.Image] = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            values = _feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            values = torch.nn.functional.normalize(values.float(), p=2, dim=-1)
        image_features.append(values.cpu().numpy().astype(np.float32, copy=False))
        for image in images:
            image.close()
    return np.concatenate(image_features, axis=0)


def _encode_texts(processor: Any, model: Any, captions: list[str]) -> np.ndarray:
    text_features: list[np.ndarray] = []
    for start in range(0, len(captions), TEXT_BATCH_SIZE):
        texts = captions[start : start + TEXT_BATCH_SIZE]
        inputs = processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        with torch.inference_mode():
            values = _feature_tensor(
                model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            )
            values = torch.nn.functional.normalize(values.float(), p=2, dim=-1)
        text_features.append(values.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(text_features, axis=0)


def encode() -> dict[str, Any]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError("Run `make fetch` before creating embeddings")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    image_rows = corpus["images"]
    captions = [caption for row in image_rows for caption in row["captions"]]
    if not image_rows or not captions:
        raise ValueError("The prepared COCO corpus is empty")

    started = time.perf_counter()
    prepare_model_cache()
    processor, model, weights_sha256 = _load_encoder(local_files_only=True)
    image_embeddings = _encode_images(processor, model, image_rows)
    text_embeddings = _encode_texts(processor, model, captions)
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("SigLIP image and text embedding dimensions do not match")
    if not np.isfinite(image_embeddings).all() or not np.isfinite(text_embeddings).all():
        raise ValueError("SigLIP emitted a non-finite embedding")

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_ids=np.asarray([row["image_id"] for row in image_rows], dtype=np.int64),
        caption_image_indices=np.repeat(np.arange(len(image_rows)), 5).astype(np.int32),
    )
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_sha256": weights_sha256,
        "image_count": int(len(image_rows)),
        "caption_count": int(len(captions)),
        "embedding_dimension": int(image_embeddings.shape[1]),
        "image_batch_size": IMAGE_BATCH_SIZE,
        "text_batch_size": TEXT_BATCH_SIZE,
        "max_text_length": MAX_TEXT_LENGTH,
        "device": "cpu",
        "seconds": float(time.perf_counter() - started),
        "fine_tuned": False,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["model"]["weights_sha256"] = weights_sha256
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-model-only",
        action="store_true",
        help="Download and verify the pinned model without rebuilding the embedding index",
    )
    args = parser.parse_args()
    if args.cache_model_only:
        print(json.dumps({"weights_sha256": prepare_model_cache()}, indent=2))
    else:
        encode()


if __name__ == "__main__":
    main()
