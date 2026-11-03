"""Acquire a deterministic, license-filtered COCO validation sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
RAW_DIR = ROOT / "data/raw"
PROCESSED_DIR = ROOT / "data/processed"
ANNOTATION_PATH = RAW_DIR / "annotations_trainval2017.zip"
CORPUS_PATH = PROCESSED_DIR / "corpus.json"
LOCAL_DATA_MANIFEST = PROCESSED_DIR / "data_manifest.json"
SAMPLE_SIZE = 500
ALLOWED_LICENSE_URLS = {
    "http://creativecommons.org/licenses/by/2.0",
    "https://creativecommons.org/licenses/by/2.0",
    "http://creativecommons.org/publicdomain/zero/1.0",
    "https://creativecommons.org/publicdomain/zero/1.0",
}
CHUNK_BYTES = 1024 * 1024


def is_allowed_license_url(url: str) -> bool:
    return str(url).strip().rstrip("/") in ALLOWED_LICENSE_URLS


def _read_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive(path: Path, archive_info: dict[str, Any]) -> None:
    if path.stat().st_size != int(archive_info["expected_bytes"]):
        raise ValueError(f"Annotation archive byte count mismatch: {path.name}")
    if _digest(path, "md5") != str(archive_info["expected_etag"]).lower():
        raise ValueError("Annotation archive MD5 does not match its S3 ETag")
    if _digest(path, "sha1") != str(archive_info["expected_sha1_reference"]).lower():
        raise ValueError("Annotation archive does not match the published SHA-1 reference")
    actual_sha256 = _digest(path, "sha256")
    expected_sha256 = archive_info.get("sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError("Annotation archive SHA-256 changed since the first verified download")
    archive_info["sha256"] = actual_sha256
    with zipfile.ZipFile(path) as archive:
        for member in (
            "annotations/captions_val2017.json",
            "annotations/instances_val2017.json",
        ):
            if member not in archive.namelist():
                raise ValueError(f"Expected COCO annotation member missing: {member}")
            with archive.open(member) as stream:
                while stream.read(CHUNK_BYTES):
                    pass  # Reading to EOF also checks the member CRC.


def fetch_annotations() -> Path:
    manifest = _read_manifest()
    info = manifest["annotations_archive"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ANNOTATION_PATH.exists():
        _validate_archive(ANNOTATION_PATH, info)
        _write_manifest(manifest)
        return ANNOTATION_PATH

    temporary = ANNOTATION_PATH.with_suffix(".zip.part")
    request = urllib.request.Request(
        info["url"], headers={"User-Agent": "p21-multimodal-retrieval/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            expected_etag = str(info["expected_etag"]).strip('"').lower()
            actual_etag = response.headers.get("ETag", "").strip('"').lower()
            if response.status != 200 or actual_etag != expected_etag:
                raise ValueError(
                    "Unexpected COCO archive response: "
                    f"status={response.status}, ETag={actual_etag}"
                )
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
        _validate_archive(temporary, info)
        temporary.replace(ANNOTATION_PATH)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError, zipfile.BadZipFile):
        temporary.unlink(missing_ok=True)
        raise
    _write_manifest(manifest)
    print(f"Verified COCO annotations: {ANNOTATION_PATH.stat().st_size:,} bytes")
    return ANNOTATION_PATH


def _annotation_objects() -> tuple[dict[str, Any], dict[str, Any]]:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError("Run `make fetch` to acquire the pinned COCO annotations first")
    with zipfile.ZipFile(ANNOTATION_PATH) as archive:
        captions = json.loads(archive.read("annotations/captions_val2017.json"))
        instances = json.loads(archive.read("annotations/instances_val2017.json"))
    return captions, instances


def _license_url_map(
    captions: dict[str, Any], instances: dict[str, Any]
) -> dict[int, dict[str, str]]:
    caption_licenses = {int(row["id"]): row for row in captions["licenses"]}
    instance_licenses = {int(row["id"]): row for row in instances["licenses"]}
    if caption_licenses != instance_licenses:
        raise ValueError("Caption and instance annotation license tables do not match")
    return caption_licenses


def _stable_rank(image_id: int) -> str:
    return hashlib.sha256(f"2026:{image_id}".encode()).hexdigest()


def prepare_corpus() -> dict[str, Any]:
    captions, instances = _annotation_objects()
    licenses = _license_url_map(captions, instances)
    caption_groups: dict[int, list[str]] = {}
    for row in captions["annotations"]:
        caption_groups.setdefault(int(row["image_id"]), []).append(str(row["caption"]).strip())

    category_names = {int(row["id"]): str(row["name"]) for row in instances["categories"]}
    image_categories: dict[int, set[str]] = {}
    for row in instances["annotations"]:
        image_categories.setdefault(int(row["image_id"]), set()).add(
            category_names[int(row["category_id"])]
        )

    caption_images = {int(row["id"]): row for row in captions["images"]}
    instance_images = {int(row["id"]): row for row in instances["images"]}
    if set(caption_images) != set(instance_images):
        raise ValueError("COCO caption and instance image ID sets do not match")

    candidates: list[dict[str, Any]] = []
    rejected_license_counts: dict[str, int] = {}
    for image_id, image in caption_images.items():
        if len(caption_groups.get(image_id, [])) != 5:
            continue
        license_id = int(image["license"])
        license_info = licenses[license_id]
        license_url = str(license_info["url"]).rstrip("/")
        if not is_allowed_license_url(license_url):
            name = str(license_info["name"])
            rejected_license_counts[name] = rejected_license_counts.get(name, 0) + 1
            continue
        instance_image = instance_images[image_id]
        if int(instance_image["license"]) != license_id:
            raise ValueError(f"COCO image license mismatch for image {image_id}")
        categories = sorted(image_categories.get(image_id, set()))
        if not categories:
            continue
        candidates.append(
            {
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "coco_url": str(image["coco_url"]),
                "flickr_url": str(image["flickr_url"]),
                "license_id": license_id,
                "license_name": str(license_info["name"]),
                "license_url": license_url,
                "captions": caption_groups[image_id],
                "categories": categories,
            }
        )

    if len(candidates) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(candidates)} COCO validation images meet the license and caption policy; "
            f"need {SAMPLE_SIZE}"
        )
    selected = sorted(candidates, key=lambda item: _stable_rank(int(item["image_id"])))[
        :SAMPLE_SIZE
    ]
    selected.sort(key=lambda item: int(item["image_id"]))
    result = {
        "dataset": "Microsoft COCO 2017 Captions validation split",
        "sample_seed": 2026,
        "selection_policy": (
            "lowest SHA-256 image-ID ranks among val2017 images with exactly five captions, "
            "at least one instance category, and CC BY 2.0 or CC0 1.0 metadata"
        ),
        "candidate_images_after_license_and_data_checks": len(candidates),
        "rejected_source_license_counts": rejected_license_counts,
        "license_table": list(licenses.values()),
        "images": selected,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    selected_counts: dict[str, int] = {}
    for row in selected:
        selected_counts[row["license_name"]] = selected_counts.get(row["license_name"], 0) + 1
    print(
        f"Selected {len(selected)} of {len(candidates):,} eligible images; "
        f"licenses={selected_counts}; rejected license groups={rejected_license_counts}"
    )
    return result


def _download_one(image: dict[str, Any]) -> dict[str, Any]:
    image_path = RAW_DIR / "images" / str(image["file_name"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    expected_prefix = b"\xff\xd8\xff"
    if not image_path.exists():
        temporary = image_path.with_suffix(".jpg.part")
        request = urllib.request.Request(
            image["coco_url"], headers={"User-Agent": "p21-multimodal-retrieval/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            if response.status != 200:
                raise ValueError(f"Image HTTP {response.status} for {image['file_name']}")
            image["download_etag"] = response.headers.get("ETag", "").strip('"')
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
        if temporary.stat().st_size < 1024 or temporary.read_bytes()[:3] != expected_prefix:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Downloaded file is not a valid JPEG: {image['file_name']}")
        temporary.replace(image_path)
    with Image.open(image_path) as decoded:
        decoded.verify()
    return {
        "image_id": int(image["image_id"]),
        "file_name": str(image["file_name"]),
        "bytes": image_path.stat().st_size,
        "sha256": _digest(image_path, "sha256"),
        "etag": image.get("download_etag"),
        "coco_url": str(image["coco_url"]),
        "flickr_url": str(image["flickr_url"]),
        "license_name": str(image["license_name"]),
        "license_url": str(image["license_url"]),
    }


def fetch_images() -> dict[str, Any]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError("Run corpus preparation before downloading its image sample")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    manifest = {
        "dataset": corpus["dataset"],
        "sample_seed": corpus["sample_seed"],
        "images": [],
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_download_one, image) for image in corpus["images"]]
        for future in as_completed(futures):
            manifest["images"].append(future.result())
    manifest["images"].sort(key=lambda row: row["image_id"])
    LOCAL_DATA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Downloaded and verified {len(manifest['images'])} license-filtered COCO images "
        f"({sum(row['bytes'] for row in manifest['images']):,} bytes total)"
    )
    return manifest


def validate_local() -> dict[str, Any]:
    manifest = _read_manifest()
    _validate_archive(ANNOTATION_PATH, manifest["annotations_archive"])
    if not CORPUS_PATH.exists() or not LOCAL_DATA_MANIFEST.exists():
        raise FileNotFoundError("Run `make fetch` before validating the local image sample")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    image_manifest = json.loads(LOCAL_DATA_MANIFEST.read_text(encoding="utf-8"))
    if len(corpus["images"]) != SAMPLE_SIZE or len(image_manifest["images"]) != SAMPLE_SIZE:
        raise ValueError("COCO sample or local image manifest has an unexpected size")
    for row in image_manifest["images"]:
        path = RAW_DIR / "images" / row["file_name"]
        if path.stat().st_size != int(row["bytes"]) or _digest(path, "sha256") != row["sha256"]:
            raise ValueError(f"Local COCO image changed: {row['file_name']}")
        if not is_allowed_license_url(row["license_url"]):
            raise ValueError(
                f"Sample contains an image outside the license allowlist: {row['image_id']}"
            )
    print(
        "COCO annotation archive, sample licenses, image sizes, and image SHA-256 hashes verified"
    )
    return image_manifest


def fetch() -> None:
    fetch_annotations()
    prepare_corpus()
    fetch_images()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("fetch", "annotations", "prepare", "images", "validate-only")
    )
    command = parser.parse_args().command
    if command == "fetch":
        fetch()
    elif command == "annotations":
        fetch_annotations()
    elif command == "prepare":
        prepare_corpus()
    elif command == "images":
        fetch_images()
    else:
        validate_local()


if __name__ == "__main__":
    main()
