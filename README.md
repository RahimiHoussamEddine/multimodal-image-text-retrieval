# Multimodal Image–Text Retrieval Workbench

A local semantic search application for finding COCO validation images from text
or finding visually similar images from an uploaded picture. It uses a frozen,
open SigLIP model and a reproducible, license-filtered 500-image sample. Model
inference runs on CPU; queries are not sent to a hosted inference service.

The project also compares bidirectional retrieval quality with a transparent
COCO object-tag TF-IDF reference and provides an interactive PCA view of the
image embeddings.

## Results

Results are from 500 deterministically selected COCO `val2017` images and their
2,500 reference captions. Recall@k counts a query as successful when a valid
paired result appears within the first *k* ranks.

| Method | Text → image R@1 | R@5 | R@10 | Image → caption R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen SigLIP | 72.08% | 93.20% | 98.08% | 85.40% | 98.20% | 99.00% |
| Category-tag TF-IDF | 8.60% | 32.00% | 47.76% | 17.60% | 34.80% | 48.60% |
| Random ranking reference | 0.20% | 1.00% | 2.00% | 0.20% | 1.00% | 1.99% |

The paired image-cluster bootstrap difference in text→image R@1 was **+63.48
percentage points** (95% percentile interval +60.24 to +66.88); for image→caption
R@1 it was **+67.80 points** (+63.00 to +72.00). The tag baseline has access to
COCO instance-category labels while SigLIP sees only pixels and caption text,
so this is an interpretable metadata reference, not a like-for-like model
comparison. Full results are in [`reports/evaluation.json`](reports/evaluation.json).

## Run locally

Requirements: Python 3.12, a CPU, and network access for the one-time data and
model downloads.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install-lock

# Download official annotations and the license-filtered image sample.
make fetch
make validate-data

# Download the exact pinned model files, create local embeddings, and evaluate.
make cache-model
make encode
make evaluate

# Launch the graphical app.
make app
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). The first model cache is
about 813 MB; the verified COCO annotation archive is about 253 MB and the 500
selected JPEGs total about 84 MB. Downloaded files and embeddings are ignored
by Git. After setup, app queries run offline.

`make encode` computes frozen embeddings; it does **not** train or fine-tune a
model. To reuse existing embeddings, run only `make evaluate` and `make app`.

## Checks

```bash
make test
make quality
make smoke-app
```

The app smoke check exercises startup, a text query, the top-k control, and
rendered matches through Streamlit AppTest. The CI workflow runs the unit,
lint, formatting, and dependency checks without downloading the dataset or
model. There is no API or Docker image: the local interactive application is
the product interface.

## Data, model, and licensing

- Dataset: [Microsoft COCO](https://cocodataset.org/), 2017 captions and
  `val2017` images. The official annotation archive is pinned by byte count,
  ETag/MD5, published SHA-1 reference, and a recorded SHA-256.
  During the verified acquisition, the official annotation endpoint served
  HTTP; the downloader validates those integrity references and ZIP CRCs before
  using the archive.
- Image selection: deterministic hash ranking of images that have five
  captions, object-category annotations, and source metadata marked CC BY 2.0
  or CC0. In this realized sample, all 500 images are marked **CC BY 2.0**.
  Images are kept locally and are not committed or redistributed by this
  repository. The app links each displayed image to its source page and license.
- COCO caption/instance annotations: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Source-image rights remain distinct and are checked per image using COCO
  metadata. See [COCO Terms of Use](https://cocodataset.org/#termsofuse) and
  [`NOTICE.md`](NOTICE.md).
- Model: [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224),
  pinned to revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`. The model is
  Apache-2.0 licensed; the weights SHA-256 is verified before use.
- Original project code: MIT License. Dataset and model terms are separate.

## Evaluation limits

This is a retrieval demonstration on a deterministic sample from COCO's
validation split, not an independent benchmark for general image search. SigLIP
was pretrained on WebLI and may have seen COCO-like or overlapping web images;
the project cannot establish that these images were unseen during pretraining.
No fine-tuning or model selection was performed on this sample. The paired
bootstrap intervals quantify sample-level differences between these methods;
they do not address pretraining contamination or deployment traffic. The
category baseline uses ground-truth object tags and is intentionally
privileged. Category-specific metrics are descriptive and should not be read as
population estimates.

## Repository layout

```text
app.py                         Streamlit search, benchmark, and embedding atlas
src/multimodal_retrieval/       Acquisition, encoding, evaluation, and ranking
data/source_manifest.json       Pinned public source and model metadata
reports/                        Aggregate evaluation and data-quality reports
scripts/smoke_app.py            Interactive app smoke check
tests/                          Unit tests for licensing and retrieval metrics
docs/                            Method, data, and reproducibility notes
```
