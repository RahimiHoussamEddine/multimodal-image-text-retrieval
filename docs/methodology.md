# Evaluation Methodology

## Corpus construction

The project downloads the official COCO 2017 annotations archive and reads the
`val2017` captions and instance annotations. Candidate images must have exactly
five captions, at least one object-category label, and per-image source-license
metadata that is CC BY 2.0 or CC0. A stable SHA-256 rank of the image ID selects
500 records from 847 eligible candidates (seed label 2026); image IDs then
determine the stored order. The selected images in this run are all CC BY 2.0.

The archive is checked against its expected size, S3 ETag/MD5, an independently
published SHA-1 reference, ZIP members and CRCs, and recorded SHA-256. Every
downloaded JPEG is decoded and stored with a local SHA-256. COCO annotation
metadata and original image licenses are treated separately. Only public
aggregate evaluation files are committed; downloaded images, captions, and
embeddings remain local and Git-ignored. In the verified run the official
annotation endpoint served HTTP, so these content checks are required before
the archive is accepted.

## Encoder and index

The encoder is `google/siglip-base-patch16-224` at immutable revision
`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`. The safetensors file is checked
against SHA-256
`2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8`. Images
and all five captions are embedded on CPU; vectors are L2-normalized and
compared by cosine similarity. Batch sizes and text truncation length are
recorded in the local metadata artifact.

The model is frozen. No fitting, fine-tuning, hyperparameter search, threshold
selection, or model selection uses this sample. `make encode` only creates the
embedding index for the app and evaluation.

## Retrieval metrics

- **Text → image:** each of 2,500 captions queries the 500-image gallery. A hit
  occurs if its paired image is in the first 1, 5, or 10 results. Median rank is
  also reported.
- **Image → caption:** each of 500 images queries the 2,500-caption gallery. A
  hit occurs if any of its five reference captions is in the first 1, 5, or 10
  results.
- **TF-IDF reference:** each image is represented by its annotated COCO object
  categories. A word and bigram TF-IDF similarity scores image tags against
  caption text. It uses ground-truth object metadata and is therefore a
  privileged, interpretable reference rather than an equally informed model.
- **Random reference:** analytical expected recall under uniformly random
  ranking.
- **Uncertainty:** for each recall cut, a paired 1,000-draw percentile bootstrap
  resamples image IDs. The five caption-query outcomes for an image stay
  together. Intervals are for SigLIP-minus-TF-IDF recall differences within this
  sample; they are not population or deployment uncertainty intervals.

## Results

| Direction | Model | R@1 | R@5 | R@10 | Median rank |
|---|---|---:|---:|---:|---:|
| Text → image | SigLIP | 0.7208 | 0.9320 | 0.9808 | 1 |
| Text → image | Category-tag TF-IDF | 0.0860 | 0.3200 | 0.4776 | 11 |
| Image → caption | SigLIP | 0.8540 | 0.9820 | 0.9900 | — |
| Image → caption | Category-tag TF-IDF | 0.1760 | 0.3480 | 0.4860 | — |

The locked report is [`../reports/evaluation.json`](../reports/evaluation.json).
The small number of images and narrow source domain limit external validity.

## Known risks

COCO is a validation split, but the frozen SigLIP model was pretrained on WebLI;
pretraining overlap with COCO or similar web images is possible and cannot be
ruled out here. Retrieval of known reference captions is easier than open-ended
semantic search against unseen text. Source-license filtering follows COCO's
per-image metadata and does not independently verify every creator or image.
Uploaded images are encoded in the running process; the app has no hosted
inference endpoint.
