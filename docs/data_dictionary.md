# Data and Artifact Dictionary

| Path | Contents | Git status |
|---|---|---|
| `data/source_manifest.json` | Public source URLs, COCO archive verification values, sample policy, model revision and weight hash | Committed |
| `data/raw/annotations_trainval2017.zip` | Official COCO annotation archive used to prepare the sample | Ignored; local download |
| `data/raw/images/*.jpg` | The 500 selected source images | Ignored; local download |
| `data/processed/corpus.json` | Selected image IDs, five captions, source links, license metadata and COCO categories | Ignored; contains caption text and source metadata |
| `data/processed/data_manifest.json` | Local image sizes, SHA-256 values, source URL and license for each downloaded image | Ignored |
| `data/processed/embedding_index.npz` | L2-normalized 768-dimensional image and caption vectors and their ID alignment | Ignored |
| `data/processed/embedding_metadata.json` | Model revision/hash, counts, batch sizes, runtime and device | Ignored |
| `.cache/huggingface/` | Pinned local model snapshot and tokenizer files | Ignored |
| `reports/data_quality.json` | Aggregate sample counts, license counts and category count | Committed |
| `reports/evaluation.json` | Aggregate retrieval metrics, bootstrap differences and category-level descriptive metrics | Committed |

## Source and license links

- [COCO dataset](https://cocodataset.org/)
- [COCO Terms of Use](https://cocodataset.org/#termsofuse)
- [COCO captions/instance annotation license](https://creativecommons.org/licenses/by/4.0/)
- [Creative Commons Attribution 2.0](https://creativecommons.org/licenses/by/2.0/)
- [SigLIP model card](https://huggingface.co/google/siglip-base-patch16-224)

The annotation license does not replace the individual image license. The app
shows an image's source link and the source-license link from its COCO metadata.
