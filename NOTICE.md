# Data and model notice

The MIT license in this repository covers only original project code. COCO's annotation files
have separate terms described on the [COCO Terms of Use](https://cocodataset.org/#termsofuse)
page. The COCO Consortium does not own the underlying photographs; each image remains subject to
its source license and terms. This project inspects the per-image license metadata and uses only
images marked as CC BY 2.0 or CC0 1.0. The app links each displayed image to its source and license.
Image files, annotations, model weights, and derived embeddings are local and excluded from Git.

The image/text encoder is Google's `google/siglip-base-patch16-224` checkpoint. Its model card
declares Apache-2.0. The pinned model revision and local weight hash are recorded in the source
manifest after acquisition. SigLIP was pretrained on WebLI; possible benchmark overlap is a
limitation of the zero-shot retrieval evaluation.

This project is an independent educational work and is not endorsed by the COCO Consortium,
Microsoft, Google, or Flickr.
