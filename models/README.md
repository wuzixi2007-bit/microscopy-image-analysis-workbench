# Model files

Large checkpoints are intentionally excluded from Git.

Place model files here using these exact names:

- `sam_vit_b_01ec64.pth` — required for the few-shot SAM workflow. Download it through Meta's official [Segment Anything repository](https://github.com/facebookresearch/segment-anything).
- `instance_segmenter.pth` — optional custom Mask R-CNN particle checkpoint.
- `crack_segmenter.pth` — optional custom Mask R-CNN microcrack checkpoint.
- `particle_classifier.npz` — lightweight classifier used by the experimental exhaustive few-shot mode; included because it is small.

Do not publish custom weights until their training-data and redistribution terms have been checked. GitHub Releases or an external model registry are better suited than normal Git history for large files.
