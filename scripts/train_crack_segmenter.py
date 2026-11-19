"""Fine-tune a COCO-pretrained Mask R-CNN on crack instance masks -- the
dedicated crack instance-segmentation model developed to give the
"专用模型" mode real crack support instead of its current particle-only
coverage (trained exclusively on EMPS, see scripts/train_instance_segmenter.py).

Two real, measured problems this addresses:
  1. The existing instance_segmenter.pth never saw a single crack image
     during training (EMPS is pure particle data).
  2. Mask R-CNN's default RPN anchor aspect ratios (0.5, 1.0, 2.0) are tuned
     for compact/blob-shaped COCO objects, not the extreme elongation of
     real cracks -- measured on 20 real DenseCrack images: median aspect
     ratio (long side / short side) 3.35, 54% of instances > 3, 26% > 5.
     This script widens the RPN anchor generator's aspect ratios to cover
     that, on top of adding crack training data.

Data: data/datasets/DenseCrack/ ONLY (510 images, microscopy-scale dense
micro-cracks, imaged via optical/fluorescence microscopy -- not SEM, but
still microscopy). Its binary (foreground/background) mask isn't a true
per-instance label like EMPS -- connected components are used to derive
"instances" instead. Unlike BAM_TiO2 (which was explicitly excluded from
particle training because touching-but-separate particles get merged by
this approach), this is lower-risk for cracks: a real crack forms one
connected path in most cases, so "one connected component = one instance"
is a much more defensible approximation here.

**data/datasets/CrackSeg9k/ was tried first and dropped from training
(kept only for out-of-domain eval, see scripts/evaluate_crack_detector.py)
-- read this before adding it back**: CrackSeg9k is a MACRO-photo crack
benchmark collection (road/pavement/wall/ceramic surface photos: Crack500,
Rissbilder, GAPS, Masonry, Ceramic, ...), not microscopy -- SOURCES.md
already flagged this ("CrackSeg9k为宏观裂纹，勿与SEM显微裂纹混为一谈")
before this script was even written. A first version trained on CrackSeg9k
(stratified-subsampled) + DenseCrack combined; strict evaluation showed a
stark gap -- DenseCrack test: 73.9% recall / 35.6% precision, vs
CrackSeg9k test: 10.0% recall / 7.7% precision. Part of that gap was
under-representation (CrackSeg9k's 13 sub-benchmarks were capped to ~83
images each regardless of true size, so e.g. Rissbilder only got ~3.8% of
its 2189 available images), but the deeper reason is CrackSeg9k is simply
the wrong domain for this project (microscopy image analysis) -- mixing it
into training dilutes the model's capacity on the domain that actually
matters, the same reasoning that already excluded BAM_TiO2 from particle
training for a different reason. Training on DenseCrack alone avoids both
problems: no dilution, and no domain mismatch.

**Mask format quirk (now only matters if CrackSeg9k is ever added back)**:
CrackSeg9k's masks are NOT clean 0/255 binary like DenseCrack's -- most use
the full 0-255 range (soft/anti-aliased edges), so a `> 0` threshold would
count anti-aliasing noise as crack pixels. CRACKSEG9K_MASK_THRESHOLD (127)
is kept around for scripts/evaluate_crack_detector.py's CrackSeg9k
out-of-domain eval path, which still needs it.

**Data quality issue, found while building scripts/evaluate_crack_detector.py
-- fixed here too**: 41/510 DenseCrack images (~8%) have a mask whose pixel
size doesn't match its own image's (mostly 512x512 image vs 256x256 mask,
one 512x512 vs 2000x2000). A real defect in the source dataset. The first
trained checkpoint (crackseg9k+densecrack combined run) predates this fix
-- CrackInstanceDataset never checked for it, so ~3% of that run's combined
training pool likely had misaligned box/mask supervision without crashing
outright (torchvision's transform silently resizes whatever mask shape it's
given to match the image, rather than raising on a mismatch). This version
resizes a mismatched mask (nearest-neighbor) to match its image before
deriving instances, same fix as the eval script.

**Licensing note**: DenseCrack is CC BY-NC-SA 4.0 (non-commercial). A model
trained on it inherits that restriction -- fine for a research/competition
context, but flag this before any commercial claim.

Usage (smoke test first, same phased approach as train_instance_segmenter.py):
    python scripts/train_crack_segmenter.py --smoke-test

Usage (full run, once smoke-test timing is reviewed):
    python scripts/train_crack_segmenter.py --epochs 20
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

from core.device import resolve_device

CRACKSEG9K_DIR = "data/datasets/CrackSeg9k"
DENSECRACK_DIR = "data/datasets/DenseCrack"
CRACKSEG9K_MASK_THRESHOLD = 127  # soft/anti-aliased masks -- see module docstring
MIN_INSTANCE_AREA_PX = 20  # matches the noise floor used elsewhere in this project
NUM_CLASSES = 2  # background + crack
SMOKE_TEST_N_IMAGES = 16
SMOKE_TEST_EPOCHS = 3

# Widened from the default (0.5, 1.0, 2.0) to cover real crack elongation
# (measured median aspect ratio 3.35 on DenseCrack, 26% of instances > 5) --
# covers both very flat and very tall/narrow boxes since a crack's axis-
# aligned bounding box can go either way depending on its orientation.
CRACK_ANCHOR_ASPECT_RATIOS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def densecrack_entries(stems: list[str]) -> list[tuple[str, str, str, int]]:
    return [(f"{DENSECRACK_DIR}/images/{s}.png", f"{DENSECRACK_DIR}/masks/{s}.png", 0) for s in stems]


def crackseg9k_subset_tag(stem: str) -> str:
    """Still used by scripts/evaluate_crack_detector.py to stratify its
    CrackSeg9k out-of-domain eval sample across sub-benchmarks (e.g.
    CRACK500 ~2500 vs Ceramic ~80) -- extracts the leading alphabetic
    prefix before the first digit/underscore run. No longer used for
    training itself (see module docstring: CrackSeg9k was dropped from
    training, macro-photo domain mismatch)."""
    import re

    m = re.match(r"^([A-Za-z]+)", stem)
    return m.group(1) if m else stem


class CrackInstanceDataset(Dataset):
    """Reads (image, binary mask) pairs from CrackSeg9k and/or DenseCrack,
    derives per-instance targets via connected components (see module
    docstring for why this is lower-risk for cracks than it was for
    BAM_TiO2's particles). Images with zero valid instances after
    thresholding (including CrackSeg9k's "noncrack" negatives) are dropped
    at construction time, same convention as EMPSInstanceDataset."""

    def __init__(self, entries: list[tuple[str, str, str, int]]):
        self.samples = []
        for image_path, mask_path, threshold in entries:
            mask = self._load_binary_mask(mask_path, threshold)
            if self._instance_regions(mask):
                self.samples.append((image_path, mask_path, threshold))

    @staticmethod
    def _load_binary_mask(mask_path: str, threshold: int) -> np.ndarray:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask > threshold

    @staticmethod
    def _instance_regions(binary_mask: np.ndarray):
        lbl = label(binary_mask)
        return [r for r in regionprops(lbl) if r.area >= MIN_INSTANCE_AREA_PX]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path, threshold = self.samples[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        binary_mask = self._load_binary_mask(mask_path, threshold)
        if binary_mask.shape != image.shape[:2]:
            # Real data defect, not our bug -- see module docstring's "data
            # quality issue" note. Resize (nearest-neighbor, keeps it
            # binary) to the image's own shape so box/mask supervision
            # stays aligned, instead of silently training on a scale
            # mismatch the way the first checkpoint did.
            resized = Image.fromarray(binary_mask).resize((image.shape[1], image.shape[0]), Image.NEAREST)
            binary_mask = np.array(resized)
        lbl = label(binary_mask)
        regions = self._instance_regions(binary_mask)

        masks, boxes = [], []
        for r in regions:
            m = lbl == r.label
            y0, x0, y1, x1 = r.bbox  # skimage regionprops bbox: (min_row, min_col, max_row, max_col)
            masks.append(m)
            boxes.append([x0, y0, x1, y1])

        masks_t = torch.from_numpy(np.stack(masks).astype(np.uint8))
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        target = {
            "boxes": boxes_t,
            "labels": torch.ones((len(regions),), dtype=torch.int64),  # single foreground class: "crack"
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros((len(regions),), dtype=torch.int64),
        }
        return image_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model() -> torch.nn.Module:
    """Same transfer-learning setup as train_instance_segmenter.py's
    build_model() (COCO-pretrained, frozen backbone, replaced box/mask
    heads), plus a widened-aspect-ratio anchor generator (see module
    docstring's anchor discussion)."""
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    # maskrcnn_resnet50_fpn_v2() always builds its own default anchor
    # generator internally and passes it to MaskRCNN() positionally --
    # passing rpn_anchor_generator via **kwargs collides with that ("got
    # multiple values"). Swap it in after construction instead: the RPN
    # head's output channel count depends on num_anchors_per_location, so
    # it has to be rebuilt too, not just the anchor generator alone.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))  # unchanged from torchvision's own default
    aspect_ratios = (CRACK_ANCHOR_ASPECT_RATIOS,) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(model.backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

    for param in model.backbone.body.parameters():
        param.requires_grad = False

    return model


def train_one_epoch(model, optimizer, loader, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    elapsed = time.time() - t0
    return total_loss / len(loader), elapsed


def sanity_inference(model, dataset, device) -> None:
    model.eval()
    image, _ = dataset[0]
    with torch.no_grad():
        prediction = model([image.to(device)])[0]
    n = len(prediction["boxes"])
    print(
        f"sanity inference on 1 training image -> {n} candidate instances "
        f"(scores: {prediction['scores'][:5].cpu().numpy().round(3).tolist()}{'...' if n > 5 else ''})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help=f"{SMOKE_TEST_N_IMAGES} images, {SMOKE_TEST_EPOCHS} epochs, times each epoch")
    parser.add_argument("--epochs", type=int, default=SMOKE_TEST_EPOCHS)
    parser.add_argument("--n-images", type=int, default=None, help="subsample DenseCrack's train.txt to this many images (default: use all 410)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--out", default="models/crack_segmenter.pth")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test:
        args.n_images = args.n_images or SMOKE_TEST_N_IMAGES
        args.epochs = SMOKE_TEST_EPOCHS
        args.out = "models/crack_segmenter_smoketest.pth"

    with open(f"{DENSECRACK_DIR}/train.txt") as f:
        densecrack_stems = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
    print(f"available: {len(densecrack_stems)} DenseCrack images (CrackSeg9k excluded from training, see module docstring)")

    rng = np.random.default_rng(args.seed)
    if args.n_images:
        idx = rng.choice(len(densecrack_stems), size=min(args.n_images, len(densecrack_stems)), replace=False)
        densecrack_stems = [densecrack_stems[i] for i in idx]
    entries = densecrack_entries(densecrack_stems)

    dataset = CrackInstanceDataset(entries)
    print(f"dataset: {len(dataset)}/{len(entries)} images have at least one valid crack instance (>= {MIN_INSTANCE_AREA_PX}px, thresholded)")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = resolve_device("cuda")
    print(f"device: {device}")
    model = build_model().to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%, backbone frozen)")

    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.epochs):
        avg_loss, elapsed = train_one_epoch(model, optimizer, loader, device)
        per_image = elapsed / len(dataset)
        print(f"epoch {epoch + 1}/{args.epochs}: avg_loss={avg_loss:.4f}  {elapsed:.1f}s total  {per_image:.2f}s/image")

    sanity_inference(model, dataset, device)

    torch.save(model.state_dict(), args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
