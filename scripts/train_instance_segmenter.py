"""Fine-tune a COCO-pretrained Mask R-CNN on EMPS's true per-instance
particle masks -- the dedicated instance-segmentation model introduced to
replace the SAM+lightweight-classifier "穷举检测" path's ~25-33%
recall ceiling, especially for touching/clustered particles that a
zero-trained cosine-similarity classifier can't separate (see README.md).

Only EMPS is used for training -- BAM_TiO2's masks are binary (foreground/
background), not per-instance, so a connected-component-derived "instance"
there would merge touching particles into one label exactly like the
failure mode this is meant to fix. EMPS's masks are real per-particle IDs
(0=background, each particle its own ID), which is what Mask R-CNN needs.

Usage (smoke test -- a handful of images, 2-3 epochs, just to confirm the
whole pipeline runs and to measure real per-epoch time on this machine
before committing to a full run):
    python scripts/train_instance_segmenter.py --smoke-test

Usage (full run, once the smoke test's timing has been reviewed):
    python scripts/train_instance_segmenter.py --epochs 20
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from core.device import resolve_device

EMPS_DIR = "data/datasets/EMPS"
MIN_INSTANCE_AREA_PX = 20  # matches the noise floor used elsewhere in this project (mask_editor.py, measurement.py)
NUM_CLASSES = 2  # background + particle
SMOKE_TEST_N_IMAGES = 16
SMOKE_TEST_EPOCHS = 3


class EMPSInstanceDataset(Dataset):
    """Reads EMPS's (image, instance-ID mask) pairs directly -- no separate
    prepare_*.py script needed, unlike MoNuSeg/NIST, because EMPS is already
    exactly the per-instance format Mask R-CNN's target dict wants (boxes +
    binary masks + labels per instance). Images below MIN_INSTANCE_AREA_PX
    for ALL their instances are dropped at construction time so the training
    loop never has to handle an empty-target image."""

    def __init__(self, stems: list[str]):
        self.samples = []
        for stem in stems:
            mask = self._load_mask(stem)
            if self._instance_ids(mask).size > 0:
                self.samples.append(stem)

    @staticmethod
    def _load_mask(stem: str) -> np.ndarray:
        mask = np.array(Image.open(f"{EMPS_DIR}/masks/{stem}.png"))
        return mask[..., 0] if mask.ndim == 3 else mask

    @staticmethod
    def _instance_ids(mask: np.ndarray) -> np.ndarray:
        ids = np.unique(mask)
        ids = ids[ids != 0]
        return np.array([i for i in ids if (mask == i).sum() >= MIN_INSTANCE_AREA_PX])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem = self.samples[idx]
        image = np.array(Image.open(f"{EMPS_DIR}/images/{stem}.png").convert("RGB"))
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = self._load_mask(stem)
        ids = self._instance_ids(mask)

        masks, boxes = [], []
        for i in ids:
            m = mask == i
            ys, xs = np.where(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            masks.append(m)
            boxes.append([x0, y0, x1, y1])

        masks_t = torch.from_numpy(np.stack(masks).astype(np.uint8))
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        target = {
            "boxes": boxes_t,
            "labels": torch.ones((len(ids),), dtype=torch.int64),  # single foreground class: "particle"
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros((len(ids),), dtype=torch.int64),
        }
        return image_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model() -> torch.nn.Module:
    """COCO-pretrained Mask R-CNN, box/mask heads replaced for our single
    "particle" class. Backbone (ResNet-50) frozen -- only the FPN, RPN, and
    ROI heads train, standard transfer-learning practice for a dataset this
    small (366 images) and meant to reduce both training time and
    overfitting risk (see the approved plan)."""
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

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
    """Confirms predict-mode actually returns usable output shapes on a
    real image -- not just that training didn't crash."""
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
    parser.add_argument("--n-images", type=int, default=None, help="subsample train.txt to this many images (default: all)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--out", default="models/instance_segmenter.pth")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test:
        args.n_images = args.n_images or SMOKE_TEST_N_IMAGES
        args.epochs = SMOKE_TEST_EPOCHS
        args.out = "models/instance_segmenter_smoketest.pth"

    with open(f"{EMPS_DIR}/train.txt") as f:
        stems = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
    rng = np.random.default_rng(args.seed)
    if args.n_images:
        stems = list(rng.choice(stems, size=min(args.n_images, len(stems)), replace=False))

    dataset = EMPSInstanceDataset(stems)
    print(f"dataset: {len(dataset)}/{len(stems)} images have at least one valid instance (>= {MIN_INSTANCE_AREA_PX}px)")
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
