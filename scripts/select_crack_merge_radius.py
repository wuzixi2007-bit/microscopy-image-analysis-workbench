"""Honest re-check of the crack merge radius (scripts/evaluate_crack_detector.py
DEFAULT_MERGE_RADIUS = 10).

That radius was picked by sweeping on the DenseCrack TEST split and the
headline numbers (74.8% recall / 73.0% precision) were then reported on the
same split -- selection and reporting on the same 100 images, so the
precision gain is optimistic. The model has no validation split (all 410
train images were used for training), so this script makes one out of the
test split itself:

  1. Run the model ONCE per image and cache its raw predictions.
  2. Split the 100 test images into two halves, A and B (random, seeded).
  3. For each half, pick the radius that maximises F1 on THAT half
     (selection set), then report recall/precision at that radius on the
     OTHER half (reporting set), which played no part in choosing it.
  4. Repeat over several random splits and pool the held-out counts, to see
     both the honest number and how stable the chosen radius is.

Both sides are merged at the same radius, exactly as in evaluate_crack_detector
(GT closing + prediction union-find) -- see that module's docstring for why
one-sided merging is worse.

Usage:
    python scripts/select_crack_merge_radius.py
    python scripts/select_crack_merge_radius.py --splits 20 --radii 0 5 10 15
"""

import argparse

import numpy as np
from PIL import Image

from core.device import resolve_device
from scripts.evaluate_crack_detector import (
    DENSECRACK_DIR,
    crack_gt_instances,
    crack_segmenter_predict,
    load_crack_segmenter,
    merge_nearby_predictions,
)
from scripts.evaluate_detector import match_strict

DEFAULT_RADII = [0, 3, 5, 8, 10, 12, 15, 20]


def f1(recall: float, precision: float) -> float:
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def counts_to_rpf(counts: np.ndarray) -> tuple[float, float, float]:
    """counts: rows of (n_gt, n_pred, n_matched), summed."""
    gt, pred, matched = counts.sum(axis=0)
    recall = matched / gt if gt else 0.0
    precision = matched / pred if pred else 0.0
    return recall, precision, f1(recall, precision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radii", type=int, nargs="+", default=DEFAULT_RADII)
    parser.add_argument("--splits", type=int, default=10, help="number of random 50/50 splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=None, help="debug: only the first N test images")
    args = parser.parse_args()

    with open(f"{DENSECRACK_DIR}/test.txt") as f:
        stems = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
    if args.n_images:
        stems = stems[: args.n_images]

    device = resolve_device("cuda")
    model = load_crack_segmenter(device)
    print(f"device: {device}; {len(stems)} DenseCrack test images; radii {args.radii}\n")

    # counts[stem][radius] = (n_gt, n_pred, n_matched)
    counts: dict[str, dict[int, tuple[int, int, int]]] = {}
    for k, stem in enumerate(stems):
        image = np.array(Image.open(f"{DENSECRACK_DIR}/images/{stem}.png").convert("RGB"))
        raw = crack_segmenter_predict(model, image, device)
        shape = image.shape[:2]
        counts[stem] = {}
        for r in args.radii:
            gt = crack_gt_instances(f"{DENSECRACK_DIR}/masks/{stem}.png", 0, shape, r)
            preds = merge_nearby_predictions(raw, r) if r > 0 else raw
            n_matched, _ = match_strict(preds, gt)
            counts[stem][r] = (len(gt), len(preds), n_matched)
        if (k + 1) % 20 == 0:
            print(f"  processed {k + 1}/{len(stems)}")

    def table(subset: list[str], r: int) -> np.ndarray:
        return np.array([counts[s][r] for s in subset])

    print("\n== Full-split sweep (the original, selection-and-report-on-the-same-images view) ==")
    print(f"{'radius':>6} {'recall':>8} {'precision':>10} {'F1':>7}")
    full = {r: counts_to_rpf(table(stems, r)) for r in args.radii}
    for r in args.radii:
        rec, prec, f = full[r]
        print(f"{r:>6} {rec:>8.1%} {prec:>10.1%} {f:>7.3f}")
    print(f"full-split F1-optimal radius: {max(args.radii, key=lambda r: full[r][2])}")

    rng = np.random.default_rng(args.seed)
    held = {"selected": [], "fixed10": [], "none": []}
    chosen = []
    for _ in range(args.splits):
        perm = rng.permutation(len(stems))
        halves = [[stems[i] for i in perm[: len(stems) // 2]], [stems[i] for i in perm[len(stems) // 2 :]]]
        for sel, rep in (halves, halves[::-1]):
            best = max(args.radii, key=lambda r: counts_to_rpf(table(sel, r))[2])
            chosen.append(best)
            held["selected"].append(table(rep, best))
            if 10 in args.radii:
                held["fixed10"].append(table(rep, 10))
            held["none"].append(table(rep, 0 if 0 in args.radii else args.radii[0]))

    print(f"\n== Held-out result: {args.splits} random 50/50 splits x 2 directions (radius chosen on one half, scored on the other) ==")
    vals, freq = np.unique(chosen, return_counts=True)
    print("radius chosen on the selection half: " + ", ".join(f"{v}px x{c}" for v, c in zip(vals, freq)))
    print(f"{'policy':<34} {'recall':>8} {'precision':>10} {'F1':>7}")
    labels = {"selected": "radius selected on other half", "fixed10": "fixed radius 10 (current default)", "none": "no merging (radius 0)"}
    for name, rows in held.items():
        if not rows:
            continue
        rec, prec, f = counts_to_rpf(np.vstack(rows))
        print(f"{labels[name]:<34} {rec:>8.1%} {prec:>10.1%} {f:>7.3f}")

    # bootstrap over images: how uncertain is the fixed-10 number on 100 images?
    if 10 in args.radii:
        boot = np.random.default_rng(args.seed + 1)
        arr = table(stems, 10)
        samples = []
        for _ in range(2000):
            idx = boot.integers(0, len(stems), len(stems))
            samples.append(counts_to_rpf(arr[idx])[:2])
        samples = np.array(samples)
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        print(f"\nbootstrap 95% CI over images at radius 10: recall {lo[0]:.1%}~{hi[0]:.1%}, precision {lo[1]:.1%}~{hi[1]:.1%}")


if __name__ == "__main__":
    main()
