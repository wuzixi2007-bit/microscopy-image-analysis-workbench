"""End-to-end evaluation of app.py's "专用模型 + 参考图打分" fusion route:

    candidates = InstanceSegmenter(raw image)                     (score >= 0.5)
    sim        = FewShotDetector.score_masks(preprocessed image, candidates)
    result     = candidates with sim >= threshold

The earlier numbers for this route only measured how well the similarity
separates same-material from different-material masks (README / the
作品方案 doc); the recall and precision of the whole pipeline were never
measured. This script does, with the same strict one-to-one matching
(IoU>=0.5, Hungarian) as scripts/evaluate_detector.py.

Reference scenarios -- what the user's click(s) came from:
  self        refs picked from the target image's OWN ground truth. The most
              favourable case (same material, same magnification); also the
              protocol of the existing few-shot baselines.
  cross-same  refs from a DIFFERENT image of the same dataset.
  cross-other refs from the OTHER dataset (EMPS <-> BAM_TiO2): the user
              picked the wrong material. The filter is supposed to reject
              nearly everything here, and does.
Reference size: "n1" = the single median-area particle (what one click is),
"n3" = small/median/large percentile particles (the few-shot baselines').

Threshold 0 = no filter (the instance model on its own). Nothing is tuned
on these numbers: the threshold sweep is reported as-is, so the app's 0.78
default can be read off against the alternatives.

Also reported per scenario: the AUC of the similarity score for telling a
candidate that matches a real particle from one that does not -- the
quantity that decides whether ANY threshold can help precision.

Usage:
    python scripts/evaluate_fusion.py
    python scripts/evaluate_fusion.py --n-emps 30 --seed 1
"""

import argparse
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

import config
from core.detectors.few_shot import FewShotDetector
from core.detectors.instance_seg import InstanceSegmenter
from core.preprocessing import preprocess
from core.reliability import FEATURE_GRID_SIZE, MIN_GRID_CELLS
from core.segmentation.sam_wrapper import SamSegmenter
from scripts.evaluate_detector import bam_tio2_gt_instances, emps_gt_instances, pick_reference_masks

THRESHOLDS = [0.0, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.85, 0.90]


def load_emps(stem):
    return np.array(Image.open(f"data/datasets/EMPS/images/{stem}.png").convert("RGB"))


def load_bam(stem):
    return np.array(Image.open(f"data/datasets/BAM_TiO2/prepared/images/{stem}").convert("RGB"))


def read_stems(path, strip_ext):
    with open(path) as f:
        stems = [line.strip() for line in f if line.strip()]
    return [s.rsplit(".", 1)[0] for s in stems] if strip_ext else stems


def iou_matrix(cands: np.ndarray, gt: list[np.ndarray]) -> np.ndarray:
    """(n_cand, n_gt) IoU. GT instances here are disjoint (distinct EMPS ids /
    connected components), so one label map gives every intersection with a
    bincount per candidate -- exact, and far cheaper than N x M full-image ANDs."""
    gt_map = np.zeros(cands.shape[1:], dtype=np.int32)
    for i, g in enumerate(gt, start=1):
        gt_map[g] = i
    gt_area = np.array([g.sum() for g in gt], dtype=np.float64)
    out = np.zeros((len(cands), len(gt)))
    for k, c in enumerate(cands):
        inter = np.bincount(gt_map[c], minlength=len(gt) + 1)[1:].astype(np.float64)
        union = c.sum() + gt_area - inter
        out[k] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return out


def match_count(iou_mat: np.ndarray, keep: np.ndarray) -> tuple[int, np.ndarray]:
    """Strict one-to-one matches among the kept rows; also which kept rows hit."""
    rows = np.flatnonzero(keep)
    if len(rows) == 0 or iou_mat.shape[1] == 0:
        return 0, np.zeros(len(keep), bool)
    sub = iou_mat[rows]
    r, c = linear_sum_assignment(-sub)
    ok = sub[r, c] >= 0.5
    hit = np.zeros(len(keep), bool)
    hit[rows[r[ok]]] = True
    return int(ok.sum()), hit


def median_reference(gt: list[np.ndarray], shape) -> list[np.ndarray]:
    h, w = shape[:2]
    cell_area = (min(h, w) / FEATURE_GRID_SIZE) ** 2
    usable = [g for g in gt if g.sum() / cell_area >= MIN_GRID_CELLS] or gt
    areas = np.array([g.sum() for g in usable])
    return [usable[int(np.argsort(areas)[len(areas) // 2])]]


def build_mosaic(emps_t: dict, bam_t: dict, rng, instance_seg, prep) -> dict:
    """EMPS image (left, native size) next to a random 512x512 BAM_TiO2 crop
    (right): a synthetic mixed-material scene, the case the reference
    filter exists for. Ground truth is material-specific -- "want EMPS"
    counts only the EMPS particles, so candidates on the BAM_TiO2 half are
    false positives, and vice versa. Candidates come from running the
    instance model on the composite, as the app would."""
    e_img, b_img = emps_t["raw"], bam_t["raw"]
    size = 512
    y0 = int(rng.integers(0, b_img.shape[0] - size + 1))
    x0 = int(rng.integers(0, b_img.shape[1] - size + 1))
    b_crop = b_img[y0 : y0 + size, x0 : x0 + size]
    height = max(e_img.shape[0], size)
    raw = np.zeros((height, e_img.shape[1] + size, 3), np.uint8)
    raw[: e_img.shape[0], : e_img.shape[1]] = e_img
    raw[:size, e_img.shape[1] :] = b_crop

    def embed(masks, y_off, x_off, crop=None):
        out = []
        for m in masks:
            if crop is not None:
                m = m[crop[0] : crop[0] + size, crop[1] : crop[1] + size]
            if m.sum() < 25:
                continue
            full = np.zeros(raw.shape[:2], bool)
            full[y_off : y_off + m.shape[0], x_off : x_off + m.shape[1]] = m
            out.append(full)
        return out

    gt_emps = embed(emps_t["gt"], 0, 0)
    gt_bam = embed(bam_t["gt"], 0, e_img.shape[1], crop=(y0, x0))
    cands = instance_seg.predict(raw)
    return dict(
        raw=raw, processed=prep(raw), cands=cands, gt_emps=gt_emps, gt_bam=gt_bam,
        iou_emps=iou_matrix(cands, gt_emps) if len(cands) and gt_emps else np.zeros((len(cands), len(gt_emps))),
        iou_bam=iou_matrix(cands, gt_bam) if len(cands) and gt_bam else np.zeros((len(cands), len(gt_bam))),
    )


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-emps", type=int, default=None, help="use only a random subset of the 99 EMPS test images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref-sizes", nargs="+", default=["n1", "n3"], choices=["n1", "n3"], help="n1 = one median particle, n3 = small/median/large")
    parser.add_argument("--mosaic-pairs", type=int, default=0, help="also build N synthetic EMPS|BAM_TiO2 side-by-side images (the mixed-material case the filter is designed for)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    emps_stems = read_stems("data/datasets/EMPS/test.txt", strip_ext=True)
    bam_stems = read_stems("data/datasets/BAM_TiO2/test.txt", strip_ext=False)
    if args.n_emps:
        emps_stems = [emps_stems[i] for i in rng.choice(len(emps_stems), args.n_emps, replace=False)]

    data = {
        "emps": dict(stems=emps_stems, load=load_emps, gt=emps_gt_instances),
        "bam": dict(stems=bam_stems, load=load_bam, gt=bam_tio2_gt_instances),
    }
    all_stems = {k: read_stems(f"data/datasets/{'EMPS' if k == 'emps' else 'BAM_TiO2'}/test.txt", k == "emps") for k in data}

    instance_seg = InstanceSegmenter(config.MODELS_DIR / "instance_segmenter.pth", "cuda")
    sam = SamSegmenter(config.SAM_CHECKPOINT, config.SAM_MODEL_TYPE, config.DEVICE)

    def prep(img):  # the app's default preprocessing for the SAM branch
        return preprocess(img, denoise_on=True, equalize_on=True, contrast_on=False)

    # per-target cache: raw/processed image, gt, candidates, IoU matrix, no-filter hits
    cache = {}

    def target(ds, stem):
        key = (ds, stem)
        if key not in cache:
            raw = data[ds]["load"](stem)
            gt = data[ds]["gt"](stem if ds == "emps" else stem)
            cands = instance_seg.predict(raw)
            cache[key] = dict(
                raw=raw, processed=prep(raw), gt=gt, cands=cands,
                iou=iou_matrix(cands, gt) if len(cands) else np.zeros((0, len(gt))),
            )
        return cache[key]

    def pick_other(ds, stem):
        pool = [s for s in all_stems[ds] if s != stem]
        return pool[int(rng.integers(len(pool)))]

    def reference_pair(ref_ds, ref_stem, n_ref):
        ref = target(ref_ds, ref_stem)
        masks = median_reference(ref["gt"], ref["raw"].shape) if n_ref == "n1" else pick_reference_masks(ref["gt"], ref["raw"].shape)
        return ref["processed"], masks

    def scenario_items(tds, rds, how, n_ref):
        for stem in data[tds]["stems"]:
            if how == "self":
                ref_ds, ref_stem = tds, stem
            else:
                ref_ds = rds
                ref_stem = pick_other(ref_ds, stem if ref_ds == tds else None)
            yield target(tds, stem), *reference_pair(ref_ds, ref_stem, n_ref)

    mosaics = []
    if args.mosaic_pairs:
        for k in range(args.mosaic_pairs):
            e_stem = all_stems["emps"][int(rng.integers(len(all_stems["emps"])))]
            b_stem = all_stems["bam"][int(rng.integers(len(all_stems["bam"])))]
            mosaics.append(build_mosaic(target("emps", e_stem), target("bam", b_stem), rng, instance_seg, prep))

    def mosaic_items(want, n_ref):
        for m in mosaics:
            ref_ds = want
            ref_stem = pick_other(ref_ds, None)
            yield dict(m, gt=m[f"gt_{want}"], iou=m[f"iou_{want}"]), *reference_pair(ref_ds, ref_stem, n_ref)

    scenarios = []
    for n_ref in args.ref_sizes:
        scenarios += [
            (f"self / {n_ref}: EMPS", lambda n=n_ref: scenario_items("emps", "emps", "self", n)),
            (f"cross-same / {n_ref}: EMPS", lambda n=n_ref: scenario_items("emps", "emps", "other", n)),
            (f"self / {n_ref}: BAM_TiO2", lambda n=n_ref: scenario_items("bam", "bam", "self", n)),
            (f"cross-same / {n_ref}: BAM_TiO2", lambda n=n_ref: scenario_items("bam", "bam", "other", n)),
            (f"cross-other / {n_ref}: EMPS ref -> BAM_TiO2", lambda n=n_ref: scenario_items("bam", "emps", "other", n)),
            (f"cross-other / {n_ref}: BAM_TiO2 ref -> EMPS", lambda n=n_ref: scenario_items("emps", "bam", "other", n)),
        ]
        if mosaics:
            scenarios += [
                (f"mosaic / {n_ref}: EMPS | BAM_TiO2 side by side, want EMPS only", lambda n=n_ref: mosaic_items("emps", n)),
                (f"mosaic / {n_ref}: EMPS | BAM_TiO2 side by side, want BAM_TiO2 only", lambda n=n_ref: mosaic_items("bam", n)),
            ]

    t0 = time.time()
    for name, make_items in scenarios:
        n_gt = np.zeros(len(THRESHOLDS))
        n_kept = np.zeros(len(THRESHOLDS))
        n_hit = np.zeros(len(THRESHOLDS))
        pooled_scores, pooled_tp = [], []
        n_pairs = n_skipped = n_bad_refs = 0
        for tg, ref_processed, ref_masks in make_items():
            if len(tg["cands"]) == 0 or not tg["gt"]:
                n_gt += len(tg["gt"])
                continue
            det = FewShotDetector(sam)
            for m in ref_masks:
                try:
                    det.set_reference(ref_processed, m)
                except ValueError:  # thin mask vanishes at SAM's 64x64 feature grid: unusable as a click, skip it
                    n_bad_refs += 1
            if det.n_references == 0:
                n_skipped += 1
                continue
            sims = det.score_masks(tg["processed"], tg["cands"])
            _, base_hit = match_count(tg["iou"], np.ones(len(sims), bool))
            pooled_scores.append(sims)
            pooled_tp.append(base_hit)
            for t, thr in enumerate(THRESHOLDS):
                keep = sims >= thr
                hits, _ = match_count(tg["iou"], keep)
                n_gt[t] += len(tg["gt"])
                n_kept[t] += keep.sum()
                n_hit[t] += hits
            n_pairs += 1
        scores = np.concatenate(pooled_scores)
        tp = np.concatenate(pooled_tp)
        print(f"\n[{name}] pairs={n_pairs}  candidates={len(scores)}  (candidates matching a wanted particle: {tp.mean():.1%})")
        if n_skipped or n_bad_refs:
            print(f"  ({n_bad_refs} reference mask(s) vanished at the 64x64 feature grid and were dropped; {n_skipped} pair(s) had none left and were skipped)")
        print(f"  similarity: matched-candidate median {np.median(scores[tp]):.3f} vs unmatched {np.median(scores[~tp]):.3f}; AUC {auc(scores, tp):.3f}")
        print(f"  {'thr':>5} {'kept':>7} {'recall':>8} {'precision':>10} {'F1':>7}")
        for t, thr in enumerate(THRESHOLDS):
            rec = n_hit[t] / n_gt[t] if n_gt[t] else 0.0
            prec = n_hit[t] / n_kept[t] if n_kept[t] else 0.0
            f = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
            tag = "  <- no filter" if thr == 0 else ("  <- app default" if thr == 0.78 else "")
            print(f"  {thr:>5.2f} {n_kept[t] / max(n_kept[0], 1):>6.0%} {rec:>8.1%} {prec:>10.1%} {f:>7.3f}{tag}")
        print(f"  (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
