"""Run one proposer over the benchmark and keep only what scoring needs.

For every tile and every ground-truth set it stores the IoU of each of the top
1000 proposals with each object, plus proposal scores and areas. Average recall
at any budget can then be recomputed without re-running a GPU.

  python src/run.py --method diffusion --split dev --t 300 --sigma 12
  python src/run.py --method sam_h --split test
Splits: dev = first 20 tile ids (used to choose t and sigma), test = the rest.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proposers as P  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
TOP = 1000
N_DEV = 20
SHOWCASE = ("t020", "t021", "t022", "t023", "t024", "t025")


def iou_matrix(masks, gt):
    """IoU of each proposal with each GT instance (ids 1..G) via bincount."""
    G = int(gt.max())
    if G == 0 or not masks:
        return np.zeros((len(masks), G), np.float32)
    gsize = np.bincount(gt.ravel(), minlength=G + 1)[1:]
    out = np.zeros((len(masks), G), np.float32)
    for i, m in enumerate(masks):
        inter = np.bincount(gt[m], minlength=G + 1)[1:]
        out[i] = inter / np.maximum(m.sum() + gsize - inter, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["diffusion", "sam_h", "sam_b", "sam_h_recall", "felz"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--t", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=12.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    meta = json.loads((HERE / "data" / "tiles.json").read_text())
    ids = [r["id"] for r in meta["tiles"]]
    ids = ids[:N_DEV] if a.split == "dev" else ids[N_DEV:]

    if a.method == "diffusion":
        model = P.DiffusionAttention()
        run = lambda img: P.diffusion_masks(model, img, t=a.t, sigma=a.sigma)
    elif a.method.startswith("sam"):
        sam = P.SAM("facebook/sam-vit-base" if a.method == "sam_b" else "facebook/sam-vit-huge")
        if a.method == "sam_h_recall":   # looser filters: more masks, for budget-matched recall
            run = lambda img: _sam_loose(sam, img)
        else:
            run = sam
    else:
        run = P.felz_masks

    tag = a.tag or (f"{a.method}_t{a.t}_s{a.sigma:g}" if a.method == "diffusion" else a.method)
    rec, show, t0 = {}, {}, time.time()
    internal_dir = HERE / "data" / "internal"
    for n, tid in enumerate(ids):
        img = np.asarray(Image.open(HERE / "data" / "tiles" / f"{tid}.png").convert("RGB"))
        gt = dict(np.load(HERE / "data" / "tiles" / f"{tid}_gt.npz"))
        if (internal_dir / f"{tid}_actief.npz").exists():
            gt["actief_internal"] = np.load(internal_dir / f"{tid}_actief.npz")["actief"]
        ts = time.time()
        masks, scores = run(img)
        masks, scores = masks[:TOP], np.asarray(scores[:TOP], np.float32)
        rec[tid] = {"scores": scores, "area": np.array([m.sum() for m in masks], np.int32),
                    "sec": time.time() - ts,
                    "iou": {k: iou_matrix(masks, v.astype(np.int64)) for k, v in gt.items()}}
        if tid in SHOWCASE:
            show[tid] = np.packbits(np.stack(masks[:40]), axis=-1) if masks else None
        if n % 10 == 0:
            print(f"  {tag} {n + 1}/{len(ids)} {tid} {len(masks)} masks {rec[tid]['sec']:.1f}s", flush=True)

    out = HERE / "results" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}_{a.split}.pkl", "wb") as f:
        pickle.dump({"method": a.method, "args": vars(a), "tiles": rec}, f)
    if show:
        with open(HERE / "data" / f"showcase_{tag}.pkl", "wb") as f:
            pickle.dump(show, f)
    print(f"{tag} {a.split}: {len(ids)} tiles in {(time.time() - t0) / 60:.1f} min")


def _sam_loose(sam, img):
    from PIL import Image as _I
    out = sam.gen(_I.fromarray(img), points_per_batch=64, pred_iou_thresh=0.5, stability_score_thresh=0.7)
    masks = [np.asarray(m, bool) for m in out["masks"]]
    scores = np.asarray([float(s) for s in out["scores"]])
    o = np.argsort(-scores)
    return [masks[i] for i in o], scores[o]


if __name__ == "__main__":
    main()
