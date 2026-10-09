"""Choose the two free knobs of the diffusion proposer on the 20 dev tiles only.

t      noise timestep (out of 1000) at which self-attention is read
sigma  colour bandwidth of the edge-preserving term; inf switches it off,
       which doubles as the ablation "does edge preservation matter?"
Selection uses public ground truth only (class-34 and BAG), mean of AR@100.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
import proposers as P  # noqa: E402
from run import N_DEV, iou_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[8.0, 16.0, float("inf")])
    ap.add_argument("--out", default="tuning_dev.json")
    a = ap.parse_args()
    meta = json.loads((HERE / "data" / "tiles.json").read_text())
    ids = [r["id"] for r in meta["tiles"]][:N_DEV]
    data = [(np.asarray(Image.open(HERE / "data" / "tiles" / f"{i}.png").convert("RGB")),
             dict(np.load(HERE / "data" / "tiles" / f"{i}_gt.npz"))) for i in ids]
    model = P.DiffusionAttention()
    rows = []
    for t in a.ts:
        for sigma in a.sigmas:
            tiles = {}
            for tid, (img, gt) in zip(ids, data):
                masks, scores = P.diffusion_masks(model, img, t=t, sigma=sigma)
                tiles[tid] = {"iou": {k: iou_matrix(masks, v.astype(np.int64)) for k, v in gt.items()}}
            r = {"t": t, "sigma": sigma,
                 "ar100_c34": M.ar(tiles, "stuff_c34", 100), "ar100_bag": M.ar(tiles, "things_bag", 100)}
            r["objective"] = (r["ar100_c34"] + r["ar100_bag"]) / 2
            rows.append(r)
            print(f"  t={t:3d} sigma={sigma:>4} AR@100 c34 {r['ar100_c34']:.3f} bag {r['ar100_bag']:.3f}", flush=True)
    best = max(rows, key=lambda r: r["objective"])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / a.out).write_text(json.dumps({"grid": rows, "chosen": best}, indent=1, default=str))
    print(f"chosen t={best['t']} sigma={best['sigma']}")


if __name__ == "__main__":
    main()
