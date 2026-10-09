"""Tables and figures for experiment 02 from results/raw/*_test.pkl.

  python src/summarize.py --methods diffusion_t300_s12 sam_h sam_h_recall sam_b felz
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import style  # noqa: E402
import metrics as M  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
GT = {"stuff_c34": "construction land (NBBG class 34, public)",
      "actief_internal": "active construction sites (CBS-verified, internal)",
      "things_bag": "buildings (BAG, public)"}
NAME = {"felz": "Felzenszwalb (no deep learning)", "sam_b": "SAM ViT-B", "sam_h": "SAM ViT-H",
        "sam_h_recall": "SAM ViT-H, loose filters"}
COL = {"felz": style.MUTED, "sam_b": "#d39a8c", "sam_h": style.CLAY, "sam_h_recall": "#7a2f22"}


def name(m):
    return NAME.get(m, "Diffusion attention (my Diffuse2Seg reading)")


def color(m):
    return COL.get(m, style.TEAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    a = ap.parse_args()
    style.apply()
    runs = {m: pickle.load(open(HERE / "results" / "raw" / f"{m}_test.pkl", "rb"))["tiles"] for m in a.methods}

    table = {}
    for m, tiles in runs.items():
        row = {"sec_per_tile": float(np.mean([r["sec"] for r in tiles.values()])),
               "masks_per_tile": float(np.mean([len(r["scores"]) for r in tiles.values()]))}
        for g in GT:
            if not any(g in r["iou"] for r in tiles.values()):
                continue
            row[g] = {f"AR@{k}": M.ar(tiles, g, k) for k in (10, 100, 1000)}
            row[g]["AR@100_ci95"] = M.ar_ci(tiles, g, 100)
            row[g]["R50@100"] = M.recall_at(tiles, g, 100, 0.5)
            row[g]["mean_best_iou@1000"] = float(M.best_iou(tiles, g, 1000).mean())
            row[g]["n_objects"] = int(len(M.best_iou(tiles, g, 1000)))
        table[m] = row
    # Recall by building size (pixels at 1 m = m2).
    for m, tiles in runs.items():
        sizes = {"200-500 m²": (200, 500), "500-2,000 m²": (500, 2000), ">2,000 m²": (2000, 1e9)}
        out = {}
        for lab, (lo, hi) in sizes.items():
            b = []
            for tid, r in tiles.items():
                gt = np.load(HERE / "data" / "tiles" / f"{tid}_gt.npz")["things_bag"]
                if gt.max() == 0:
                    continue
                area = np.bincount(gt.ravel(), minlength=gt.max() + 1)[1:]
                best = r["iou"]["things_bag"][:100].max(0) if len(r["iou"]["things_bag"][:100]) else np.zeros(len(area))
                sel = (area >= lo) & (area < hi)
                b.append(best[sel])
            b = np.concatenate(b)
            out[lab] = {"n": int(len(b)), "R50@100": float((b >= 0.5).mean()) if len(b) else float("nan")}
        table[m]["bag_by_size"] = out

    (HERE / "results" / "summary.json").write_text(json.dumps(table, indent=1))
    for m, r in table.items():
        print(f"\n{name(m)} [{m}]  {r['sec_per_tile']:.1f} s/tile, {r['masks_per_tile']:.0f} masks/tile")
        for g in GT:
            if g in r:
                x = r[g]
                print(f"  {g:16s} n={x['n_objects']:4d} AR@10 {x['AR@10']:.3f} AR@100 {x['AR@100']:.3f} "
                      f"[{x['AR@100_ci95'][0]:.3f},{x['AR@100_ci95'][1]:.3f}] AR@1000 {x['AR@1000']:.3f} "
                      f"R50@100 {x['R50@100']:.3f}")
        print("  buildings R50@100 by size:", {k: round(v["R50@100"], 3) for k, v in r["bag_by_size"].items()})

    # AR vs budget, one panel per ground truth.
    ks = np.unique(np.round(np.logspace(0, 3, 25)).astype(int))
    gts = [g for g in GT if all(g in r["iou"] for t in runs.values() for r in list(t.values())[:1])]
    fig, axs = plt.subplots(1, len(gts), figsize=(5.2 * len(gts), 4.2), sharey=True)
    for ax, g in zip(np.atleast_1d(axs), gts):
        for m, tiles in runs.items():
            ax.plot(ks, [M.ar(tiles, g, k) for k in ks], color=color(m), lw=2, label=name(m))
        ax.set_xscale("log")
        ax.set(xlabel="proposals kept per tile", title=GT[g])
    np.atleast_1d(axs)[0].set_ylabel("average recall (IoU 0.5–0.95)")
    np.atleast_1d(axs)[0].legend(fontsize=8.5, loc="upper left")
    style.save(fig, HERE / "figures" / "ar_vs_budget.png")

    # Showcase: public ground truth only (no internal geometry is drawn).
    shows = {}
    for m in a.methods:
        p = HERE / "data" / f"showcase_{m}.pkl"
        if p.exists():
            shows[m] = pickle.load(open(p, "rb"))
    pick = [m for m in shows if m.startswith("diffusion")][:1] + [m for m in ("sam_h",) if m in shows]
    tids = sorted(set.intersection(*[set(shows[m]) for m in pick]))[:4] if pick else []
    if tids:
        fig, axs = plt.subplots(len(tids), 1 + len(pick), figsize=(4.2 * (1 + len(pick)), 4.2 * len(tids)))
        for i, tid in enumerate(tids):
            img = np.asarray(Image.open(HERE / "data" / "tiles" / f"{tid}.png").convert("RGB"))
            gt = np.load(HERE / "data" / "tiles" / f"{tid}_gt.npz")
            ax = axs[i, 0]
            ax.imshow(img)
            ax.contour(gt["stuff_c34"] > 0, [0.5], colors=["#ffb000"], linewidths=1.8)
            ax.contour(gt["things_bag"] > 0, [0.5], colors=["white"], linewidths=0.6)
            ax.set_title("PDOK 2022 · class 34 (orange) · BAG (white)" if i == 0 else "", fontsize=9)
            for j, m in enumerate(pick):
                ms = np.unpackbits(shows[m][tid], axis=-1)[..., :img.shape[1]].astype(bool)[:25]
                over = img.astype(float) / 255 * 0.55
                rng = np.random.default_rng(0)
                for mk in ms[::-1]:
                    over[mk] = 0.45 * over[mk] + 0.55 * rng.uniform(0.2, 1, 3)
                axs[i, j + 1].imshow(over)
                axs[i, j + 1].set_title(f"{name(m)}: top 25" if i == 0 else "", fontsize=9)
        for ax in axs.ravel():
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
        fig.text(0.5, 0.0, "Tiles are 512 × 512 m at 1 m. Imagery © Beeldmateriaal Nederland / PDOK (CC BY 4.0).",
                 ha="center", fontsize=9)
        style.save(fig, HERE / "figures" / "showcase.png")


if __name__ == "__main__":
    main()
