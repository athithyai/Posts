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
GT = {"stuff_c34": "Construction land (public map)",
      "actief_internal": "Active construction sites (verified, internal)",
      "things_bag": "Buildings (public register)"}
NAME = {"felz": "Felzenszwalb superpixels", "sam_b": "SAM ViT-B", "sam_h": "SAM ViT-H",
        "sam_h_recall": "SAM ViT-H, loose filters", "diffusion_t100_sinf": "Diffusion, no edge term"}
STYLE = {"felz": (style.GREY, "-"), "sam_b": ("#cf9a8e", "-"), "sam_h": (style.CLAY, "-"),
         "sam_h_recall": (style.CLAY, "--"), "diffusion_t100_sinf": (style.TEAL, ":")}


def name(m):
    return NAME.get(m, "Diffusion attention")


def color(m):
    return STYLE.get(m, (style.TEAL, "-"))[0]


def dash(m):
    return STYLE.get(m, (style.TEAL, "-"))[1]


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
        sizes = {"200–500 m²": (200, 500), "500–2,000 m²": (500, 2000), "over 2,000 m²": (2000, 1e9)}
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

    # Paired comparisons that the post leans on.
    pairs = [("diffusion_t100_s32", "sam_h"), ("diffusion_t100_s32", "sam_h_recall"),
             ("diffusion_t100_s32", "diffusion_t100_sinf"), ("diffusion_t100_s32", "felz")]
    paired = {}
    for x, y in pairs:
        if x in runs and y in runs:
            paired[f"{x} - {y}"] = {g: {f"AR@{k}": M.paired_ar_delta(runs[x], runs[y], g, k) for k in (100, 1000)}
                                    for g in GT if g in next(iter(runs[x].values()))["iou"]}
    table["paired"] = paired
    for pk, v in paired.items():
        print(pk)
        for g, d in v.items():
            print("   ", g, {k: f"{x['delta']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" for k, x in d.items()})
    (HERE / "results" / "summary.json").write_text(json.dumps(table, indent=1))
    for m, r in ((m, r) for m, r in table.items() if m != "paired"):
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
    fig, axs = plt.subplots(1, len(gts), figsize=(5.0 * len(gts), 4.4), sharey=True)
    for ax, g in zip(np.atleast_1d(axs), gts):
        for m, tiles in runs.items():
            main_line = m in ("sam_h", "felz") or (m.startswith("diffusion") and m != "diffusion_t100_sinf")
            ax.plot(ks, [M.ar(tiles, g, k) for k in ks], color=color(m), ls=dash(m),
                    lw=2.4 if main_line else 1.4, label=name(m))
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        ax.set_title(GT[g], fontsize=11.5)
        ax.set_xlabel("masks kept per tile")
    np.atleast_1d(axs)[0].set_ylabel("average recall (IoU 0.5 to 0.95)")
    np.atleast_1d(axs)[0].legend(loc="upper left", fontsize=8.5)
    style.save(fig, HERE / "figures" / "ar_vs_budget.png")

    # Buildings by size: where the diffusion grid runs out of resolution.
    d_key = next((m for m in runs if m.startswith("diffusion") and m != "diffusion_t100_sinf"), None)
    if d_key and "sam_h" in runs:
        sizes = list(table[d_key]["bag_by_size"])
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        x = np.arange(len(sizes))
        for off, m in ((-0.18, d_key), (0.18, "sam_h")):
            v = [table[m]["bag_by_size"][s]["R50@100"] for s in sizes]
            ax.bar(x + off, v, width=0.34, color=color(m), label=name(m))
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 0.015, f"{vi:.0%}", ha="center", fontsize=9, color=color(m))
        ns = [table[d_key]["bag_by_size"][s]["n"] for s in sizes]
        ax.set_xticks(x, [f"{s}\n({n:,} buildings)" for s, n in zip(sizes, ns)])
        ax.set_ylim(0, 0.8)
        ax.set_yticks([0, 0.2, 0.4, 0.6], ["0%", "20%", "40%", "60%"])
        ax.set_ylabel("buildings found (IoU ≥ 0.5, top 100)")
        ax.legend(loc="upper left")
        ax.set_title("Small buildings fall between the diffusion model's 8 m cells")
        style.save(fig, HERE / "figures" / "buildings_by_size.png")

    # Showcase: public ground truth only (no internal geometry is drawn).
    shows = {}
    for m in a.methods:
        p = HERE / "data" / f"showcase_{m}.pkl"
        if p.exists():
            shows[m] = pickle.load(open(p, "rb"))
    pick = [m for m in shows if m.startswith("diffusion") and m != "diffusion_t100_sinf"][:1] + \
        [m for m in ("sam_h",) if m in shows]
    tids = sorted(set.intersection(*[set(shows[m]) for m in pick]))[:4] if pick else []
    if tids:
        n = len(tids)
        fig, axs = plt.subplots(n, 1 + len(pick), figsize=(4.0 * (1 + len(pick)), 4.0 * n))
        for i, tid in enumerate(tids):
            img = np.asarray(Image.open(HERE / "data" / "tiles" / f"{tid}.png").convert("RGB"))
            grey = np.repeat(img.mean(2, keepdims=True), 3, 2) / 255
            soft = 0.35 + 0.6 * grey                       # dimmed greyscale so outlines carry the colour
            gt = np.load(HERE / "data" / "tiles" / f"{tid}_gt.npz")
            axs[i, 0].imshow(img)
            axs[i, 0].contour(gt["stuff_c34"] > 0, [0.5], colors=["white"], linewidths=2.2)
            axs[i, 0].contour(gt["things_bag"] > 0, [0.5], colors=["white"], linewidths=0.5)
            for j, m in enumerate(pick):
                ms = np.unpackbits(shows[m][tid], axis=-1)[..., :img.shape[1]].astype(bool)[:25]
                axs[i, j + 1].imshow(soft)
                for mk in ms:
                    axs[i, j + 1].contour(mk, [0.5], colors=[color(m)], linewidths=1.0)
        heads = ["Aerial photo with the map\n(thick: construction land, thin: buildings)"] + \
            [f"{name(m)}\n(top 25 masks)" for m in pick]
        for j, h in enumerate(heads):
            axs[0, j].set_title(h, fontsize=10.5, color=style.INK if j == 0 else color(pick[j - 1]))
        for ax in axs.ravel():
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.text(0.5, 0.005, "Each tile is 512 × 512 m at 1 m. Imagery: Beeldmateriaal Nederland / PDOK, CC BY 4.0.",
                 ha="center", fontsize=9, color=style.GREY)
        style.save(fig, HERE / "figures" / "showcase.png")


if __name__ == "__main__":
    main()
