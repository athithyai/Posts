"""Figures for experiment 01, from results/ and data/maps.npz.

White background, greys, one accent colour (the embedding). Lines are labelled
directly; no legends where a label next to the line will do.

  python src/figures.py --s2 data/s2_2020_scenes.npz
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent.parent
NAME = {"linear_s2": "Linear model on Sentinel-2 features", "knn_spheer": "Nearest neighbours on the embedding",
        "rf_spheer": "Random forest on the embedding", "rf_both": "Random forest on both",
        "linear_spheer": "Linear model on the embedding"}


def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", required=True)
    a = ap.parse_args()
    style.apply()
    res = json.loads((HERE / "results" / "metrics.json").read_text())
    fig_dir = HERE / "figures"
    for old in ("macro_f1.png", "spheer_pca.png"):          # replaced by simpler figures
        (fig_dir / old).unlink(missing_ok=True)

    # 1. The honest comparison: paired difference from the random-forest baseline.
    d = res["paired_vs_rf_s2"]
    order = sorted(d, key=lambda m: d[m]["delta"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axvline(0, color=style.INK, lw=0.8)
    for i, m in enumerate(order):
        lo, hi = d[m]["ci95"]
        col = style.ACCENT if lo > 0 else style.GREY
        ax.plot([lo, hi], [i, i], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(d[m]["delta"], i, "o", color=col, ms=7)
        ax.text(hi + 0.004, i, f"{d[m]['delta']:+.3f}", va="center", fontsize=9, color=col)
    ax.set_yticks(range(len(order)), [NAME[m] for m in order])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("difference in macro-F1 from a random forest on Sentinel-2 features (95% CI)")
    ax.set_title("All five alternatives score higher; three of them reliably")
    ax.text(0.0, -1.25, "random forest baseline: macro-F1 "
            f"{res['models']['rf_s2']['macro_f1']:.3f}", fontsize=8.5, color=style.GREY, ha="center")
    style.save(fig, fig_dir / "paired_difference.png")

    # 2. Labels needed: the baseline vs the simplest use of the embedding.
    c = pd.read_csv(HERE / "results" / "label_efficiency.csv")
    g = c.groupby(["model", "per_class"]).macro_f1.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for m, col, lab in (("rf_s2", style.CLAY, "random forest,\nSentinel-2 features"),
                        ("linear_spheer", style.ACCENT, "linear model,\nembedding")):
        ax.plot(g.index, g[m], color=col, lw=2.4, marker="o", ms=4)
        ax.text(g.index[-1] * 1.15, g[m].iloc[-1], lab, color=col, va="center", fontsize=9.5)
    ax.plot([50, 1000], [g["rf_s2"][1000]] * 2, color=style.INK, lw=0.8, ls=":")
    ax.plot([50], [g["linear_spheer"][50]], "o", ms=10, mfc="none", mec=style.INK, mew=1)
    ax.text(160, g["rf_s2"][1000] - 0.022,
            f"50 labels per class on the embedding ({g['linear_spheer'][50]:.3f})\n"
            f"beat 1,000 per class on Sentinel-2 ({g['rf_s2'][1000]:.3f})", fontsize=8.5, ha="center")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100, 250, 1000], ["5", "10", "25", "50", "100", "250", "1,000"])
    ax.set_xlim(4, 6000)
    ax.set(xlabel="labelled pixels per class", ylabel="macro-F1 (mean of 5 draws)")
    ax.set_title("The gap is steady: it does not grow when labels are scarce")
    style.save(fig, fig_dir / "label_efficiency.png")

    # 3. Per class.
    pc = pd.read_csv(HERE / "results" / "per_class_f1.csv").pivot(index="class", columns="model", values="f1")
    delta = (pc["linear_spheer"] - pc["rf_s2"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(delta) + 1.0))
    ax.barh(range(len(delta)), delta.values, height=0.6,
            color=[style.TEAL if v > 0 else style.CLAY for v in delta.values])
    ax.set_yticks(range(len(delta)), delta.index, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=style.INK, lw=0.8)
    ax.spines["left"].set_visible(False)
    for i, v in enumerate(delta.values):
        if abs(v) > 0.03:
            ax.text(v + (0.006 if v > 0 else -0.006), i, f"{v:+.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=8.5)
    ax.set_xlim(min(delta.min() - 0.06, -0.05), delta.max() + 0.05)
    ax.set_xlabel("F1 of the embedding minus F1 of the forest")
    ax.set_title("It helps where use and layout matter, and hurts on wet nature")
    style.save(fig, fig_dir / "per_class_delta.png")

    # 4. One map: where the embedding fixed or broke the forest's answer.
    mp = np.load(HERE / "data" / "maps.npz")
    lab, valid = mp["label"], mp["valid"]
    rf_ok = (mp["rf_s2"] == lab) & valid
    em_ok = (mp["linear_spheer"] == lab) & valid
    land = valid & (lab != 70)
    h, w = lab.shape
    rr, cc = np.nonzero(land[h // 3:, w // 3:])
    crop = (slice(max(rr.min() + h // 3 - 5, 0), None), slice(max(cc.min() + w // 3 - 5, 0), None))
    code = np.zeros(lab.shape, np.uint8)                   # 0 not scored (white)
    code[valid & rf_ok & em_ok] = 1                        # both right: very light grey
    code[valid & ~rf_ok & ~em_ok] = 2                      # both wrong: mid grey
    code[valid & rf_ok & ~em_ok] = 3                       # embedding broke it: black
    code[valid & ~rf_ok & em_ok] = 4                       # embedding fixed it: accent
    cmap = ListedColormap([style.WHITE, style.FAINT, "#c3cacb", style.CLAY, style.TEAL])

    z = np.load(a.s2, allow_pickle=True)
    months = np.array([int(dd[5:7]) for dd in z["dates"]])
    x = z["x"][(months >= 6) & (months <= 8)].astype(np.float32)
    rgb = np.nanmedian(x[:, [2, 1, 0]], axis=0).transpose(1, 2, 0)
    rgb = np.dstack([stretch(rgb[..., i]) for i in range(3)])

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(rgb[crop])
    axs[0].set_title("Lelystad, Sentinel-2 summer 2020", fontsize=10.5)
    axs[1].imshow(code[crop], cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    sub = code[crop]
    fixed, broke = int((sub == 4).sum()), int((sub == 3).sum())
    axs[1].set_title(f"Embedding vs forest, pixel by pixel", fontsize=10.5)
    for ax in axs:
        clean(ax)
    keys = [(style.TEAL, f"embedding right, forest wrong ({fixed:,} px)"),
            (style.CLAY, f"forest right, embedding wrong ({broke:,} px)"),
            ("#c3cacb", "both wrong"), (style.FAINT, "both right")]
    for k, (col, t) in enumerate(keys):
        axs[1].add_patch(plt.Rectangle((1.03, 0.92 - k * 0.07), 0.04, 0.04, transform=axs[1].transAxes,
                                       color=col, clip_on=False, ec="#cccccc", lw=0.5))
        axs[1].text(1.09, 0.94 - k * 0.07, t, transform=axs[1].transAxes, va="center", fontsize=9)
    axs[1].text(1.03, 0.55, "The lake is light grey: both\nmodels get water right.\nWhite: mixed pixels at class\n"
                "borders, not scored.\nEvery prediction is out-of-fold.",
                transform=axs[1].transAxes, fontsize=8.5, color=style.GREY, va="top")
    style.save(fig, fig_dir / "maps.png")


if __name__ == "__main__":
    main()
