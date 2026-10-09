"""Figures for experiment 01, from results/ and data/maps.npz.

  python src/figures.py --spheer <...>/markermeer_31UFU_2020_ov1.npz --s2 data/s2_2020_scenes.npz
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
LABEL = {"rf_s2": "Random forest · Sentinel-2 features", "linear_s2": "Linear · Sentinel-2 features",
         "rf_spheer": "Random forest · Spheer embedding", "linear_spheer": "Linear · Spheer embedding",
         "knn_spheer": "k-NN · Spheer embedding", "rf_both": "Random forest · both"}
COLOR = {"rf_s2": style.CLAY, "linear_s2": "#d39a8c", "rf_spheer": style.TEAL,
         "linear_spheer": style.LEAF, "knn_spheer": "#8fb592", "rf_both": style.INK}


def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spheer", required=True)
    ap.add_argument("--s2", required=True)
    a = ap.parse_args()
    style.apply()
    res = json.loads((HERE / "results" / "metrics.json").read_text())
    fig_dir = HERE / "figures"

    # 1. Headline bars with block-bootstrap CIs.
    order = sorted(res["models"], key=lambda m: res["models"][m]["macro_f1"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, m in enumerate(order):
        r = res["models"][m]
        lo, hi = r["macro_f1_ci95"]
        ax.barh(i, r["macro_f1"], color=COLOR[m], height=0.62)
        ax.errorbar(r["macro_f1"], i, xerr=[[r["macro_f1"] - lo], [hi - r["macro_f1"]]],
                    color=style.INK, capsize=3, lw=1)
        ax.text(hi + 0.01, i, f"{r['macro_f1']:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    ax.set_xlim(0, min(1, max(res["models"][m]["macro_f1_ci95"][1] for m in order) + 0.12))
    ax.set_xlabel(f"macro-F1 over {len(res['classes'])} CBS land-use classes (95% CI, 1 km block bootstrap)")
    ax.set_title("Same pixels, same year, same labels")
    ax.grid(axis="y", visible=False)
    style.save(fig, fig_dir / "macro_f1.png")

    # 2. Label efficiency.
    p = HERE / "results" / "label_efficiency.csv"
    if p.exists():
        c = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for m in ("rf_s2", "rf_spheer", "linear_spheer", "knn_spheer"):
            g = c[c.model == m].groupby("per_class").macro_f1
            mu, sd = g.mean(), g.std()
            ax.plot(mu.index, mu.values, marker="o", ms=4, color=COLOR[m], label=LABEL[m], lw=2)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=COLOR[m], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set(xlabel="labelled pixels per class (log scale)", ylabel="macro-F1",
               title="How many labels does each approach need?")
        ax.legend(fontsize=8.5, loc="lower right")
        style.save(fig, fig_dir / "label_efficiency.png")

    # 3. Per-class difference: where does the embedding help or hurt?
    pc = pd.read_csv(HERE / "results" / "per_class_f1.csv")
    w = pc.pivot(index="class", columns="model", values="f1")
    d = (w["linear_spheer"] - w["rf_s2"]).sort_values()
    px = {res["class_names"][str(k)] if str(k) in res["class_names"] else k: v
          for k, v in res["class_pixels"].items()}
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(d) + 1.2))
    ax.barh(range(len(d)), d.values, color=[style.LEAF if v > 0 else style.CLAY for v in d.values], height=0.65)
    ax.set_yticks(range(len(d)), [f"{n}  ({px.get(n, 0):,} px)" for n in d.index], fontsize=8.5)
    ax.axvline(0, color=style.INK, lw=0.8)
    ax.set_xlabel("F1(linear on Spheer) − F1(random forest on Sentinel-2)")
    ax.set_title("Per class: green = embedding wins")
    ax.grid(axis="y", visible=False)
    style.save(fig, fig_dir / "per_class_delta.png")

    # 4. Maps.
    mp = np.load(HERE / "data" / "maps.npz")
    lab, valid = mp["label"], mp["valid"]
    z = np.load(a.s2, allow_pickle=True)
    months = np.array([int(d[5:7]) for d in z["dates"]])
    x = z["x"][(months >= 6) & (months <= 8)].astype(np.float32)
    rgb = np.nanmedian(x[:, [2, 1, 0]], axis=0).transpose(1, 2, 0)
    rgb = np.dstack([stretch(rgb[..., i]) for i in range(3)])
    E = np.load(a.spheer, allow_pickle=True)["a"].astype(np.float32)
    f = E.reshape(100, -1).T
    f = f - np.nanmean(f, 0)
    _, _, vt = np.linalg.svd(np.nan_to_num(f[::7]), full_matrices=False)
    pca = (np.nan_to_num(f) @ vt[:3].T).reshape(*E.shape[1:], 3)
    pca = np.dstack([stretch(pca[..., i]) for i in range(3)])

    classes = res["classes"]
    rng = np.random.default_rng(3)
    cols = rng.uniform(0.25, 0.9, (len(classes), 3))
    named = {20: "#b5523b", 51: "#d9c77c", 60: "#2f6b3a", 62: "#7fb38a", 70: "#2f6f8f", 78: "#6fa8c7",
             40: "#8cc56b", 24: "#7a6c8f", 11: "#555555", 34: "#e3913b", 35: "#c1a58c", 61: "#b9c98a",
             75: "#4f93b8", 43: "#a7d08c", 41: "#b8d86b", 21: "#c97a6a", 22: "#a86a7a", 23: "#bd8aa0"}
    for i, c in enumerate(classes):
        if c in named:
            cols[i] = [int(named[c][k:k + 2], 16) / 255 for k in (1, 3, 5)]
    cmap = ListedColormap(np.vstack([[0.96, 0.96, 0.94], cols]))
    idx = {c: i + 1 for i, c in enumerate(classes)}
    to_idx = np.vectorize(lambda v: idx.get(int(v), 0))

    def show(arr):
        return np.where(valid, to_idx(arr), 0)

    # The scene is ~90% open water; the land is in the south-east corner.
    land = valid & (lab != 70)
    rr, cc = np.nonzero(land[land.shape[0] // 3:, land.shape[1] // 3:])
    r0, c0 = rr.min() + land.shape[0] // 3, cc.min() + land.shape[1] // 3
    crop = (slice(max(r0 - 5, 0), None), slice(max(c0 - 5, 0), None))
    sub = valid[crop]
    f1c = {m: f1_crop(lab[crop][sub], mp[m][crop][sub], classes) for m in ("rf_s2", "linear_spheer")}

    fig, axs = plt.subplots(1, 5, figsize=(18, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1, 1, 1]})
    axs[0].imshow(rgb)
    h, w = lab.shape
    axs[0].add_patch(plt.Rectangle((crop[1].start, crop[0].start), w - crop[1].start, h - crop[0].start,
                                   fill=False, ec="white", lw=1.5))
    axs[0].set_title("Full scene, summer 2020\n(box = zoom)", fontsize=10)
    panels = [(rgb[crop], "Zoom: Lelystad\n(Sentinel-2)", None),
              (show(lab)[crop], "CBS NBBG2020 label\n(pure pixels)", cmap),
              (show(mp["rf_s2"])[crop], f"RF on Sentinel-2\nmacro-F1 in zoom {f1c['rf_s2']:.2f}", cmap),
              (show(mp["linear_spheer"])[crop],
               f"Linear on Spheer\nmacro-F1 in zoom {f1c['linear_spheer']:.2f}", cmap)]
    for ax, (im, t, cm) in zip(axs[1:], panels):
        ax.imshow(im, cmap=cm, vmin=0, vmax=len(classes), interpolation="nearest") if cm else ax.imshow(im)
        ax.set_title(t, fontsize=10)
    for ax in axs:
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    from matplotlib.patches import Patch
    handles = [Patch(color=cmap(i + 1), label=res["class_names"][str(c)]) for i, c in enumerate(classes)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.08))
    fig.text(0.5, -0.12, "11.2 × 11.5 km at 20 m. Predictions are out-of-fold: every pixel is predicted by a "
             "model that never saw its 1 km block.", ha="center", fontsize=9)
    style.save(fig, fig_dir / "maps.png")

    # Spheer embedding as false colour, on its own.
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(pca)
    ax.set_title("Spheer 2020 embedding, first 3 principal components", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    style.save(fig, fig_dir / "spheer_pca.png")


def f1_crop(y, p, classes):
    from sklearn.metrics import f1_score
    present = [c for c in classes if (y == c).any()]
    return f1_score(y, p, labels=present, average="macro", zero_division=0)


if __name__ == "__main__":
    main()
