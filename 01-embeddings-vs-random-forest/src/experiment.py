"""Is a foundation-model embedding better than a random forest on raw pixels?

Same 11 x 11.5 km of the Lelystad coast, same year (2020), same 20 m grid:
  S2       46 hand-built features from 56 clear Sentinel-2 L2A dates
           (quarterly band medians, quarterly NDVI, annual NDVI p10/p90)
  Spheer   the 100-d annual embedding (Spheer FM Albatross), nothing else
Label: CBS NBBG2020 land use, rasterised to the grid; pixels with mixed labels
in their 3 x 3 neighbourhood are dropped. Validation: 5 folds of 1 km blocks
with a 100 m buffer removed from training, never a random pixel split.

  python src/experiment.py --spheer <...>/markermeer_31UFU_2020_ov1.npz \
      --s2 data/s2_2020_scenes.npz --nbbg2020 <...>/CBS_Publicatiebestand_NBBG2020_v3_def.gpkg
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import projfix  # noqa: E402,F401

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

HERE = Path(__file__).resolve().parent.parent
SEED = 24
BLOCK_PX = 50          # 1 km at 20 m
BUFFER_PX = 5          # 100 m kept out of training around every test block
N_FOLDS = 5
PER_CLASS = 2000       # balanced training sample per class per fold
MIN_PIXELS = 300       # a class needs this many pure pixels ...
MIN_BLOCKS = 3         # ... spread over this many blocks, or it is dropped
N_BOOT = 1000

# CBS BBG/NBBG land-use codes (English short names).
NAMES = {10: "railway", 11: "main road", 12: "airport", 20: "residential", 21: "retail & catering",
         22: "public facilities", 23: "socio-cultural", 24: "industry & business", 30: "landfill",
         31: "scrapyard", 32: "cemetery", 33: "mineral extraction", 34: "construction site",
         35: "other semi-built", 40: "park", 41: "sports grounds", 42: "allotments",
         43: "day recreation", 44: "holiday parks", 50: "greenhouses", 51: "agriculture",
         60: "forest", 61: "dry nature", 62: "wet nature", 70: "IJsselmeer/Markermeer",
         71: "closed sea arm", 72: "Rhine & Meuse", 73: "border lakes", 74: "reservoirs",
         75: "recreational water", 76: "extraction water", 77: "settling ponds", 78: "other inland water",
         80: "Wadden Sea", 81: "Eastern Scheldt", 82: "Western Scheldt", 83: "North Sea"}


# ── features ────────────────────────────────────────────────────────────────
def s2_features(path):
    z = np.load(path, allow_pickle=True)
    x = z["x"].astype(np.float32)                      # (T, 10, H, W), NaN = not clear
    months = np.array([int(d[5:7]) for d in z["dates"]])
    bands = list(z["bands"])
    r, n = bands.index("B04"), bands.index("B08")
    ndvi = (x[:, n] - x[:, r]) / np.clip(x[:, n] + x[:, r], 1e-4, None)
    annual = np.nanmedian(x, axis=0)
    feats, names = [], []
    for q in range(4):
        m = (months > 3 * q) & (months <= 3 * q + 3)
        med = np.nanmedian(x[m], axis=0)
        med = np.where(np.isfinite(med), med, annual)  # a quarter with no clear view
        feats.append(med)
        names += [f"{b}_Q{q + 1}" for b in bands]
        nd = np.nanmedian(ndvi[m], axis=0)
        feats.append(np.where(np.isfinite(nd), nd, np.nanmedian(ndvi, axis=0))[None])
        names.append(f"NDVI_Q{q + 1}")
    feats.append(np.nanpercentile(ndvi, [10, 90], axis=0))
    names += ["NDVI_p10", "NDVI_p90"]
    f = np.concatenate(feats).astype(np.float32)
    clear_obs = np.isfinite(x[:, 0]).sum(0)
    return f, names, clear_obs, len(z["dates"])


def spheer(path):
    z = np.load(path, allow_pickle=True)
    return z["a"].astype(np.float32), Affine(*z["t"][:6]), str(z["crs"])


def labels(nbbg, transform, shape, crs):
    lyr = pyogrio.list_layers(nbbg)[0][0]
    h, w = shape
    b = (transform.c, transform.f + transform.e * h, transform.c + transform.a * w, transform.f)
    bb = gpd.GeoSeries.from_xy([b[0], b[2]], [b[1], b[3]], crs=crs).to_crs(28992).total_bounds
    g = pyogrio.read_dataframe(nbbg, layer=lyr, bbox=tuple(bb)).to_crs(crs)
    col = [c for c in g.columns if c.upper().startswith("NBBG")][0]
    lab = rasterize(((geom, int(v)) for geom, v in zip(g.geometry, g[col])), out_shape=shape,
                    transform=transform, fill=0, dtype="int32")
    pure = (maximum_filter(lab, 3) == minimum_filter(lab, 3)) & (lab > 0)
    return lab, pure


# ── validation design ───────────────────────────────────────────────────────
def block_folds(shape, seed=SEED):
    h, w = shape
    by, bx = np.mgrid[0:h, 0:w]
    bid = (by // BLOCK_PX) * ((w + BLOCK_PX - 1) // BLOCK_PX) + bx // BLOCK_PX
    u = np.unique(bid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(u)
    fold_of = {b: i % N_FOLDS for i, b in enumerate(perm)}
    return bid, np.vectorize(fold_of.get)(bid)


def train_mask(folds, k):
    """Everything not in fold k, minus a buffer around fold k."""
    near = maximum_filter((folds == k).astype(np.uint8), size=2 * BUFFER_PX + 1).astype(bool)
    return ~near


def sample(idx, y, per_class, rng):
    out = []
    for c in np.unique(y[idx]):
        ii = idx[y[idx] == c]
        out.append(rng.choice(ii, min(per_class, len(ii)), replace=False))
    return np.concatenate(out)


MODELS = {
    "rf_s2": ("s2", lambda: RandomForestClassifier(500, n_jobs=-1, random_state=SEED, min_samples_leaf=1)),
    "linear_s2": ("s2", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))),
    "rf_spheer": ("spheer", lambda: RandomForestClassifier(500, n_jobs=-1, random_state=SEED)),
    "linear_spheer": ("spheer", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))),
    "knn_spheer": ("spheer", lambda: make_pipeline(Normalizer(), KNeighborsClassifier(15, weights="distance"))),
    "rf_both": ("both", lambda: RandomForestClassifier(500, n_jobs=-1, random_state=SEED)),
}


def boot_ci(y, p, blocks, classes, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    u = np.unique(blocks)
    rows = {b: np.flatnonzero(blocks == b) for b in u}
    vals = []
    for _ in range(n):
        pick = np.concatenate([rows[b] for b in rng.choice(u, len(u), replace=True)])
        vals.append(f1_score(y[pick], p[pick], labels=classes, average="macro", zero_division=0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_delta(y, p_a, p_b, blocks, classes, n=N_BOOT, seed=SEED):
    """Bootstrap of F1(a) - F1(b) with the SAME resampled blocks for both, so
    block difficulty cancels and only the model difference is left."""
    rng = np.random.default_rng(seed)
    u = np.unique(blocks)
    rows = {b: np.flatnonzero(blocks == b) for b in u}
    d = []
    for _ in range(n):
        pick = np.concatenate([rows[b] for b in rng.choice(u, len(u), replace=True)])
        f = lambda p: f1_score(y[pick], p[pick], labels=classes, average="macro", zero_division=0)
        d.append(f(p_a) - f(p_b))
    d = np.array(d)
    return {"delta": float(f1_score(y, p_a, labels=classes, average="macro") -
                           f1_score(y, p_b, labels=classes, average="macro")),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_of_resamples_a_wins": float((d > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spheer", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--nbbg2020", required=True)
    ap.add_argument("--skip-curve", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    E, transform, crs = spheer(a.spheer)
    S, s2_names, clear_obs, n_dates = s2_features(a.s2)
    shape = E.shape[1:]
    lab, pure = labels(a.nbbg2020, transform, shape, crs)
    valid = pure & np.isfinite(E).all(0) & np.isfinite(S).all(0)
    bid, folds = block_folds(shape)

    # Class filter: enough pure pixels in enough distinct blocks.
    keep = []
    for c in np.unique(lab[valid]):
        m = valid & (lab == c)
        if m.sum() >= MIN_PIXELS and len(np.unique(bid[m])) >= MIN_BLOCKS:
            keep.append(int(c))
    dropped = sorted(set(np.unique(lab[lab > 0]).tolist()) - set(keep))
    valid &= np.isin(lab, keep)
    print(f"grid {shape}, {valid.sum()} pure labelled pixels, {len(keep)} classes kept: "
          f"{[NAMES.get(c, c) for c in keep]}; dropped {dropped}")

    flat = np.flatnonzero(valid.ravel())
    y = lab.ravel()[flat]
    Xs = {"s2": S.reshape(S.shape[0], -1)[:, flat].T, "spheer": E.reshape(100, -1)[:, flat].T}
    Xs["both"] = np.hstack([Xs["s2"], Xs["spheer"]])
    fold = folds.ravel()[flat]
    block = bid.ravel()[flat]
    tmask_full = {k: train_mask(folds, k).ravel()[flat] for k in range(N_FOLDS)}
    classes = np.array(keep)

    # ── main comparison ──
    rng = np.random.default_rng(SEED)
    oof = {m: np.zeros(len(y), np.int32) for m in MODELS}
    tr_idx = {k: sample(np.flatnonzero(tmask_full[k] & (fold != k)), y, PER_CLASS, rng) for k in range(N_FOLDS)}
    for name, (fs, make) in MODELS.items():
        t = time.time()
        for k in range(N_FOLDS):
            te = np.flatnonzero(fold == k)
            m = make().fit(Xs[fs][tr_idx[k]], y[tr_idx[k]])
            oof[name][te] = m.predict(Xs[fs][te])
        print(f"  {name:14s} macro-F1 {f1_score(y, oof[name], labels=classes, average='macro'):.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)

    res = {"grid": list(shape), "gsd_m": float(transform.a), "crs": crs, "n_s2_dates": int(n_dates),
           "s2_features": s2_names, "n_pixels": int(len(y)), "classes": keep,
           "class_names": {c: NAMES.get(c, str(c)) for c in keep}, "dropped_classes": dropped,
           "class_pixels": {int(c): int((y == c).sum()) for c in keep},
           "design": {"seed": SEED, "block_m": BLOCK_PX * 20, "buffer_m": BUFFER_PX * 20,
                      "folds": N_FOLDS, "per_class_train": PER_CLASS, "bootstrap": N_BOOT},
           "models": {}}
    per_class = []
    for name in MODELS:
        p = oof[name]
        res["models"][name] = {
            "macro_f1": float(f1_score(y, p, labels=classes, average="macro")),
            "macro_f1_ci95": boot_ci(y, p, block, classes),
            "overall_accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(np.mean([(p[y == c] == c).mean() for c in classes]))}
        for c, f in zip(classes, f1_score(y, p, labels=classes, average=None)):
            per_class.append({"model": name, "code": int(c), "class": NAMES.get(int(c), str(c)), "f1": float(f)})
    res["paired_vs_rf_s2"] = {name: paired_delta(y, oof[name], oof["rf_s2"], block, classes)
                              for name in MODELS if name != "rf_s2"}
    for name, d in res["paired_vs_rf_s2"].items():
        print(f"  {name:14s} - rf_s2 = {d['delta']:+.3f}  CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
              f"wins in {d['share_of_resamples_a_wins']:.0%} of resamples", flush=True)

    # ── label efficiency: how many labelled pixels per class does each need? ──
    curve = []
    if not a.skip_curve:
        for n in (5, 10, 25, 50, 100, 250, 1000):
            for rep in range(5):
                r2 = np.random.default_rng(SEED + 1000 * rep + n)
                preds = {m: np.zeros(len(y), np.int32) for m in ("rf_s2", "rf_spheer", "linear_spheer", "knn_spheer")}
                for k in range(N_FOLDS):
                    tri = sample(np.flatnonzero(tmask_full[k] & (fold != k)), y, n, r2)
                    te = np.flatnonzero(fold == k)
                    for m in preds:
                        fs, make = MODELS[m]
                        est = make()
                        if m == "knn_spheer":
                            est.set_params(kneighborsclassifier__n_neighbors=min(15, n))
                        preds[m][te] = est.fit(Xs[fs][tri], y[tri]).predict(Xs[fs][te])
                for m, p in preds.items():
                    curve.append({"model": m, "per_class": n, "rep": rep,
                                  "macro_f1": float(f1_score(y, p, labels=classes, average="macro"))})
            print(f"  curve n={n:5d} " + "  ".join(
                f"{m} {np.mean([c['macro_f1'] for c in curve if c['model'] == m and c['per_class'] == n]):.3f}"
                for m in ("rf_s2", "rf_spheer", "linear_spheer", "knn_spheer")), flush=True)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(per_class).to_csv(out / "per_class_f1.csv", index=False)
    if curve:
        pd.DataFrame(curve).to_csv(out / "label_efficiency.csv", index=False)

    # Maps for the figures; predictions are out-of-fold everywhere.
    maps = {"label": lab, "valid": valid}
    for m in ("rf_s2", "linear_spheer"):
        pm = np.zeros(lab.size, np.int32)
        pm[flat] = oof[m]
        maps[m] = pm.reshape(shape)
    np.savez_compressed(HERE / "data" / "maps.npz", **maps, folds=folds, clear_obs=clear_obs)

    # Shipping probe for Hugging Face: fitted on every pure pixel.
    idx = sample(np.arange(len(y)), y, PER_CLASS, rng)
    probe = MODELS["linear_spheer"][1]().fit(Xs["spheer"][idx], y[idx])
    import joblib
    (HERE / "model").mkdir(exist_ok=True)
    joblib.dump({"pipeline": probe, "classes": keep, "names": {c: NAMES.get(c) for c in keep}},
                HERE / "model" / "probe.joblib")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
