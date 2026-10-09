"""Does a confidence of 0.8 mean 80%? Calibration of construction-site candidate scores.

Input is the construction-site detector's candidate table for the 2024 test sheets (one row per
candidate polygon), auto-labelled by overlap with interpreter-verified active
construction sites. It is CBS-internal, so it is not in this repository; every
number in results/ is an aggregate of it.

  python src/calibrate.py --candidates <detector>/data/cand_2024_test_feat_scored.csv

Columns used: confidence (detector), rerank_score (out-of-fold re-ranker),
y (1 = real site), ignore (braak / below-MMU, excluded), blad_nr (map sheet).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

SEED = 24
DETECTION_THRESHOLD = 0.30   # pixel threshold the candidates were cut at (the detector's inference run)
N_FOLDS = 5
N_BINS = 10
N_BOOT = 2000


def sheet_folds(sheets, n=N_FOLDS, seed=SEED):
    """Whole map sheets per fold: neighbouring sites share imagery and context,
    so a random split leaks. Same rule as the detector's own validation."""
    u = np.array(sorted(pd.unique(sheets)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    assign = {s: i % n for i, s in enumerate(u)}
    return np.array([assign[s] for s in sheets])


def ece(p, y, n_bins=N_BINS):
    """Expected calibration error over equal-count bins (robust to a skewed
    score distribution, unlike equal-width bins)."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def reliability(p, y, n_bins=N_BINS):
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            k, n = y[m].sum(), m.sum()
            # Wilson interval: honest error bars on a bin's observed precision.
            z = 1.96
            ph = k / n
            c = (ph + z * z / (2 * n)) / (1 + z * z / n)
            h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            rows.append({"bin": b, "n": int(n), "mean_score": float(p[m].mean()),
                         "observed": float(ph), "lo": float(c - h), "hi": float(c + h)})
    return pd.DataFrame(rows)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calibrate_oof(score, y, folds, method):
    """Fit the calibrator on four folds of sheets, apply it to the fifth."""
    out = np.zeros_like(score)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if method == "platt":
            m = LogisticRegression(C=1e6).fit(logit(score[tr])[:, None], y[tr])
            out[te] = m.predict_proba(logit(score[te])[:, None])[:, 1]
        else:
            m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score[tr], y[tr])
            out[te] = m.predict(score[te])
    return out


def metrics(p, y):
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": ece(p, y),
            "mean_score": float(p.mean()), "base_rate": float(y.mean())}


def cluster_bootstrap(scores, y, sheets, n_boot=N_BOOT, seed=SEED):
    """Resample whole sheets, not candidates: the uncertainty that matters is
    'would this hold on a different set of map sheets'."""
    rng = np.random.default_rng(seed)
    u = pd.unique(sheets)
    rows_by = {s: np.flatnonzero(sheets == s) for s in u}
    out = {name: {k: [] for k in ("auc", "ap", "brier", "ece")} for name in scores}
    for _ in range(n_boot):
        pick = np.concatenate([rows_by[s] for s in rng.choice(u, len(u), replace=True)])
        yy = y[pick]
        if yy.min() == yy.max():
            continue
        for name, p in scores.items():
            m = metrics(p[pick], yy)
            for k in out[name]:
                out[name][k].append(m[k])
    return {name: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                   for k, v in d.items()} for name, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    a = ap.parse_args()

    X = pd.read_csv(a.candidates)
    n_all = len(X)
    X = X[X.ignore == 0].reset_index(drop=True)
    y = X.y.values.astype(int)
    sheets = X.blad_nr.astype(str).values
    folds = sheet_folds(sheets)
    print(f"{n_all} candidates, {len(X)} scoreable, {y.sum()} real ({y.mean():.1%}), "
          f"{len(pd.unique(sheets))} sheets")

    raw = X.confidence.values.astype(float)
    rr = X.rerank_score.values.astype(float)
    scores = {
        "detector_raw": raw,
        "detector_platt": calibrate_oof(raw, y, folds, "platt"),
        "detector_isotonic": calibrate_oof(raw, y, folds, "isotonic"),
        "reranker_raw": rr,
        "reranker_isotonic": calibrate_oof(rr, y, folds, "isotonic"),
    }

    res = {"n_candidates": int(len(X)), "n_excluded_ignore": int(n_all - len(X)),
           "n_real": int(y.sum()), "n_sheets": int(len(pd.unique(sheets))),
           "seed": SEED, "folds": N_FOLDS, "bins": N_BINS, "bootstrap": N_BOOT,
           "metrics": {k: metrics(v, y) for k, v in scores.items()}}
    print("bootstrapping by sheet ...")
    res["ci95"] = cluster_bootstrap(scores, y, sheets)

    # What the confidence promises versus what an interpreter gets.
    promise = []
    for name in ("detector_raw", "detector_platt", "detector_isotonic", "reranker_isotonic"):
        p = scores[name]
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            m = p >= t
            promise.append({"score": name, "threshold": t, "n_flagged": int(m.sum()),
                            "share_flagged": float(m.mean()),
                            "precision": float(y[m].mean()) if m.any() else float("nan"),
                            "recall": float(y[m].sum() / y.sum())})
    promise = pd.DataFrame(promise)

    # Official-statistics use: estimate how many sites a sheet holds by summing
    # probabilities, before anyone has looked. Only calibrated scores can do this.
    sheet_rows = []
    for s in pd.unique(sheets):
        m = sheets == s
        row = {"sheet": s, "n_candidates": int(m.sum()), "n_real": int(y[m].sum())}
        for name in ("detector_raw", "detector_platt", "detector_isotonic", "reranker_isotonic"):
            row[f"expected_{name}"] = float(scores[name][m].sum())
        sheet_rows.append(row)
    sh = pd.DataFrame(sheet_rows).sort_values("n_candidates", ascending=False).reset_index(drop=True)
    # Sheet names locate internal labels; publish an anonymous id instead.
    sh["sheet"] = [f"S{i + 1:02d}" for i in range(len(sh))]
    count = {}
    for name in ("detector_raw", "detector_platt", "detector_isotonic", "reranker_isotonic"):
        e = sh[f"expected_{name}"] - sh.n_real
        count[name] = {"total_expected": float(sh[f"expected_{name}"].sum()),
                       "total_real": int(sh.n_real.sum()),
                       "sheet_mae": float(e.abs().mean()), "sheet_bias": float(e.mean())}
    res["sheet_counts"] = count

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    promise.to_csv(out / "threshold_promise.csv", index=False)
    sh.to_csv(out / "sheet_counts.csv", index=False)
    rel = pd.concat([reliability(v, y).assign(score=k) for k, v in scores.items()])
    rel.to_csv(out / "reliability_bins.csv", index=False)

    figures(scores, y, rel, sh, res, HERE / "figures")

    print(f"\n{'score':20}{'AUC':>7}{'AP':>7}{'Brier':>8}{'ECE':>7}{'mean p':>8}")
    for k, m in res["metrics"].items():
        print(f"{k:20}{m['auc']:7.3f}{m['ap']:7.3f}{m['brier']:8.3f}{m['ece']:7.3f}{m['mean_score']:8.3f}")
    print(f"base rate {y.mean():.3f}")
    for k, c in count.items():
        print(f"sheet counts {k:20} expected {c['total_expected']:7.1f} vs real {c['total_real']} "
              f"| MAE/sheet {c['sheet_mae']:.2f} bias {c['sheet_bias']:+.2f}")


def figures(scores, y, rel, sh, res, fig_dir):
    import matplotlib.pyplot as plt
    style.apply()
    raw, cal = "detector_raw", "detector_platt"

    # 1. The headline: what the score says vs what is true.
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], color=style.LIGHT, lw=1.2)
    ax.text(0.96, 0.9, "honest", color=style.GREY, fontsize=9, ha="right", rotation=42)
    for name, col in ((raw, style.CLAY), (cal, style.TEAL)):
        r = rel[rel.score == name]
        ax.fill_between(r.mean_score, r.lo, r.hi, color=col, alpha=0.13, lw=0)
        ax.plot(r.mean_score, r.observed, color=col, marker="o", ms=4.5, lw=2.2)
    r0, r1 = rel[rel.score == raw], rel[rel.score == cal]
    ax.text(r0.mean_score.iloc[-1] + 0.02, r0.observed.iloc[-1] - 0.06,
            f"as shipped\nerror {res['metrics'][raw]['ece']:.2f}", color=style.CLAY, fontsize=10, va="top")
    ax.text(0.05, 0.72, f"after 2-parameter\ncalibration\nerror {res['metrics'][cal]['ece']:.2f}",
            color=style.TEAL, fontsize=10, va="center", ha="left")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="what the score says",
           ylabel="share that an interpreter confirmed")
    ax.set_title("When the model said 0.8, it was right about 40% of the time")
    style.save(fig, fig_dir / "reliability.png")

    # 2. Why: every candidate already cleared the 0.30 pixel threshold (the detector's inference run).
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(scores[raw][y == 0], bins, color=style.LIGHT, label="not confirmed")
    ax.hist(scores[raw][y == 1], bins, color=style.TEAL, label="confirmed")
    ax.axvline(DETECTION_THRESHOLD, color=style.INK, lw=0.8, ls=":")
    ax.text(DETECTION_THRESHOLD - 0.01, ax.get_ylim()[1] * 0.45, "pixel\nthreshold", ha="right",
            fontsize=8.5, color=style.INK)
    ax.set(xlabel="score as shipped", ylabel="candidates", xlim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Every candidate is an average of pixels that already passed 0.30")
    style.save(fig, fig_dir / "score_histogram.png")

    # 3. Planning use: add up probabilities per sheet = expected confirmed candidates.
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    mx = max(sh.n_real.max(), sh[f"expected_{raw}"].max()) * 1.08
    ax.plot([0, mx], [0, mx], color=style.LIGHT, lw=1.2)
    for name, col, lab in ((raw, style.CLAY, "sum of raw scores"),
                           (cal, style.TEAL, "sum of calibrated probabilities")):
        ax.scatter(sh.n_real, sh[f"expected_{name}"], s=26, color=col, alpha=0.9, lw=0)
        c = res["sheet_counts"][name]
        ax.text(0.03, 0.97 if name == raw else 0.89,
                f"{lab}: {c['total_expected']:.0f}",
                transform=ax.transAxes, color=col, fontsize=10, va="top")
    ax.text(0.03, 0.81, f"candidates actually confirmed: {int(sh.n_real.sum())}", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.set(xlim=(0, mx), ylim=(0, mx), xlabel="candidates confirmed on a map sheet",
           ylabel="confirmations the model expects on that sheet")
    ax.set_title("How many candidates on a sheet will an interpreter confirm?")
    style.save(fig, fig_dir / "sheet_counts.png")


if __name__ == "__main__":
    main()
