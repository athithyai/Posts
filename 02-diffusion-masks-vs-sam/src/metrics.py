"""Average recall of object proposals, COCO style.

For every ground-truth object take the best IoU among a tile's top-K
proposals; recall at threshold th is the share of objects whose best IoU >= th;
AR@K averages recall over th = 0.50, 0.55, ..., 0.95.
"""
import numpy as np

THRESH = np.arange(0.5, 0.951, 0.05)


def best_iou(tiles, gt_key, k):
    out = []
    for r in tiles.values():
        m = r["iou"].get(gt_key)
        if m is None or m.shape[1] == 0:
            continue
        top = m[:k]
        out.append(top.max(0) if len(top) else np.zeros(m.shape[1], np.float32))
    return np.concatenate(out) if out else np.zeros(0)


def ar(tiles, gt_key, k):
    b = best_iou(tiles, gt_key, k)
    return float(np.mean([(b >= t).mean() for t in THRESH])) if len(b) else float("nan")


def recall_at(tiles, gt_key, k, th=0.5):
    b = best_iou(tiles, gt_key, k)
    return float((b >= th).mean()) if len(b) else float("nan")


def ar_ci(tiles, gt_key, k, n=1000, seed=24):
    """Bootstrap over tiles."""
    ids = [t for t, r in tiles.items() if r["iou"].get(gt_key) is not None and r["iou"][gt_key].shape[1]]
    rng = np.random.default_rng(seed)
    per = {t: (tiles[t]["iou"][gt_key][:k].max(0) if len(tiles[t]["iou"][gt_key][:k])
               else np.zeros(tiles[t]["iou"][gt_key].shape[1])) for t in ids}
    vals = []
    for _ in range(n):
        b = np.concatenate([per[t] for t in rng.choice(ids, len(ids), replace=True)])
        vals.append(np.mean([(b >= th).mean() for th in THRESH]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_ar_delta(tiles_a, tiles_b, gt_key, k, n=1000, seed=24):
    """AR@k(a) - AR@k(b) with the same resampled tiles for both methods."""
    ids = [t for t in tiles_a if t in tiles_b and tiles_a[t]["iou"].get(gt_key) is not None
           and tiles_a[t]["iou"][gt_key].shape[1]]

    def per(tiles, t):
        m = tiles[t]["iou"][gt_key][:k]
        return m.max(0) if len(m) else np.zeros(tiles[t]["iou"][gt_key].shape[1])

    pa = {t: per(tiles_a, t) for t in ids}
    pb = {t: per(tiles_b, t) for t in ids}
    ar_of = lambda b: np.mean([(b >= th).mean() for th in THRESH])
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        d.append(ar_of(np.concatenate([pa[t] for t in pick])) - ar_of(np.concatenate([pb[t] for t in pick])))
    full = ar_of(np.concatenate([pa[t] for t in ids])) - ar_of(np.concatenate([pb[t] for t in ids]))
    return {"delta": float(full), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
