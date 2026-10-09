"""Publish candidates the way Google's ALU publishes fields: one feature, one
stable id, one probability you can add up.

ALU gives every mapped feature a plus code of its centroid as id, an area and
a 0-1 confidence. This script does the same for construction-site candidates,
with the confidence replaced by a calibrated probability (isotonic, fitted on
the 2024 test sheets), and writes a GeoJSON a statistician can sum.

  python src/alu_schema.py --candidates <detector>/data/cand_2024_test_feat_scored.csv \
      --geometry <detector>/data/cand_2024_test.gpkg --out data/candidates_alu.geojson
The output contains model outputs over CBS-internal sheets; it stays local.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import projfix  # noqa: E402,F401

import numpy as np

ALPHABET = "23456789CFGHJMPQRVWX"
SCHEMA = {
    "id": "Open Location Code (plus code, 10 digits, ~14 m) of the polygon centroid",
    "feature_type": "always 'construction_site_candidate'",
    "area_m2": "polygon area in EPSG:28992",
    "p_site": "calibrated probability that an interpreter confirms this candidate as an active "
              "site (0-1); sums to the expected number of confirmed candidates",
    "score_raw": "raw detector score (a ranking signal, not a probability)",
    "imagery_year": "PDOK luchtfoto year the candidate was detected on",
    "model": "detector + calibrator version",
}


def plus_code(lat, lng):
    """10-digit Open Location Code with exact integer arithmetic."""
    lat = min(max(lat, -90.0), 90.0 - 1e-9)
    lng = ((lng + 180.0) % 360.0) - 180.0
    la = int(np.floor((lat + 90.0) * 8000))      # 1/8000 deg = finest pair resolution
    lo = int(np.floor((lng + 180.0) * 8000))
    pairs = []
    for _ in range(5):
        pairs.append(ALPHABET[la % 20] + ALPHABET[lo % 20])
        la //= 20
        lo //= 20
    code = "".join(reversed(pairs))
    return code[:8] + "+" + code[8:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import geopandas as gpd
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression

    X = pd.read_csv(a.candidates)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(X.rerank_score, X.y)
    g = gpd.read_file(a.geometry, engine="pyogrio")
    # The unscored feature table is row-for-row with the geometry (s10); the
    # scored one dropped braak / below-MMU rows (s11). Re-apply that filter.
    full = pd.read_csv(Path(a.candidates).with_name(Path(a.candidates).name.replace("_scored", "")))
    assert len(full) == len(g), "geometry and feature tables do not line up"
    g = g[full.ignore.values == 0].reset_index(drop=True)
    assert np.allclose(g.area_m2.values, X.area_m2.values), "row order differs"
    cen = g.geometry.centroid.to_crs(4326)
    out = gpd.GeoDataFrame({
        "id": [plus_code(p.y, p.x) for p in cen],
        "feature_type": "construction_site_candidate",
        "area_m2": g.area_m2.round(0).astype(int),
        "p_site": np.round(iso.predict(X.rerank_score), 3),
        "score_raw": X.confidence.round(3),
        "imagery_year": 2024,
        "model": "construction-detector+reranker / isotonic-2024test",
    }, geometry=g.geometry.to_crs(4326).values, crs=4326)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_file(a.out, driver="GeoJSON")
    print(f"{len(out)} features, expected sites {out.p_site.sum():.1f}, "
          f"unique ids {out.id.nunique()} -> {a.out}")


if __name__ == "__main__":
    main()
