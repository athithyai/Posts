"""Build the benchmark: 512 x 512 px aerial tiles at 1 m with public ground truth.

Everything here is open data, so anyone can rebuild the exact benchmark:
  imagery   PDOK luchtfoto RGB 2022_orthoHR (8 cm, served at 1 m), CC BY 4.0
  'stuff'   CBS NBBG2022 class 34 Bouwterrein (construction land), CC BY 4.0
  'things'  BAG panden (building footprints) from the PDOK BAG WFS, CC0

Tiles are centred on randomly drawn class-34 polygons (seeded), shifted by a
random offset so the target is never exactly in the middle. Optionally, a
CBS-internal set of interpreter-verified *active* construction sites can be
attached for scoring only (--internal); it is never written to the public set.

  python src/build_tiles.py --nbbg2022 <...>/CBS_Publicatiebestand_NBBG2022_v1_def.gpkg \
      [--internal <BT24>/data/labels.gpkg] --n 200
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import projfix  # noqa: E402,F401

import geopandas as gpd
import numpy as np
import pyogrio
import requests
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

HERE = Path(__file__).resolve().parent.parent
WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
LAYER = "2022_orthoHR"
BAG_WFS = "https://service.pdok.nl/lv/bag/wfs/v2_0"
PX, GSD = 512, 1.0
SEED = 24
MIN_STUFF_M2 = 5_000      # a construction polygon must keep 0.5 ha inside the tile
MIN_BUILDING_M2 = 200     # ~200 px at 1 m: below this nothing segments reliably


def get(url, params, tries=5):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def fetch_image(b):
    r = get(WMS, {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": LAYER,
                  "CRS": "EPSG:28992", "BBOX": ",".join(f"{v:.1f}" for v in b), "WIDTH": PX,
                  "HEIGHT": PX, "FORMAT": "image/png", "STYLES": ""})
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def fetch_buildings(b):
    feats, start = [], 0
    while True:
        r = get(BAG_WFS, {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                          "TYPENAMES": "bag:pand", "SRSNAME": "EPSG:28992",
                          "BBOX": ",".join(f"{v:.1f}" for v in b) + ",EPSG:28992",
                          "OUTPUTFORMAT": "application/json", "COUNT": 1000, "STARTINDEX": start})
        page = r.json().get("features", [])
        feats += page
        if len(page) < 1000:
            break
        start += 1000
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=28992)
    g = gpd.GeoDataFrame.from_features(feats, crs=28992)
    return g[g.geometry.notna()]


def instance_raster(geoms, b):
    """One id per object, 0 = background. Objects are rasterised largest-first
    so a small object lying on a large one keeps its own pixels."""
    tr = from_origin(b[0], b[3], GSD, GSD)
    geoms = sorted(geoms, key=lambda g: -g.area)
    if not geoms:
        return np.zeros((PX, PX), np.uint16)
    return rasterize(((g, i + 1) for i, g in enumerate(geoms)), out_shape=(PX, PX),
                     transform=tr, fill=0, dtype="uint16")


def objects_in(gdf, tile, min_m2):
    clip = gdf.clip(tile)
    clip = clip[clip.geometry.area >= min_m2]
    return [g for g in clip.geometry.explode(index_parts=False) if g.area >= min_m2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbbg2022", required=True)
    ap.add_argument("--internal", default="", help="BT24 labels.gpkg (CBS-internal, scoring only)")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    layer = pyogrio.list_layers(a.nbbg2022)[0][0]
    nb = pyogrio.read_dataframe(a.nbbg2022, layer=layer, where="NBBG22 = 34")
    nb = nb.explode(index_parts=False)
    nb = nb[(nb.area >= 10_000) & (nb.area <= 200_000)]   # 1-20 ha: fits a 512 m tile
    nb.geometry = nb.geometry.force_2d()
    nb = nb.reset_index(drop=True)
    print(f"{len(nb)} class-34 polygons of 1-20 ha nationwide")

    internal = None
    if a.internal:
        s = pyogrio.read_dataframe(a.internal, layer="sites")
        internal = s[(s.jaar == 2022) & (s.klasse == "actief")].copy()
        internal.geometry = internal.geometry.force_2d()
        print(f"{len(internal)} internal active sites (2022) for scoring only")

    out = HERE / "data" / "tiles"
    out.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(nb))
    taken, rows = [], []
    for i in order:
        if len(rows) >= a.n:
            break
        c = nb.geometry.iloc[i].centroid
        dx, dy = rng.uniform(-120, 120, 2)
        x0 = np.floor((c.x + dx - PX * GSD / 2) / 8) * 8
        y0 = np.floor((c.y + dy - PX * GSD / 2) / 8) * 8
        b = (x0, y0, x0 + PX * GSD, y0 + PX * GSD)
        tile = box(*b)
        if any(tile.intersects(t) for t in taken):          # no overlapping tiles
            continue
        stuff = objects_in(nb, tile, MIN_STUFF_M2)
        if not stuff:
            continue
        try:
            img = fetch_image(b)
        except RuntimeError:
            continue
        if (img.max(axis=2) < 8).mean() > 0.05:             # outside coverage
            continue
        bld = fetch_buildings(b)
        things = objects_in(bld, tile, MIN_BUILDING_M2) if len(bld) else []
        tid = f"t{len(rows):03d}"
        Image.fromarray(img).save(out / f"{tid}.png")
        gt = {"stuff_c34": instance_raster(stuff, b), "things_bag": instance_raster(things, b)}
        np.savez_compressed(out / f"{tid}_gt.npz", **gt)
        row = {"id": tid, "bbox_rd": [float(v) for v in b], "n_stuff_c34": len(stuff),
               "n_things_bag": len(things)}
        if internal is not None:
            act = objects_in(internal, tile, MIN_STUFF_M2)
            idir = HERE / "data" / "internal"           # git-ignored, never published
            idir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(idir / f"{tid}_actief.npz", actief=instance_raster(act, b))
            row["n_actief_internal"] = len(act)
        rows.append(row)
        taken.append(tile)
        print(f"  {tid} {np.round(b[:2]).astype(int).tolist()} c34 {len(stuff):2d} "
              f"bag {len(things):3d}" + (f" actief {row['n_actief_internal']}" if internal is not None else ""),
              flush=True)

    meta = {"imagery": f"PDOK luchtfotorgb {LAYER}", "px": PX, "gsd_m": GSD, "seed": SEED,
            "min_stuff_m2": MIN_STUFF_M2, "min_building_m2": MIN_BUILDING_M2, "tiles": rows}
    (HERE / "data" / "tiles.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(rows)} tiles -> {out}")


if __name__ == "__main__":
    main()
