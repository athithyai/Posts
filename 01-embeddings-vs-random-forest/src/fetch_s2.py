"""Sentinel-2 L2A 2020 for the exact pixel grid of the Spheer extract.

The comparison is only fair if both feature sets describe the same pixels from
the same year. So the grid is not chosen here: it is read from the Spheer
cache (EPSG:32631, 20 m, 561 x 574 px over the Lelystad coast), and every
Sentinel-2 scene is resampled onto it.

Source: Microsoft Planetary Computer, collection sentinel-2-l2a (free, no
account; URLs are signed anonymously). Reads are windowed COG reads at 20 m, so
GDAL serves the 10 m bands from their overviews and only the AOI is transferred.

  python src/fetch_s2.py --spheer <Spheer_EM>/cache/markermeer_31UFU_2020_ov1.npz
Writes data/s2_2020_scenes.npz (all clear observations) - not committed.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import projfix  # noqa: E402,F401  (before rasterio)

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

HERE = Path(__file__).resolve().parent.parent
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# Scene classification: keep vegetation, bare soil, water, unclassified, snow.
# Drop no-data, saturated, dark/shadow, cloud shadow, clouds (medium/high), cirrus.
SCL_BAD = (0, 1, 2, 3, 8, 9, 10)
MAX_SCENE_CLOUD = 60      # scene-level filter; pixel-level masking does the rest
MIN_CLEAR = 0.30          # skip a date if under 30% of the AOI is clear


def grid_from_spheer(path):
    z = np.load(path, allow_pickle=True)
    _, h, w = z["a"].shape
    t = Affine(*z["t"][:6])
    return t, w, h, str(z["crs"])


def read_onto_grid(href, transform, w, h, crs, resampling):
    with rasterio.open(href) as src, WarpedVRT(src, crs=crs, transform=transform, width=w,
                                               height=h, resampling=resampling) as vrt:
        return vrt.read(1)


def fetch_item(item, transform, w, h, crs):
    item = pc.sign(item)
    scl = read_onto_grid(item.assets["SCL"].href, transform, w, h, crs, Resampling.nearest)
    clear = ~np.isin(scl, SCL_BAD)
    if clear.mean() < MIN_CLEAR:
        return item.id, None, float(clear.mean())
    stack = np.stack([read_onto_grid(item.assets[b].href, transform, w, h, crs, Resampling.average)
                      for b in BANDS]).astype(np.float32)
    # L2A processing baseline >= 04.00 adds a +1000 offset to reflectances.
    offset = 1000.0 if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    refl = (stack - offset) / 10000.0
    refl[:, ~clear] = np.nan
    return item.id, (item.datetime.strftime("%Y-%m-%d"), refl.astype(np.float16)), float(clear.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spheer", required=True, help="Spheer 2020 cache .npz (defines the grid)")
    ap.add_argument("--year", type=int, default=2020)
    a = ap.parse_args()

    transform, w, h, crs = grid_from_spheer(a.spheer)
    bounds = (transform.c, transform.f + transform.e * h, transform.c + transform.a * w, transform.f)
    bbox = transform_bounds(crs, "EPSG:4326", *bounds)
    print(f"grid {w}x{h} @ {transform.a:.0f} m {crs}, bbox4326 {np.round(bbox, 4).tolist()}")

    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{a.year}-01-01/{a.year}-12-31",
        query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}}).items())
    # One tile only: 31UFU covers the AOI; neighbouring-tile duplicates add nothing.
    items = [i for i in items if i.properties.get("s2:mgrs_tile") == "31UFU"]
    print(f"{len(items)} scenes with cloud < {MAX_SCENE_CLOUD}% on tile 31UFU")

    kept, log = {}, []
    with ThreadPoolExecutor(6) as ex:
        for iid, res, clr in ex.map(lambda it: fetch_item(it, transform, w, h, crs), items):
            log.append({"id": iid, "clear": round(clr, 3), "kept": res is not None})
            if res is not None:
                d, arr = res
                # Two scenes on one day (overlapping orbits): keep the clearer one.
                if d not in kept or np.isfinite(arr[0]).mean() > np.isfinite(kept[d][0]).mean():
                    kept[d] = arr
            print(f"  {iid[:40]:40s} clear {clr:5.1%} {'kept' if res is not None else 'skip'}")

    dates = sorted(kept)
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"s2_{a.year}_scenes.npz", x=np.stack([kept[d] for d in dates]),
                        dates=np.array(dates), bands=np.array(BANDS),
                        transform=np.array(transform)[:6], crs=crs)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "s2_scene_log.json").write_text(json.dumps(
        {"stac": STAC, "collection": "sentinel-2-l2a", "year": a.year, "tile": "31UFU",
         "max_scene_cloud": MAX_SCENE_CLOUD, "min_clear": MIN_CLEAR, "scl_masked": SCL_BAD,
         "dates_kept": dates, "scenes": log}, indent=1))
    print(f"kept {len(dates)} dates -> {out / f's2_{a.year}_scenes.npz'}")


if __name__ == "__main__":
    main()
