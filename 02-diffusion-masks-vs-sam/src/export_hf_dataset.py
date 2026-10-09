"""Package the open-data benchmark for the Hugging Face Hub.

Only public layers go in: PDOK imagery, NBBG2022 class 34, BAG buildings.
The CBS-internal active-site masks (data/internal/) are never read here.

  python src/export_hf_dataset.py                      # writes hf_dataset/ (local)
  python src/export_hf_dataset.py --push AthithyaLogan/nl-aerial-proposal-bench
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
N_DEV = 20
PUBLIC = ("stuff_c34", "things_bag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", default="")
    a = ap.parse_args()
    meta = json.loads((HERE / "data" / "tiles.json").read_text())
    out = HERE / "hf_dataset"
    if out.exists():
        shutil.rmtree(out)
    for split in ("dev", "test"):
        (out / split).mkdir(parents=True)
        (out / "masks" / split).mkdir(parents=True)

    n_obj = {k: 0 for k in PUBLIC}
    for i, r in enumerate(meta["tiles"]):
        split = "dev" if i < N_DEV else "test"
        tid = r["id"]
        shutil.copy(HERE / "data" / "tiles" / f"{tid}.png", out / split / f"{tid}.png")
        gt = np.load(HERE / "data" / "tiles" / f"{tid}_gt.npz")
        assert set(gt.files) == set(PUBLIC), f"unexpected layers in {tid}: {gt.files}"
        rec = {"file_name": f"{tid}.png", "id": tid, "bbox_rd_new": r["bbox_rd"],
               "crs": "EPSG:28992", "gsd_m": meta["gsd_m"]}
        for k in PUBLIC:
            Image.fromarray(gt[k].astype(np.uint16)).save(out / "masks" / split / f"{tid}_{k}.png")
            rec[f"{k}_mask"] = f"masks/{split}/{tid}_{k}.png"
            rec[f"n_{k}"] = int(gt[k].max())
            n_obj[k] += int(gt[k].max())
        with open(out / split / "metadata.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # Round-trip check: 16-bit PNGs must decode to the same instance ids.
    probe = meta["tiles"][0]["id"]
    for k in PUBLIC:
        back = np.asarray(Image.open(out / "masks" / "dev" / f"{probe}_{k}.png"))
        assert (back == np.load(HERE / "data" / "tiles" / f"{probe}_gt.npz")[k]).all()

    (out / "README.md").write_text(CARD.format(n=len(meta["tiles"]), c34=n_obj["stuff_c34"],
                                               bag=n_obj["things_bag"], seed=meta["seed"],
                                               mins=meta["min_stuff_m2"], minb=meta["min_building_m2"]),
                                   encoding="utf-8")
    print(f"wrote {out}: {len(meta['tiles'])} tiles, {n_obj}")
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push, repo_type="dataset",
                          commit_message="200 PDOK 2022 tiles with NBBG class-34 and BAG instance masks")
        print(f"pushed https://huggingface.co/datasets/{a.push}")


CARD = """---
license: cc-by-4.0
task_categories: [image-segmentation]
tags: [aerial-imagery, netherlands, construction-sites, buildings, object-proposals, earth-observation]
size_categories: [n<1K]
configs:
- config_name: default
  data_files:
  - split: dev
    path: dev/*
  - split: test
    path: test/*
---

# NL aerial proposal benchmark (200 tiles)

{n} tiles of Dutch aerial imagery, each 512 × 512 px at 1 m (512 × 512 m), with instance masks
for two kinds of object:

- **construction land**, CBS NBBG2022 class 34 *Bouwterrein* ("stuff": amorphous, defined partly by use). {c34} objects.
- **buildings**, BAG *panden* ("things": crisp footprints, ≥ {minb} m²). {bag} objects.

It was built for [experiment 02 of my Posts series](https://github.com/athithyai/Posts/tree/main/02-diffusion-masks-vs-sam),
which compares unsupervised mask proposals from diffusion self-attention against Segment Anything.
The metric used there is COCO-style average recall at K proposals per tile.

## How the tiles were chosen

Tiles are centred on randomly drawn NBBG2022 class-34 polygons of 1–20 ha anywhere in the
Netherlands (seed {seed}), with a random offset of up to ±120 m and no two tiles overlapping.
Every tile contains at least one construction object, kept only if ≥ {mins} m² of it lies inside the
tile. Splits: `dev` = the first 20 tiles (for choosing hyperparameters), `test` = the other 180.

## Files

- `dev/tNNN.png`, `test/tNNN.png`: RGB, PDOK `2022_orthoHR` (8 cm orthophoto served at 1 m).
- `masks/<split>/tNNN_stuff_c34.png`, `masks/<split>/tNNN_things_bag.png`: 16-bit instance masks
  (0 = background, 1…N = objects). Paths are in the metadata columns `stuff_c34_mask`, `things_bag_mask`.
- `<split>/metadata.jsonl`: tile ID, bounding box in RD New (EPSG:28992), object counts.

Rebuild from source with `build_tiles.py` in the repository.

## Licences and attribution

- Imagery: © Beeldmateriaal Nederland, via PDOK, CC BY 4.0
- Construction land: CBS, NBBG2022 publicatiebestand, CC BY 4.0
- Buildings: Kadaster, BAG via PDOK, CC0
- This compilation: CC BY 4.0, Athithya Loganathan

## Caveats

Class 34 includes serviced but un-built plots ("braak"), which look like grass. So
part of the construction ground truth isn't visible in the image at all, by definition. Building
masks are cadastral footprints, not roof outlines, so they can be offset from what the image shows.
"""

if __name__ == "__main__":
    main()
