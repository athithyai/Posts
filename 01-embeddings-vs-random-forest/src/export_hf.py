"""Package the linear probe for the Hugging Face Hub as safetensors + JSON.

No pickle: the probe is 11 x 100 weights plus a standardiser, so it is stored
as plain tensors and applied with numpy. The model card is written alongside.

  python src/export_hf.py            # writes hf/ (local)
  python src/export_hf.py --push AthithyaLogan/spheer-nbbg-linear-probe
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from safetensors.numpy import save_file

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", default="")
    a = ap.parse_args()
    b = joblib.load(HERE / "model" / "probe.joblib")
    pipe = b["pipeline"]
    sc, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    out = HERE / "hf"
    out.mkdir(exist_ok=True)
    save_file({"mean": sc.mean_.astype(np.float32), "scale": sc.scale_.astype(np.float32),
               "coef": lr.coef_.astype(np.float32), "intercept": lr.intercept_.astype(np.float32)},
              str(out / "probe.safetensors"))
    res = json.loads((HERE / "results" / "metrics.json").read_text())
    cfg = {"input": "Spheer FM Albatross annual embedding, 100-d (spheer/spheer-fm-embeddings)",
           "classes": [int(c) for c in lr.classes_],
           "class_names": {int(c): b["names"][int(c)] for c in lr.classes_},
           "label_source": "CBS NBBG2020 land use", "trained_on": "Lelystad coast, MGRS 31UFU, 2020, 20 m",
           "cv_macro_f1": res["models"]["linear_spheer"]["macro_f1"],
           "cv_macro_f1_ci95": res["models"]["linear_spheer"]["macro_f1_ci95"]}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "README.md").write_text(CARD.format(
        f1=cfg["cv_macro_f1"], lo=cfg["cv_macro_f1_ci95"][0], hi=cfg["cv_macro_f1_ci95"][1],
        rf=res["models"]["rf_s2"]["macro_f1"],
        d=res["paired_vs_rf_s2"]["linear_spheer"]["delta"],
        dlo=res["paired_vs_rf_s2"]["linear_spheer"]["ci95"][0],
        dhi=res["paired_vs_rf_s2"]["linear_spheer"]["ci95"][1],
        classes=", ".join(f"{c} {n}" for c, n in cfg["class_names"].items())), encoding="utf-8")
    print(f"wrote {out}")
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push, repo_type="model",
                          commit_message="Linear probe: Spheer embedding -> CBS NBBG2020 land use")
        print(f"pushed https://huggingface.co/{a.push}")


CARD = """---
license: mit
tags: [earth-observation, remote-sensing, land-use, linear-probe, sentinel-2, netherlands]
---

# Spheer embedding → CBS land use, linear probe (Lelystad, 2020)

A 100 → 11 logistic-regression probe that maps the annual
[Spheer FM](https://huggingface.co/datasets/spheer/spheer-fm-embeddings) Sentinel-2 embedding
to Statistics Netherlands (CBS) NBBG2020 land-use classes. It's the model behind
[experiment 01 of my Posts series](https://github.com/athithyai/Posts/tree/main/01-embeddings-vs-random-forest),
which asks whether a precomputed embedding beats a random forest on raw Sentinel-2.

**It's a research artefact, not a land-use product.** It was trained on a single
11 × 11.5 km window (Lelystad coast, MGRS 31UFU, 20 m), 89% of which is open water.

## Results (5-fold, 1 km spatial blocks, 100 m buffer)

- macro-F1 over 11 classes: **{f1:.3f}** (95% block-bootstrap CI {lo:.3f}–{hi:.3f})
- random forest on 46 hand-built Sentinel-2 features, same pixels: {rf:.3f}
- paired difference: **{d:+.3f}** (95% CI {dlo:+.3f} to {dhi:+.3f})

Classes: {classes}.

## Use

```python
import json, numpy as np
from safetensors.numpy import load_file
w = load_file("probe.safetensors"); cfg = json.load(open("config.json"))
# emb: (N, 100) float32 Spheer 2020 embeddings, same model version as the training data
z = (emb - w["mean"]) / w["scale"]
logits = z @ w["coef"].T + w["intercept"]
pred = np.array(cfg["classes"])[logits.argmax(1)]
```

The embeddings are gated; request access on the dataset page. Don't mix model versions,
because embeddings from different Spheer versions live in different spaces.

## Limitations

One window, one year, 11 classes (agriculture, roads and construction had too few
pure pixels here and were dropped). CBS classes describe *use*, not cover, and are
partly unobservable from 20 m pixels. Expect it to transfer poorly outside the
training area without refitting.
"""

if __name__ == "__main__":
    main()
