# Posts

**Small, honest experiments that test ideas from my feed against my own work.**

I work on Earth observation for official statistics in the Netherlands: CBS land-use maps, construction-site detection from aerial photos, and foundation-model embeddings. Every week my feed has a dozen confident claims about geospatial AI. This repository is where I take one, turn it into a hypothesis about my own data, and run it.

Every experiment follows the same structure: context, hypothesis, data, method, results, thoughts, limitations. Every number in a post comes from a script in the same folder, with a fixed seed and spatially blocked validation (never a random pixel split). Each folder has a `README.md` (the blog post) and a `linkedin.md` (the short version).

| # | Prompted by | Question on my data | Answer in one line |
|---|---|---|---|
| [01](01-embeddings-vs-random-forest/) | ECCV 2026: *"I wish there were an embedding better than a random forest"* | Does a precomputed Sentinel-2 foundation-model embedding beat a random forest on raw Sentinel-2, for CBS land-use classes? | Yes, modestly: a linear model on the embedding scores macro-F1 0.592 vs 0.551 (paired +0.041, 95% CI +0.006 to +0.066), and needs 20x fewer labels |
| [02](02-diffusion-masks-vs-sam/) | Diffuse2Seg: *diffusion models can segment anything without supervision* | Do diffusion self-attention masks find Dutch construction sites and buildings better than SAM? | Only with SAM at its defaults. With looser filters SAM wins on construction sites too (AR@100 0.29 vs 0.25) and wins easily on buildings (0.29 vs 0.08) |
| [03](03-calibrated-confidence/) | Google ALU: one confidence per mapped field | Does my construction detector's 0.8 mean 80%? | No: when it said 0.8 it was right about 40% of the time. Two parameters fix it: summed scores predicted 991 confirmations, calibrated ones 319, actual 317 |

## What is and isn't in here

Committed: code, figures, aggregate metrics, and anything built from **open data**: PDOK aerial imagery (CC BY 4.0), CBS NBBG land-use publication files (CC BY 4.0), BAG buildings (CC0), Copernicus Sentinel-2 via Microsoft Planetary Computer.

Not committed:
- **CBS-internal labels** (the 2024 construction-site verification set). Experiments that use them say so and publish aggregates only.
- **Spheer FM embeddings.** The dataset is gated on Hugging Face; request access there.
- **Rebuildable rasters and imagery tiles.** Every `data/` folder is recreated by the scripts.

## Running

```bash
conda create -n posts python=3.11 && conda activate posts
pip install -r requirements.txt
```

Each folder's README has its exact commands. Every figure is drawn by the experiment's own Python script in one shared style ([`common/style.py`](common/style.py)): white background, Space Grotesk (bundled, OFL), teal for the thing the post is about, clay for the baseline or the problem, and greys for context. Everything was run on one Windows laptop with an RTX 5070 (8 GB), one GPU job at a time.

## Licence

Code MIT; text and figures CC BY 4.0. Data keeps its own licence; see [LICENSE](LICENSE).

— Athithya Loganathan · [athithyaloganathan.com](https://athithyaloganathan.com)
