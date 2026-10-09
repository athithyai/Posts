# A confidence of 0.8 should mean 80%

*Experiment 03 in the [Posts](../README.md) series. Prompted by a LinkedIn post on Google's Agricultural Landscape Understanding (ALU).*

**TL;DR.** My construction-site detector gives every candidate a confidence score. I checked it against what interpreters actually confirmed, and it was badly overconfident. The average score was 0.67, but only 22% of candidates were real. Summing the scores predicted **991 sites on 26 map sheets; there were 317**. A two-parameter calibration fitted on held-out sheets fixed it (error 0.458 → 0.02). Then the summed probabilities predicted **318**. So the score is now something you can count with, not just something you sort by.

---

## Context

A post about [Google ALU](https://developers.google.com/agricultural-understanding/understanding-data) went past my feed this week. ALU maps fields, ponds, wells and trees from satellite imagery. What caught my eye is that it ships every feature with an ID (a plus code of its centroid), an area and a **confidence between 0 and 1**. That's the right product design. A statistician can filter by it, sort by it or add it up.

My own system works the same way. [BT24](https://github.com/athithyai/bt24-bouwterrein) finds active construction sites in Dutch aerial photos as a candidate layer for the CBS land-use statistics (BBG), and every polygon it outputs carries a `confidence`. When I wrote the line that computes it, I never asked the obvious follow-up: **if I hand an interpreter a layer that says 0.8, does 0.8 mean anything?**

## Hypothesis

1. **H1:** The shipped BT24 confidence is not calibrated. A score of 0.8 does not mean an 80% chance of being a real site.
2. **H2:** A simple calibrator fitted on *other* map sheets (Platt or isotonic) makes it calibrated without hurting the ranking.
3. **H3:** Once calibrated, summed probabilities estimate how many sites a sheet holds before anyone opens it. That's the property official statistics needs, and a raw score can't provide it.

## Data

- **Candidates.** All BT24 candidates on the 2024 test sheets: 1,601 polygons on 26 kaartbladen (12.5 × 10 km sheets). 131 lie on fallow building plots or below the 1 ha minimum mapping unit. The CBS mapping rules exclude those, so they're neither right nor wrong and are left out. That leaves **1,470 candidates**.
- **Truth.** A candidate is real (`y = 1`) if it overlaps a construction site that a CBS photo-interpreter verified as *active* in the 2024 map. 317 are real (21.6%).
- **Scores compared.** (a) The detector confidence as shipped: the mean pixel probability inside the polygon. (b) The second-stage re-ranker's out-of-fold score, a gradient-boosted model on 31 context features: polygon shape, how many other candidates are nearby, the 2022 land-use class, image change against earlier years, NDVI inside versus around the polygon, and overlap with BRP crop parcels.
- The labels are CBS-internal, so this repository has the code and every aggregate result but not the candidate table.

## Method

- **Validation unit: the map sheet.** Neighbouring candidates share a flight, a season and often a construction project, so a random split leaks. Calibrators are fitted on four folds of whole sheets and applied to the fifth (5-fold, seed 24). Every number below is out-of-fold.
- **Calibrators.** Platt scaling (logistic regression on the logit of the score, 2 parameters) and isotonic regression (monotone step function).
- **Metrics.**
  - **ECE** (expected calibration error): over 10 equal-count bins, the average gap between stated confidence and observed precision.
  - **Brier score** for overall probabilistic quality.
  - **AUC and AP** to check that ranking wasn't damaged.
- **Uncertainty.** 95% intervals from 2,000 bootstrap resamples of *whole sheets*, i.e. "would this hold on a different set of sheets".
- **The statistician's test.** For each sheet, compare the sum of probabilities with the number of real sites.

## Results

![Reliability diagram](figures/reliability.png)

| score | AUC | AP | Brier | **ECE** [95% CI] | mean score |
|---|---|---|---|---|---|
| detector, as shipped | 0.823 | 0.566 | 0.348 | **0.458** [0.421, 0.491] | 0.674 |
| detector + Platt | 0.819 | 0.561 | 0.129 | **0.022** [0.019, 0.050] | 0.217 |
| detector + isotonic | 0.811 | 0.536 | 0.128 | **0.024** [0.018, 0.054] | 0.216 |
| re-ranker (out-of-fold) | 0.879 | 0.713 | 0.104 | **0.022** [0.017, 0.053] | 0.199 |
| re-ranker + isotonic | 0.869 | 0.685 | 0.105 | **0.022** [0.017, 0.047] | 0.217 |

Share of candidates that are real: 0.216.

**H1 holds, by a wide margin.** The red curve sits far below the diagonal. Candidates scored around 0.55 are real about 4% of the time, and even the most confident tenth (mean score 0.87) is real only 68% of the time. The histogram shows why:

![Score histogram](figures/score_histogram.png)

The confidence is the mean probability of pixels *that already passed the 0.5 threshold*, so it can't go below 0.5. It's a ranking signal squeezed into the top half of the scale. Nothing in its construction ever forced it to be a probability.

**H2 holds.** Two parameters (Platt) bring ECE from 0.458 to 0.022 and AUC barely moves (0.823 → 0.819). Isotonic gives the same calibration but loses a little ranking quality (0.811), because it collapses scores into ties. One thing I didn't expect: the re-ranker was *already* calibrated (ECE 0.022) before I touched it. It's a gradient-boosted classifier trained with log-loss on the same distribution, so probability is what it learned to produce. Isotonic on top adds nothing.

**H3 holds.**

![Sheet counts](figures/sheet_counts.png)

| per-sheet estimate from | total expected (real: 317) | mean abs. error per sheet | bias per sheet |
|---|---|---|---|
| raw detector confidence | 990.6 | 25.9 sites | +25.9 |
| calibrated detector | 318.2 | 3.9 sites | +0.05 |
| calibrated re-ranker | 318.3 | 3.2 sites | +0.05 |

Summing raw confidences overcounts by a factor of three on every sheet. Summing calibrated probabilities gets the 26-sheet total within two sites, with no systematic bias. A typical sheet is off by about three sites in either direction.

What the calibrated re-ranker promises at each threshold (out-of-fold):

| keep candidates with p ≥ | flagged | precision | recall |
|---|---|---|---|
| 0.5 | 16% | 0.71 | 0.52 |
| 0.7 | 10% | 0.86 | 0.40 |
| 0.9 | 3% | 0.93 | 0.13 |

With the raw score, "≥ 0.8" sounded like a high bar and delivered 60% precision. With the calibrated one, the number on the label is a lower bound you can actually rely on.

## Publishing it the way ALU does

[`src/alu_schema.py`](src/alu_schema.py) writes the candidates as one GeoJSON feature each, following ALU's pattern:

```json
{
  "id": "9F36XXXX+XX",
  "feature_type": "construction_site_candidate",
  "area_m2": 10640,
  "p_site": 0.81,
  "score_raw": 0.968,
  "imagery_year": 2024,
  "model": "bt24-detector+reranker / isotonic-2024test"
}
```

The `id` is a 10-digit Open Location Code of the centroid (about 14 m), computed with exact integer arithmetic. It's tested against the reference example `8FVC9G8F+6X`. (The digits are masked here because the candidate sits on an internal test sheet.) `p_site` is what makes the file useful: `sum(p_site)` is an estimate, not a vibe.

## Thoughts

- **Confidence is a contract with the reader, not a model output.** I'd been treating it as a sort key, which is fine for an interpreter working top-down through a list. But the moment the number is printed in an attribute table, somebody will filter at 0.8 or add it up. For a statistics office, that second use is the whole point.
- **Calibration is cheap; the hard part is having the right labels.** Two parameters, fitted in milliseconds. The expensive part is the interpreter-verified truth, and BT24 only has that because the 2024 map exists.
- **It will drift.** BT24's main lesson so far is that models trained on one flight year break on another. A calibrator is a model too. A 2024 calibrator applied to 2025 imagery is a hypothesis, not a fact. The plan: refit the two Platt parameters each year on the interpreter review set (the first 48 candidates are waiting on a reviewer). Two parameters can be estimated from 50 labels; a full isotonic curve can't.
- **This is the cheapest improvement BT24 has had.** No retraining, no new data, no GPU. One function and one honest plot.

## Limitations

- One year (2024), 26 sheets, 1,470 candidates. The bootstrap intervals are honest about sheet-to-sheet variation, not about other years.
- "Real" means *overlaps a verified active site*. A candidate that clips the corner of a real site counts as correct. That's generous to the detector, but it's the same rule for every score compared.
- The shipped calibrator in `alu_schema.py` is fitted on all 26 sheets, so its in-sample total (317.0) is exact by construction. The out-of-fold figure (318.3) is the honest one.

## Reproduce

```bash
pip install -r ../requirements.txt
python src/calibrate.py --candidates <BT24>/data/cand_2024_test_feat_scored.csv
python src/alu_schema.py --candidates <BT24>/data/cand_2024_test_feat_scored.csv \
    --geometry <BT24>/data/cand_2024_test.gpkg --out data/candidates_alu.geojson
```

It runs on a laptop CPU in about 20 seconds. The input table needs CBS access. Aggregates are in [`results/`](results/): `metrics.json`, `reliability_bins.csv`, `threshold_promise.csv` and `sheet_counts.csv` (sheet IDs anonymised).
