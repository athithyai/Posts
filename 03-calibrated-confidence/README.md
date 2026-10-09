# My model said 0.8. It meant 0.4.

*Experiment 03 in the [Posts](../README.md) series.*

---

Every polygon my construction-site model draws comes with a number between 0 and 1. I put it there myself, in a column called `confidence`, and I used it the way you'd expect: to sort the list, so the most likely sites come first.

What I never did was ask what the number *means*. If the model hands someone a polygon marked 0.8, is that an 80% chance of a real construction site? Or is it just "higher than 0.7"?

It sounds like a pedantic question until you remember who reads the table. Google's Agricultural Landscape Understanding, for example, publishes every field it maps with an ID, an area and a confidence between 0 and 1, and the people downstream will filter on it and add it up. Mine is meant for a statistics office. Adding things up is the job.

So this experiment is about one column.

## 1. The model behind the number

The model maps active construction sites in Dutch aerial photos, as a candidate layer for the photo-interpreters who maintain the CBS land-use map. It doesn't replace them. It points them at the 1% of a map sheet worth looking at, so they don't have to scan all of it.

It works in two stages.

**Stage 1, the detector.** A frozen **DINOv2-base** vision transformer (about 87M parameters, kept frozen) turns each 512 × 512 m tile of RGB aerial imagery, at 1 m per pixel, into features. A small convolutional decoder (**1.16M trainable parameters**) reads features from four depths of the transformer and predicts, for every pixel, the probability that it belongs to an active construction site.

This detector was trained on the **2022** land-use map only and applied here to **2024** imagery it had never seen. Pixels above **0.30** are grouped into polygons, and polygons under 1 ha are dropped (the official minimum mapping unit). Each polygon's `confidence` is the mean probability of its pixels.

**Stage 2, the re-ranker.** A gradient-boosted tree classifier (scikit-learn's `HistGradientBoostingClassifier`) re-orders the polygons using **31 context features** the detector can't see:
- polygon shape and size
- how many other candidates are nearby
- the land-use class in 2022
- image change compared with earlier years
- vegetation inside the polygon versus around it
- overlap with registered farm parcels

It outputs its own score per polygon.

## 2. Hypotheses

1. **H1.** The detector's `confidence` is not calibrated: among candidates scored around 0.8, far fewer than 80% are real.
2. **H2.** A simple calibration fitted on *other* map sheets fixes that without changing the ranking.
3. **H3.** Once calibrated, the probabilities can be added up per map sheet to predict how many candidates an interpreter will confirm, before anyone looks. The raw score can't do that.

## 3. Data

For 2024 I have something precious: interpreters went through the test map sheets and verified which construction sites are actually active.
- **Candidates:** the model produced **1,601 polygons on 26 map sheets** (each sheet 12.5 × 10 km). 131 of them lie on serviced-but-unbuilt plots or below the minimum mapping unit. The CBS rules exclude those, so they're neither right nor wrong and I left them out. That leaves **1,470**.
- **Truth:** a candidate counts as **confirmed** if it overlaps a site an interpreter verified as active. **317** of the 1,470 are confirmed: 22%.

## 4. Method

- **Calibration.** I tried two standard methods:
  - *Platt scaling*: a logistic curve on the score, with two parameters.
  - *Isotonic regression*: a step function that only has to go up.
- **Validation.** Each calibrator is fitted on four-fifths of the map sheets and tested on the rest, rotating until every sheet has been tested once. Whole sheets, never individual candidates: neighbouring candidates share a flight, a season and often a building project, and splitting them would leak.
- **Measuring calibration.** I sort candidates by score, cut them into ten equal groups, and compare the average score in each group with the share that was confirmed. The average gap is the *expected calibration error* (ECE). Zero means honest.
- **Uncertainty.** Every interval is a bootstrap over whole map sheets (2,000 resamples).

## 5. Results

### 5.1 The score was badly overconfident (H1: yes)

![Reliability diagram](figures/reliability.png)

If the score were honest, the dots would sit on the grey diagonal. The clay line is what I had actually shipped:
- candidates scored around 0.55 were confirmed about **4%** of the time
- candidates scored around 0.8 were confirmed about **42%** of the time
- even the most confident tenth, averaging 0.87, was right only **68%** of the time

The calibration error is **0.46** (95% interval 0.42–0.49). For a number that's supposed to be between 0 and 1, that's about as wrong as it can usefully be.

The cause turned out to be in my own definition of the score:

![Score histogram](figures/score_histogram.png)

A pixel only joins a polygon if its probability is above 0.30, and the score is the average of those pixels. So a candidate simply can't score low: the lowest in the whole set is 0.39. The score works as a *ranking*, and nothing about how I built it ever forced it to be a probability.

### 5.2 Two parameters fix it (H2: yes)

The teal line in the first figure is the same detector after Platt scaling, fitted and tested on different sheets. The calibration error drops from **0.46 to 0.02**. The ranking barely moves (AUC 0.823 → 0.819), because I didn't make the model any smarter. I just made it describe itself honestly.

Isotonic regression calibrates just as well (0.02) but loses a little more ranking quality (AUC 0.811), because it merges scores into ties. With the same calibration and fewer parameters, Platt is the better choice here.

### 5.3 The sums become useful (H3: yes)

![Sheet counts](figures/sheet_counts.png)

| adding up | predicted confirmations | actually confirmed | typical error per sheet |
|---|---|---|---|
| the raw scores | 991 | 317 | 26, and always too high |
| the calibrated probabilities | 319 | 317 | about 4, with no bias |

The raw scores overestimated every single sheet, by about three times. The calibrated probabilities get the total within two, and a typical sheet within about four.

One thing to be careful about: this counts **candidates that turn out to be real**, not construction sites. One site can be hit by two candidates, and some sites aren't flagged at all (these 26 sheets hold 420 verified sites). So the sum tells you how much true work is in the list, not how much construction exists. Those are two different estimates, and only the first is solved here.

### 5.4 A surprise: the second stage was already honest

I checked the re-ranker's scores fully expecting the same problem. They were already calibrated (error 0.02) before I touched them. In hindsight that makes sense. The re-ranker is a classifier trained to output probabilities on exactly this kind of data, so that's what it learned to produce. The detector's score was assembled by hand.

## 6. Reflections

1. **A confidence column is a promise to whoever reads the table.** I treated it as a sort key, because that's how I used it. But I'm not the only reader, and the others will do arithmetic with it.
2. **Scores you compute are not probabilities unless something trained them to be.** The hand-built score was off by a factor of three. The trained one was fine. I'll remember that the next time I invent a "confidence".
3. **This was the cheapest improvement the model has had.** No new data, no retraining, no GPU. Two parameters and one honest plot. I wish I'd drawn that plot on the first day.
4. **It will drift, so I'll keep checking.** The biggest lesson this model has taught me so far is that something trained on one flight year can quietly break on the next, and a calibrator is a model too. My plan is to refit the two parameters each year on the interpreters' review set. Two parameters can be estimated from a few dozen reviewed candidates; anything more flexible would need far more.

## 7. Limitations

1. One year and 26 sheets. The intervals cover variation between sheets, not between years.
2. "Confirmed" means the candidate overlaps a verified site. That's generous, since clipping a corner counts, but it's the same rule for every score compared.
3. The labels are CBS-internal. The code and every aggregate number are public; the candidate table is not.

## 8. Details and reproduction

<details>
<summary><strong>Full numbers, the export format, and how to rerun it</strong></summary>

**All scores, out-of-fold.** Bracketed ranges are 95% intervals from a sheet bootstrap.

| score | AUC | ECE [95% CI] | Brier |
|---|---|---|---|
| detector, as shipped | 0.823 | 0.458 [0.421, 0.491] | 0.348 |
| detector + Platt (2 parameters) | 0.819 | 0.022 [0.019, 0.050] | 0.129 |
| detector + isotonic | 0.811 | 0.024 [0.018, 0.054] | 0.128 |
| re-ranker, untouched | 0.879 | 0.022 [0.017, 0.053] | 0.104 |
| re-ranker + isotonic | 0.869 | 0.022 [0.017, 0.047] | 0.105 |

**Publishing it the way ALU does.** [`src/alu_schema.py`](src/alu_schema.py) writes every candidate as one GeoJSON feature with:
- `id`: a plus code of the centroid (10 digits, about 14 m), computed with exact integer arithmetic and checked against the reference example `8FVC9G8F+6X`
- `area_m2`
- `score_raw`
- `p_site`: the calibrated probability, so that `sum(p_site)` is an estimate rather than a guess

**Reproduce.**

```bash
pip install -r ../requirements.txt
python src/calibrate.py --candidates <detector>/data/cand_2024_test_feat_scored.csv
python src/alu_schema.py --candidates <detector>/data/cand_2024_test_feat_scored.csv \
    --geometry <detector>/data/cand_2024_test.gpkg --out data/candidates_alu.geojson
```

About 20 seconds on a laptop CPU. The script draws the figures too. Aggregates are in [`results/`](results/), with sheet IDs anonymised.

</details>
