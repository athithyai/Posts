My model said 0.8. It meant 0.4.

Every polygon my construction-site model draws on Dutch aerial photos comes with a confidence between 0 and 1. I put that column there myself, and I used it to sort the list.

I never asked what it means. So this week I checked.

The model: a frozen DINOv2 vision transformer with a small trained decoder (1.2M parameters). It was trained on the 2022 land-use map and run on 2024 imagery it had never seen.

The test: 1,470 candidates on 26 map sheets, each compared with what photo-interpreters actually confirmed.

1. Average confidence: 0.67. Share confirmed: 22%.
2. When the model said 0.8, it was right about 40% of the time.
3. Adding up the scores predicted 991 confirmed candidates. There were 317.

The cause was in my own code. A polygon is built from pixels that already passed a 0.30 threshold, and the score is their average, so it can't be low. A fine way to sort a list. Never a probability.

The fix: two parameters. Platt scaling, fitted on some map sheets and tested on others.
→ calibration error 0.46 → 0.02
→ ranking unchanged (AUC 0.823 → 0.819)
→ summed probabilities: 319 predicted, 317 confirmed

The surprise: the second stage of the model, a gradient-boosted re-ranker, was already calibrated. It had been trained to output probabilities. The detector's score was assembled by hand.

What I'm taking away: a confidence column is a promise to whoever reads the table. In a statistics office, someone will add it up.

Next: refit those two parameters every year, because a calibrator drifts like any other model.

Full write-up, figures and code (the labels are CBS-internal and stay in-house):
https://github.com/athithyai/Posts/tree/main/03-calibrated-confidence

#EarthObservation #GeoAI #OfficialStatistics #MachineLearning
