My model said 0.8. It meant 0.4.

Google's ALU publishes every field it maps with an ID, an area and a confidence between 0 and 1. That's the right design: a statistician can filter it, sort it, add it up.

My construction-site detector for the CBS land-use map does the same thing. This week I finally asked the question I'd skipped: does its confidence mean anything?

I took 1,470 candidates on 26 map sheets from 2024 and checked them against what photo-interpreters actually confirmed.

→ Average stated confidence: 0.67. Share actually real: 22%.
→ Candidates scored "0.8 or higher" were right 60% of the time.
→ Summing the scores predicted 991 construction sites. There were 317.

The reason was in my own code. The score is the mean probability of pixels that had already passed the 0.5 threshold, so it can't go below 0.5. It was a sort key dressed up as a probability.

The fix took two parameters. Platt scaling, fitted on four folds of map sheets and tested on the fifth:
→ calibration error 0.458 → 0.022
→ ranking unchanged (AUC 0.823 → 0.819)
→ summed probabilities: 318 expected vs 317 real, about 3 sites off per sheet, no bias

That last line matters most for official statistics. A calibrated layer can estimate how much construction a sheet holds before anyone opens it. A raw score can't.

Two honest caveats:
1. This is one year. My detector already taught me that models break across flight years, and a calibrator is a model too. The plan is to refit the two parameters every year on the interpreter review set.
2. "Real" means the candidate overlaps a verified site. That's generous, but it's the same rule for every score compared.

The cheapest improvement this project has had: no retraining, no new data, no GPU. One function and one honest plot.

Code, figures and every aggregate number (the labels are CBS-internal, so they stay in-house): https://github.com/athithyai/Posts/tree/main/03-calibrated-confidence

If you publish model outputs with a confidence column: have you checked what your 0.8 means?

#EarthObservation #GeoAI #OfficialStatistics #MachineLearning #Calibration #RemoteSensing
