My model said 0.8. It meant 0.4.

This week a post about Google's ALU went past my feed. Every field it maps comes with an ID, an area, and a confidence between 0 and 1.

Good design. And it made me a bit uncomfortable, because my own construction-site detector for the Dutch land-use map does the same thing, and I had never checked whether its number meant anything.

So I checked.

1,470 candidates on 26 map sheets, each one compared with what photo-interpreters actually confirmed.

The average confidence was 0.67.
The share that turned out to be real was 22%.
When the model said 0.8, it was right about 40% of the time.

The cause was in my own code. A candidate is built from pixels that already passed a threshold, and the score is their average. So it can't be low. It's a fine way to sort a list. It was never a probability.

The fix took two numbers. A simple calibration, fitted on some map sheets and tested on others:
– calibration error 0.46 → 0.02
– ranking unchanged

Here is why I care. In a statistics office, someone will add that column up.

Adding up my original scores predicted 991 confirmed candidates. The interpreters confirmed 317.
Adding up the calibrated ones predicted 319.

Same model, same candidates. One number you can plan with, one you can't.

What I'm taking away: a confidence column is a promise to whoever reads the table. I used mine as a sort key and never asked what it promised to everyone else.

Next: refit those two numbers every year, because a calibrator drifts just like a model does.

Full story, figures and code (the labels are CBS-internal and stay in-house):
https://github.com/athithyai/Posts/tree/main/03-calibrated-confidence

If you publish a confidence column: do you know what your 0.8 means?

#EarthObservation #GeoAI #OfficialStatistics #MachineLearning
