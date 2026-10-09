"I just wish there would be an embedding which is better than a random forest."

That sentence came out of a conversation at ECCV this year, and I've heard versions of it for three years. We keep building geospatial foundation models that win the benchmarks, and practitioners keep going back to a random forest because it just works.

So I tested it on the kind of map I work with: the official CBS land-use map. Lelystad coast, 2020, 20 m pixels.

In one corner: a random forest (500 trees) on 46 hand-built features from 56 clear Sentinel-2 dates. Seasonal band medians and NDVI. The recipe every course teaches.

In the other: a logistic regression on the Spheer foundation-model embedding. 100 numbers per pixel, precomputed. Nothing else.

Same pixels, same labels, whole 1 km blocks held out for testing.

1. The embedding won. Macro-F1 0.592 vs 0.551. The paired 95% interval stays above zero, so the win is real. It's four points, not a leap.
2. The win came from the inputs, not the model. A linear model on the Sentinel-2 features did no better than the forest.
3. The gap did not grow when labels got scarce. I was sure it would. But 50 labels per class on the embedding still beat 1,000 on the forest.
4. It's not better everywhere. Much better on sports grounds (+0.25 F1) and parks (+0.13), classes you recognise by layout. Worse on wet nature (−0.14), where seasonal flooding is the signal.

So, is there an embedding better than a random forest? Here, yes. And the best way to use it was the simplest one: a straight line, no GPU.

Two caveats. It's one window, mostly lake. And "compute-equitable" moved my cost from processing to downloading.

Full write-up, figures and code: https://github.com/athithyai/Posts/tree/main/01-embeddings-vs-random-forest

#ECCV2026 #EarthObservation #FoundationModels #GeoAI
