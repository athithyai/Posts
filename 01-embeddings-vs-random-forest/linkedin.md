"I just wish there would be an embedding which is better than a random forest."

That line from an ECCV 2026 reflection has been in my head all week. It sums up three years of geospatial foundation models: they win the benchmarks, and then practitioners go back to a random forest because it just works.

So I tried it on my own work.

One window of the Dutch coast near Lelystad, 2020, at 20 m. The official CBS land-use map as the labels. Two contenders:

A random forest on 46 hand-built features from 56 clear Sentinel-2 dates. The recipe every course teaches.

A plain linear model on a precomputed foundation-model embedding (Spheer). 100 numbers per pixel. Nothing else.

Same pixels, same labels, whole 1 km blocks held out for testing.

The embedding won. Macro-F1 0.592 against 0.551.

Real (the paired 95% interval stays above zero), but modest: four points.

What I didn't expect:

The win came from the inputs, not the model. A linear model on the Sentinel-2 features did no better than the forest.

The gap didn't grow when labels got scarce. I was sure it would. But the embedding with 50 labels per class still beat the forest with 1,000. That's twenty times less labelling.

And it isn't better everywhere. It's much better on sports grounds, parks and shops: classes you recognise by layout, not colour. It's worse on wet nature, where the seasonal flooding is the signal.

So yes, here there is an embedding better than a random forest. And the best way to use it was the simplest one: no fine-tuning, no GPU.

Two honest caveats. It's one window, mostly lake. And "compute-equitable" moved my cost from processing to downloading: one tile-year is about 50 GB.

Full story, figures and code: https://github.com/athithyai/Posts/tree/main/01-embeddings-vs-random-forest

#ECCV2026 #EarthObservation #FoundationModels #GeoAI
