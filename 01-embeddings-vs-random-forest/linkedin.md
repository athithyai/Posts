"I just wish there would be an embedding which is better than a random forest."

That line from an ECCV 2026 reflection stuck with me, so I tested it on my own data.

Same 20 m pixels of the Lelystad coast. Same year (2020). Same labels: the official CBS land-use map. Two contenders:

🌲 A random forest on 46 hand-built Sentinel-2 features (quarterly medians and NDVI from 56 clear dates), the recipe every remote-sensing course teaches.
🧭 A plain linear model on the precomputed Spheer foundation-model embedding: 100 numbers per pixel, nothing else.

Validation on 1 km spatial blocks with a 100 m buffer. Never a random pixel split.

What I found:

→ Linear on the embedding: macro-F1 0.592. Random forest: 0.551. The paired difference is +0.041 (95% CI +0.006 to +0.066). Real, but a few points, not a leap.
→ The gain comes from the representation, not the classifier. A linear model on the Sentinel-2 features ties the forest.
→ With 50 labelled pixels per class, the embedding beat the forest trained on 1,000.
→ I expected the gap to widen with fewer labels. It didn't. It stayed at about 3–4 points at every budget.

The part I find most useful is per class. The embedding wins big where a class is defined by arrangement and use: sports grounds +0.25 F1, parks +0.13, shops +0.14. It loses on wet nature (−0.14), where seasonal flooding is the signal and explicit quarterly features still do better.

Two honest caveats:
1. One 11 × 11.5 km window, 89% water, 11 classes. No agriculture, no construction. That's the next experiment.
2. "Compute-equitable" embeddings moved my cost from processing to downloading: one tile-year at full resolution is about 50 GB.

So yes, there is an embedding better than a random forest here, and the best head for it was the simplest one.

Code, figures, every number: https://github.com/athithyai/Posts/tree/main/01-embeddings-vs-random-forest
The probe (safetensors, no pickle): https://huggingface.co/AthithyaLogan/spheer-nbbg-linear-probe

#ECCV2026 #EarthObservation #FoundationModels #GeoAI #RemoteSensing #Sentinel2
