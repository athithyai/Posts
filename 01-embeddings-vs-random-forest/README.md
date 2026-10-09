# Is there an embedding better than a random forest?

*Experiment 01 in the [Posts](../README.md) series.*

---

There's a sentence I keep running into in remote sensing, in one form or another. This year it came out of a conversation at ECCV:

> "I just wish there would be an embedding which is better than a random forest."

It describes my field rather precisely. For three years we've been building geospatial foundation models: big networks pretrained on millions of satellite images, which top the benchmarks. And when real work needs doing (a land-cover map for a ministry, a habitat layer for an ecologist), people quietly go back to a random forest on a few band medians. It's cheap, it's well understood, and it works.

The hopeful version of the story is *precomputed embeddings*: someone runs the big model once over a whole country and publishes the result, a short vector of numbers for every pixel, so that everyone else can skip the expensive part. I work on land use for official statistics, where the GPU budget is roughly zero and the cost of being wrong is high. If that promise holds anywhere, it should hold for me.

I have one such embedding on my disk, and I have the map a random forest would be up against. So I set up the fight.

## 1. The question and the hypotheses

**Question.** On the same pixels, with the same labels, does a precomputed foundation-model embedding beat a random forest on hand-built Sentinel-2 features?

1. **H1.** A simple classifier on the embedding scores at least as well as a random forest on Sentinel-2 features.
2. **H2.** The embedding's advantage grows as labels get scarce. This is the usual promise of a foundation model.
3. **H3.** The advantage is concentrated in land-use classes defined by *context and arrangement* (parks, sports grounds), not in classes defined by colour (water).

## 2. The contenders

**The embedding: Spheer FM "Albatross".** A foundation model trained without labels on Sentinel-2 imagery. It summarises a whole year of satellite observations into **100 numbers per pixel** at 10 m. The embeddings are published on Hugging Face ([spheer/spheer-fm-embeddings](https://huggingface.co/datasets/spheer/spheer-fm-embeddings), gated) for the Netherlands, 2017–2020. I don't compute anything from imagery on this side; I only read the 100 numbers.

**The baseline: hand-built Sentinel-2 features.** What any remote-sensing course would teach.
- I downloaded every Sentinel-2 L2A scene of 2020 over the same area: 64 scenes under 60% cloud, of which **56 dates** were usable once clouds, shadows and cirrus were masked.
- For each pixel I computed the median of all 10 bands in each quarter of the year, the median vegetation index (NDVI) in each quarter, and the 10th and 90th percentile of NDVI over the year.
- That makes **46 features per pixel**.

**The classifiers.** All from scikit-learn, all left at sensible defaults, nobody tuned:
1. **Random forest**, 500 trees, on the Sentinel-2 features. This is the baseline.
2. Random forest, 500 trees, on the embedding.
3. Random forest on both feature sets together.
4. **Logistic regression** (a linear model) on the embedding, with standardised inputs.
5. Logistic regression on the Sentinel-2 features.
6. **k-nearest neighbours** (k = 15, cosine distance) on the embedding.

## 3. Data

- **Area.** An 11.2 × 11.5 km window on the Lelystad coast: the Markermeer, the Marker Wadden islands, the Houtribdijk and the town of Lelystad. It's the one Spheer extract I have locally. The embeddings are large (about 50 GB per tile per year at full resolution) and slow to download, so I worked at their 20 m overview level.
- **Labels.** The official **CBS land-use map** for 2020 (NBBG2020, public). This is a map of land *use*, not land cover. It doesn't say "grass"; it says "park", "sports ground", "residential". That makes it hard, and that's why I wanted it.
- **Grid.** Every Sentinel-2 scene was resampled onto exactly the embedding's 20 m grid (561 × 574 pixels), so both sides describe the same pixels.
- **Cleaning.** Pixels on the border between two land-use classes are mixed, so I dropped any pixel whose 3 × 3 neighbourhood contains more than one class. That leaves **297,911 pixels**.
- **Classes.** I kept the **11 classes** with at least 300 pixels in at least 3 places. Agriculture, roads and construction exist here but are too small to evaluate.

One caveat up front, because it shapes everything else: **89% of the window is the Markermeer lake.** Every model gets the water right, so overall accuracy is useless here (everyone scores about 0.96). I compare models on **macro-F1**, which gives each of the 11 classes an equal say.

## 4. Method

1. **No random pixel splits.** Neighbouring pixels are nearly identical, so a random split lets a model peek at the answer. I cut the area into 1 km blocks, dealt them into five groups, and held out one group at a time. A 100 m buffer around the held-out blocks was removed from training too. Every pixel is predicted by a model that never saw its block.
2. **Balanced training.** Each model trains on up to 2,000 pixels per class, drawn from the training blocks.
3. **Paired comparison.** To check that a difference isn't luck, I resampled the blocks 1,000 times and scored both models on the *same* resampled blocks each time. Hard blocks then count against both, and only the real difference between the models remains.
4. **Label budget.** To test H2, I retrained with 5, 10, 25, 50, 100, 250 and 1,000 labelled pixels per class, five random draws each.

## 5. Results

### 5.1 The embedding wins, modestly (H1: yes)

![Paired difference](figures/paired_difference.png)

The plain **linear model on the 100 embedding numbers** reaches macro-F1 **0.592**. The **random forest on 46 Sentinel-2 features** reaches **0.551**.

The difference is **+0.041**, with a 95% interval of +0.006 to +0.066. It doesn't include zero, so the win is real. It's also not a leap: four points of F1.

Two details surprised me more than the headline:
- **The win comes from the inputs, not the classifier.** A linear model on the Sentinel-2 features is no better than the forest (+0.003). Swapping the *features* is what moves the result.
- **The simplest classifier was the best one.** A random forest on the embedding (+0.026) does slightly worse than a straight line on it (+0.041). Adding the Sentinel-2 features on top adds nothing (+0.039). Whatever these classes need is already in the 100 numbers, readable with a ruler.

### 5.2 The gap doesn't grow when labels are scarce (H2: no)

![Label efficiency](figures/label_efficiency.png)

I fully expected the embedding to pull away as training data shrank. It didn't. It stays about three to four points ahead all the way from 5 labels per class to 1,000. That's a less exciting result than I hoped for, and I think it's worth saying out loud.

Read from the other side, it's still useful. **The linear model with 50 labelled pixels per class (0.579) beats the random forest with 1,000 (0.548).** If labels are what you pay for, and in my work they are, that's twenty times less labelling for a slightly better map.

### 5.3 It helps where layout matters, and hurts where seasons matter (H3: yes, with a twist)

![Per-class difference](figures/per_class_delta.png)

| class | random forest, Sentinel-2 | linear model, embedding | difference |
|---|---|---|---|
| sports grounds | 0.33 | 0.57 | **+0.25** |
| retail & catering | 0.04 | 0.18 | +0.14 |
| park | 0.51 | 0.64 | +0.13 |
| wet nature | 0.86 | 0.72 | **−0.14** |

The embedding wins on classes a person recognises by *arrangement*. A sports field is grass with lines next to a car park. A park is grass with paths and scattered trees. A single pixel's colour can't tell you that, but its neighbourhood can, and the embedding seems to have learned something about neighbourhoods.

The twist is **wet nature**, the reed beds and marshes of the Marker Wadden, where the forest is clearly better. That class is defined by water over the seasons: flooded in winter, green in summer. My quarterly features spell that out explicitly. A single annual summary may smooth it away. That's a hypothesis I haven't tested, but it fits.

On the map, teal is where only the embedding was right and clay is where only the forest was:

![Maps](figures/maps.png)

Across the town, the embedding fixes about 2,400 pixels the forest got wrong and breaks about 1,300 that the forest got right. It's better overall, but not uniformly better.

## 6. Reflections

1. **Here, there is an embedding better than a random forest, and the best way to use it is the simplest one.** No fine-tuning, no GPU, a model that trains in seconds. That's the version of a foundation model a statistics office can actually adopt, and I find it genuinely encouraging.
2. **"Better" is four points, not a new era.** Every model here lands between 0.55 and 0.6. CBS classes describe what land is *used for*: "retail" is a functional category, and a 20 m pixel can only see part of it. No representation of the pixels can recover a label that isn't in them.
3. **"Compute-equitable" moved my cost rather than removing it.** The embedding saved me the whole Sentinel-2 pipeline: finding scenes, masking clouds, compositing, designing features. But at the download speed I get, one small window was all I could afford. Precomputed embeddings are only as equitable as the access to them.
4. **The best practical answer is probably both.** The per-class split suggests embedding *plus* a handful of physically meaningful seasonal features, kept for exactly the classes where the seasons are the signal.

What I really want to test next is agriculture and construction, the classes my work depends on. This window has almost none of either.

## 7. Limitations

1. One window, one year, mostly water. The intervals cover variation between blocks inside this window, not other regions.
2. 20 m instead of the native 10 m. Small urban classes would likely benefit from the finer grid, for both sides.
3. No tuning. A tuned forest might close part of the gap; so might a tuned linear model.
4. Pure pixels only, which makes the task easier than a wall-to-wall map, equally for every model.

## 8. Details and reproduction

<details>
<summary><strong>All numbers, the model on Hugging Face, and how to rerun it</strong></summary>

| model | macro-F1 | paired difference vs random forest on Sentinel-2 [95% CI] |
|---|---|---|
| Random forest, Sentinel-2 | 0.551 | baseline |
| Logistic regression, Sentinel-2 | 0.554 | +0.003 [−0.019, +0.024] |
| k-NN, embedding | 0.561 | +0.011 [−0.015, +0.041] |
| Random forest, embedding | 0.577 | +0.026 [+0.009, +0.046] |
| Random forest, both | 0.589 | +0.039 [+0.022, +0.054] |
| **Logistic regression, embedding** | **0.592** | **+0.041 [+0.006, +0.066]** |

**On Hugging Face.** The winning linear model is [AthithyaLogan/spheer-nbbg-linear-probe](https://huggingface.co/AthithyaLogan/spheer-nbbg-linear-probe): 11 × 100 weights plus the standardisation, in safetensors, applied with plain numpy (no pickle).

**Reproduce.**

```bash
pip install -r ../requirements.txt
python src/fetch_s2.py   --spheer <cache>/markermeer_31UFU_2020_ov1.npz      # ~200 MB, ~5 min
python src/experiment.py --spheer <cache>/markermeer_31UFU_2020_ov1.npz \
       --s2 data/s2_2020_scenes.npz --nbbg2020 <CBS_Publicatiebestand_NBBG2020_v3_def.gpkg>   # ~16 min CPU
python src/figures.py    --s2 data/s2_2020_scenes.npz
python src/export_hf.py
```

Seeds are fixed (24), and re-running reproduces every number exactly (checked). Results are in [`results/`](results/), including every Sentinel-2 scene ID used.

</details>
