# Can diffusion models segment Dutch construction sites without supervision?

*Experiment 02 in the [Posts](../README.md) series. Prompted by [Diffuse2Seg](https://arxiv.org/abs/2609.06491).*

> **Status: running.** The benchmark is built and the hyperparameters are chosen on the dev tiles; the
> test-set runs are on the GPU now. The full write-up replaces this page when they finish.

**Question.** Diffuse2Seg reports that the self-attention of a text-to-image diffusion model can propose
masks for "anything", without labels, and beat earlier label generators by 4–7 points of average recall.
No code is public, so I reimplemented the core idea from the abstract and compare it with Segment
Anything and a classical baseline on Dutch aerial imagery.

**Benchmark (open data).** 200 tiles of 512 × 512 m at 1 m from PDOK 2022, with instance masks for
CBS NBBG2022 construction land (291 objects) and BAG buildings (3,189). It's on Hugging Face as
[AthithyaLogan/nl-aerial-proposal-bench](https://huggingface.co/datasets/AthithyaLogan/nl-aerial-proposal-bench).
20 tiles for development and 180 for test.

**Methods.** My diffusion-attention proposer ([`src/proposers.py`](src/proposers.py), SD 2.1 base), SAM ViT-H and
ViT-B automatic mask generation, and Felzenszwalb superpixels. Metric: COCO-style average recall at
K proposals per tile.

**So far (dev tiles only, used for choosing settings):** the noise timestep matters most (t = 100 of 1000
is best). The edge-preserving colour term only helps when it's weak (σ = 32); strong versions lower
recall. See [`results/tuning_dev_grid1.json`](results/tuning_dev_grid1.json) and
[`results/tuning_dev_grid2.json`](results/tuning_dev_grid2.json).
