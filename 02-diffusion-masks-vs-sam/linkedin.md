Can an image generator find construction sites without ever being shown one?

Every construction site my detector knows about was drawn by a person. Labels are the slowest, most expensive part of everything I build.

So the Diffuse2Seg idea caught my attention. Text-to-image diffusion models must understand where objects begin and end in order to draw them. Read their internal attention, and you get masks with no labels at all.

No code is public, so I rebuilt the idea from the abstract on Stable Diffusion 2.1 and tested it on Dutch aerial photos against Segment Anything (SAM).

The benchmark: 200 tiles of 512 × 512 m from the 2022 national orthophoto, with construction land from the CBS land-use map and buildings from the national register. All open data, and published on Hugging Face so anyone can rerun it.

What I found:

1. Construction sites: it depends on how you hold SAM. At default settings, diffusion beats SAM (AR@100 0.25 vs 0.21). Loosen SAM's quality filter and SAM wins (0.29), twice as fast.

2. Buildings: not close. SAM 0.29, diffusion 0.08. On buildings over 2,000 m² they're equal. On houses of 200–500 m², diffusion finds none. Its internal grid has 8 m cells, and a house is two or three of them.

3. The "edge-preserving" step made no measurable difference on the test tiles.

4. The metric quietly picks the winner. At 1,000 masks per image (the paper's choice), diffusion edges ahead because it simply proposes more. At 100, a budget a person might actually review, it doesn't.

My first headline ("diffusion beats SAM on construction sites") disappeared the moment I checked SAM's defaults. That's the lesson I'm keeping.

For my work today: SAM with looser filters. Diffusion attention is an interesting tool for big, soft-edged regions, not "segment anything" at 1 m.

The paper's bigger claim, training a segmenter on these label-free masks, is still untested. That's the next experiment.

Full write-up, figures and code: https://github.com/athithyai/Posts/tree/main/02-diffusion-masks-vs-sam
Benchmark: https://huggingface.co/datasets/AthithyaLogan/nl-aerial-proposal-bench

#ComputerVision #EarthObservation #GeoAI #SegmentAnything #DiffusionModels
