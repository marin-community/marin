#### Problem

* We need to establish canonical Datakit parameters (e.g. deduplication strategy)
* We should provide a reusable, reasonable efficient and continuously updating testbed for data experimentation (e.g. mixing)

#### Solution

* Use [RFC-0: Datakit](https://docs.google.com/document/d/1kDSzONg32zv2VnCO4FJiMP0fcjRSjgP0uTDpI4_C4O0) pipelines to define roughly 1T token Datakit Testbed dataset, sampled proportionally (by provenance) from up-to-date raw inputs
  * 1T because we want at least 500B on the output after deduplication
  * Recomputed on update to canonical raw sources
* MoE experiment harness with simulated epoching and proportional mixing reusing the Grug MoE recipe
* Baseline (deliberately trivial):
  * No deduplication
  * Single constant quality score
  * Topic by provenance

The Testbed dataset will enable many experiments. The immediate questions:

* What is the canonical deduplication strategy?
* What is the canonical topic clustering?
* How many and what quality scores should we use?

#### Implementation

* Reuse existing pieces:
  * Proportional mixing
  * Grug MoE recipe
  * Dedup
  * Datakit ferry
* New work:
  * Revive and wire existing simulated epoching through the grug MoE
  * By provenance sampler on the normalized data

#### Experiment protocol

* Configuration ranking: train each configuration at a single compute-optimal baseline point, rank on Paloma macro
* Canonical confirmation: run IsoFLOP on winner configuration and baseline

#### Open Questions

* Should we use/start-with dense models to reduce complexity?
  * Will: MoE's will probably show more data dependent effects at the same FLOP count, so don't think this is necessary\!
* Is perplexity evaluation metric sufficient, or should we also use downstream task evals?
  * Will: Uncheatable is a good starting point \- MMLU logprob is not bad either
* How to handle distribution drift when the Testbed gets updated with new raw sources?
  * Test if the canonical configuration survives Testbed re-sampling?
