# Hero mixture log

This records the hero's mixture in training order. All entries use
`store_4d2e363d`: 40 topic clusters × five quality buckets. Steps count completed
training updates; a mixture listed at step N applies after N updates have completed.

As recorded on September 14, 2026, the initial mixture is in use. Phases 2 and 3 are planned in
[PR #9162](https://github.com/marin-community/marin/pull/9162); this log does not
record them as deployed. Append future mixture changes with their effective step
and weight source.

## Phase 1 — Initial Harrier · step 0

Software infrastructure, low-level code, and natural-science research lead the
mixture, with substantial general software development, history, and math.

Weights: `phases[0].weights` (`initial`) in
[`harrier_mix_2026_08_18.json`](https://github.com/marin-community/marin/blob/7c996f66bfa7baf7bf81cb5c6d62cc8b8dc720fb/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.json).

## Phase 2 — Main mixture · planned step 108,000 (~27.7%)

Increase command-line agent transcripts, law, and world news. Reduce general
software development, finance, history, and math. Within software, the emphasis
shifts toward agent transcripts and performance logs/low-level code.

Weights: `phases[1].weights` (`main`) in
[`harrier_mix_2026_08_18.json`](https://github.com/marin-community/marin/blob/7c996f66bfa7baf7bf81cb5c6d62cc8b8dc720fb/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.json).
The relaunch is configured to restore permanent checkpoint `step-108000`.

## Phase 3 — Cooldown mixture · planned step 312,192 (~80%)

Increase natural-science research from 6.9% in phase 2 to 11.3%. Low-level code and agent
transcripts retain large shares; general software development and math fall
further. The highest classifier-rated quality bucket, Q4, rises from 22.9% in
phase 2 to 28.0% in phase 3.

Weights: `phases[2].weights` (`cooldown`) in the same JSON. This phase runs to
the configured end at step 390,251.

<details>
<summary>Selected topic shares and sources</summary>

Shares sum each topic's configured weights across its five quality buckets.
They describe sampling weights, not measured token counts. Labels summarize
clusters' dominant content; a cluster can contain other topics.

| Topic | Phase 1: initial | Phase 2: main | Phase 3: cooldown |
|---|---:|---:|---:|
| Performance logs and low-level code | 9.0% | 10.6% | 11.4% |
| Software infrastructure and security | 9.3% | 8.3% | 8.0% |
| Command-line agent transcripts | 1.3% | 6.0% | 6.6% |
| General software development and web code | 7.9% | 4.5% | 3.1% |
| Natural-science research | 8.8% | 6.9% | 11.3% |
| History, literature, and heritage | 6.9% | 3.7% | 3.7% |
| Finance, insurance, and markets | 5.6% | 2.4% | 2.8% |
| Mathematics problems and proofs | 4.3% | 1.8% | 1.4% |
| Law, courts, and regulation | 1.1% | 3.6% | 2.9% |
| Geopolitics and world news | 0.8% | 2.7% | 1.8% |

See the [cluster labels and examples](https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling)
and [all topic and quality shares](https://storage.googleapis.com/marin-public/held/h100-mix25-paloma/mixture-phases-2026.09.13.1/index.html).

The shared GB200 ladder switches at ~27.7% and ~80% of each run's configured
training steps, rounded to mixture blocks. The earlier [H100 comparison](https://github.com/marin-community/marin/issues/9126)
switches to the main mixture at ~25% of training steps.

</details>
