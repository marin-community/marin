# Hero data mixture phases

The hero starts on Harrier, shifts toward agent transcripts and low-level code,
then increases natural-science research during cooldown. All three phases sample
from the same 200 groups: 40 topic clusters × five quality buckets in `store_4d2e363d`.

- **Initial Harrier, 0–27.7%:** Software infrastructure, low-level code, and science
  lead the mixture, with substantial general software development, history, and math.
- **Selected phase 0, 27.7–80%:** More command-line agent transcripts, law, and world
  news; less general software development, finance, history, and math. The shift
  within software is toward agent transcripts and performance logs/low-level code.
- **Selected phase 1, 80–100%:** Natural-science research rises sharply. Low-level
  code and agent transcripts retain large shares, while general software development
  and math fall further. The highest classifier-rated quality bucket, Q4, rises
  from 22.9% in selected phase 0 to 28.0% in selected phase 1.

These are the shared hero/GB200 ladder schedule's approximate fractions of training
steps. The hero switches at steps **108,000** and **312,192** of 390,251; smaller
rungs round transitions to mixture-block boundaries. The earlier H100 comparison
in [#9126](https://github.com/marin-community/marin/issues/9126) switches from Harrier
to selected phase 0 at ~25% of training steps.

<details>
<summary>Selected topic shares and sources</summary>

Shares sum each topic's configured weights across its five quality buckets.
They describe sampling weights, not measured token counts. Labels summarize
clusters' dominant content; a cluster can contain other topics.

| Topic | Initial Harrier | Selected phase 0 | Selected phase 1 |
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

The initial weights are `phase0_weights` in
[`harrier_mix_2026_08_18.json`](../../experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.json).
The selected phases are `phase0_weights` and `phase1_weights` in
[`best_mixture_996f489106c7b922.json`](../../experiments/grug/moe_hero_ep/best_mixture_996f489106c7b922.json).
See the [cluster labels and examples](https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling)
and [all topic and quality shares](https://storage.googleapis.com/marin-public/held/h100-mix25-paloma/mixture-phases-2026.09.13.1/index.html).

</details>
