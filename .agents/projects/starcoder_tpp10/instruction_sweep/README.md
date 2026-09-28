# Dolmino FLAN (instruction) TPP-10 domain sweep (started 19 September 2026)

Purpose: a Figure 4 domain with a low optimal epoch count on its on-target evaluation. The swarm
marginal-optimum check (`two_phase_many/reference_outputs/marginal_epoch_optima_20260919/`) found
that the `dolmino_synth_instruction` bucket, which is the Dolmino pool's FLAN component, has the
lowest marginal optima of any bucket on the Delphi 3e18 swarm: about 1 epoch on CSQA, SciQ and
DROP with 2 to 5 percent benefit and 4 to 7 percent excess by 16 epochs, against 5 to 13 epochs
for every scarce natural-text domain (math, code, arXiv) on its own evaluation.

Everything except the domain text and the added evaluations is identical to the Wikipedia,
FineMath and arXiv sweeps: the frozen TPP-10 design, the 190,316,544-token parent pool with the
frozen parent permutation, one nested 10,485,760-token matched subset (seed 20260912), the frozen
allocator and Nemotron-CC blend, trainer seed 20260910, and the grid p = 0, 5, 10, 20, 30, 50, 70, 100.

## Domain source

Dolma 3 Dolmino pool component `dolmino_1-flan` (`gs://marin-us-central1/raw/dolma3_dolmino_pool-72089d/data/dolmino_1-flan/`,
206 zstd JSONL shards, 26.8 GB). Four shards chosen by the seeded rule of the earlier domains
(seed 20260916, ascending sha256 of "seed:uri"): `tulu_flan-0020`, `-0206`, `-0131`, `-0207`; the
fifth in that order, `tulu_flan-0133`, is the held-out evaluation source. Pins in
`experiments/domain_phase_mix/tpp10_instruction_sweep_assets/sources.json`. The zstd reader
(`prepare_tpp10_instruction_sweep.zstd_documents`) mirrors the frozen gzip reader with the same
pinned generation, size and CRC32C checks and the same bounded read budget.

## Evaluations at every point (final step, full sets)

- Seven Uncheatable components and Paloma programming languages, as before.
- Five QA validation sets rendered on 19 September as one document per example,
  `Question: <question>\nAnswer: <correct answer text>\n`, from the pinned allenai parquet snapshots
  in central1: ARC-Easy (570), ARC-Challenge (299), OpenBookQA (500), QASC (926), SciQ (1000).
  Rendered gzip files under `gs://marin-us-central1/raw/tpp10_qa_evals/2026.09.19/`. This is
  document BPB over question plus answer, not the Table 9 continuation-only BPB.
- The held-out FLAN shard, first 4,194,304 tokens (2,048 sequences).

On-target candidates: the five QA sets and the held-out FLAN shard. Expected from the swarm: an
optimum near 1 to 2 epochs on the QA sets with a visible rise past 4 epochs.

## Stage 1: proxy sweep (submitted 06:19:53 UTC, 19 September)

- Iris parent `/calvinxu/tpp10-instruction-proxy-sweep`, us-central1-a, 2 CPU / 8 GB coordinator,
  non-preemptible, interactive, 48 h timeout, no automatic retries; preparation tasks at
  2 CPU / 4 GiB (built into the module after the arXiv recovery); children inherit the interactive
  band and request v5p-8. Bundle 21.1 MB.
- Plan `plan_proxy.json` (sha256 ce880b54…), release `release_proxy.json`, exact command
  `submit_proxy.sh`, log `submission_proxy.log`. Durable copies under
  `gs://marin-us-central1/experiments/tpp10_instruction_sweep/<plan_sha256>/`.
- Order: raw pool, parent, matched subset, six evaluation caches, then the p=100 execution check,
  then the remaining seven points. Training compute 1.99e17 FLOPs.
- Monitor: `monitor_proxy.sh` (nohup) writes `monitor_proxy.log` every 15 minutes and collects
  into `results/` when the parent is terminal. Manual collection:

```bash
MARIN_PREFIX=gs://marin-us-central1 uv run --offline --no-sync python -m experiments.domain_phase_mix.launch_tpp10_instruction_sweep --plan .agents/projects/starcoder_tpp10/instruction_sweep/plan_proxy.json --collect .agents/projects/starcoder_tpp10/instruction_sweep/results/measurements.csv
```

## Stage 2: target sweep (authorized 19 September, submitted 14:30 PDT, collected 20:10 PDT)

Iris parent `/calvinxu/tpp10-instruction-target-sweep` (plan `plan_target.json`, sha256 0e4fb759…, release
`release_target.json`, script `submit_target.sh`): the same eight points for the 301M target, 6.7e18 FLOPs
each. The p=100 execution check ran alone from 14:45 to 16:55 PDT (2 h 17 min of training at 1.4 steps/s
plus evaluations); the seven remaining points then all found v5p-8 slices at once and finished by 19:40
PDT. Results in `results_target/` (measurements, analysis, `instruction_target_vs_proxy_on_target.png`).

### Target results

All five QA-format evaluations minimize at p = 10 percent, 1.58 epochs, exactly where the proxy put them,
and the basin is much sharper than at proxy scale: the 7.9-epoch point is 12 to 15 percent worse than the
minimum (proxy: 0.6 to 4.4), 11.1 epochs 10 to 21 percent (proxy 5 to 7), 15.8 epochs 35 to 60 percent.
The p=20 (3.2 ep.) run sits slightly above the p=30 (4.8 ep.) run on every QA set, which single runs cannot
distinguish from seed noise. The in-domain FLAN held-out shard keeps improving to the p=100 boundary
(143 percent depth); Uncheatable macro and Wikipedia English minimize at 0.8 epochs with 1.5 and 0.8
percent depth. Excess over the grid minimum:

| Evaluation | target optimum | depth (p=0) | 0.8 ep. | 1.6 ep. | 3.2 ep. | 4.8 ep. | 7.9 ep. | 11.1 ep. | 15.8 ep. | proxy optimum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SciQ | 1.6 | 7.5% | +4.2% | 0 | +4.0% | +2.8% | +14.9% | +12.3% | +49.3% | 1.6 |
| ARC-Easy | 1.6 | 3.4% | +2.4% | 0 | +3.4% | +2.0% | +13.7% | +19.2% | +55.1% | 1.6 |
| ARC-Challenge | 1.6 | 3.0% | +2.1% | 0 | +4.0% | +2.7% | +13.3% | +21.0% | +60.1% | 1.6 |
| OpenBookQA | 1.6 | 3.4% | +1.9% | 0 | +1.2% | +0.8% | +11.7% | +14.6% | +46.4% | 1.6 |
| QASC | 1.6 | 6.6% | +5.1% | 0 | +4.3% | +1.8% | +15.1% | +9.7% | +34.9% | 1.6 |
| FLAN held-out shard | 15.8 | 143.2% | +108.2% | +98.7% | +80.7% | +60.1% | +28.1% | +8.4% | 0 | 11.1 |
| Uncheatable macro | 0.8 | 1.5% | 0 | +0.5% | +3.4% | +6.7% | +12.3% | +17.0% | +194.5% | 0.8 |

For Figure 4 this is the low-epoch domain: a 1.6-epoch optimum with steep harm past five epochs at
target scale, against 11.1 for FineMath on GSM8K and StarCoder on code.

## Code

- `experiments/domain_phase_mix/prepare_tpp10_instruction_sweep.py`: zstd reader, pool steps, evaluation caches.
- `experiments/domain_phase_mix/launch_tpp10_instruction_sweep.py`: plan, submission, collection.
- `experiments/domain_phase_mix/analyze_tpp10_arxiv_sweep.py`: shared per-evaluation minima and excess.
- `tests/test_tpp10_instruction_sweep.py`: zstd decoding bounds, stream equality with the frozen launcher, evaluation tags, metrics.

## Recovery, 19 September 19:39 UTC: bundle must keep .html

The first parent started at 19:03 UTC once the CPU pool freed up, but its preparation tasks would
have failed at cache consolidation like the arXiv one did (`FileNotFoundError:
/app/lib/zephyr/src/zephyr/dashboard/index.html`, because the bundle excluded `.html` and the
nested Zephyr cache-probe coordinator needs that file). Cancelled and resubmitted as
`/calvinxu/tpp10-instruction-proxy-sweep-r2` (log `submission_proxy_r2.log`) with `html` dropped
from the media exclude; plan, release and resources unchanged.

## Recovery 2, 19 September 20:05 UTC: the branch's Iris bundler excluded .html

See the arXiv README, recovery 3: the branch-local `DEFAULT_EXCLUDE` in Iris's bundler dropped
`.html`, which the Zephyr cache-probe coordinator needs. `\.html$` removed from it; resubmitted as
`/calvinxu/tpp10-instruction-proxy-sweep-r3` (log `submission_proxy_r3.log`); plan, release and
resources unchanged.

## Proxy results (collected 21:35 UTC, 19 September; parent -r3 succeeded)

All five QA-format evaluations have an interior minimum at p = 10 percent, 1.58 epochs; the
in-domain held-out FLAN shard minimizes at 11.1 epochs. Excess over the minimum on the grid
0 / 0.8 / 1.6 / 3.2 / 4.8 / 7.9 / 11.1 / 15.8 epochs:

| Evaluation | benefit depth (p=0) | 0.8 ep. | 1.6 ep. | 3.2 ep. | 4.8 ep. | 7.9 ep. | 11.1 ep. | 15.8 ep. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciQ | 10.9% | +4.6% | 0 | +1.2% | +1.2% | +3.1% | +6.3% | +38.9% |
| ARC-Easy | 9.3% | +3.0% | 0 | +1.0% | +1.1% | +0.6% | +5.5% | +30.6% |
| ARC-Challenge | 7.8% | +2.2% | 0 | +1.1% | +0.7% | +1.0% | +5.0% | +30.6% |
| OpenBookQA | 11.1% | +3.8% | 0 | +0.9% | +1.7% | +2.4% | +7.3% | +31.6% |
| QASC | 11.9% | +5.2% | 0 | +1.5% | +2.2% | +4.4% | +7.4% | +37.9% |
| FLAN held-out shard | 18.5% | +10.4% | +7.6% | +4.1% | +2.5% | +0.2% | 0 | +9.6% |

The swarm's prediction (about 1 epoch on QA evaluations, 2 to 5 percent depth) is confirmed in
location and exceeded in depth. The retrained p=0 run reproduces the arXiv sweep's p=0 exactly
(Paloma code 1.7792 in both; the archived 11 September run had 1.7713 under the older lock).
Plot: `results/instruction_proxy_on_target.png`.

### Overlap audit of the QA evaluations against the FLAN pool (20 September, 20:55 PDT)

Whitespace-normalized, case-folded exact substring search of every evaluation question (25+ characters)
over the first 260 MB of decompressed text of each of the four pool shards (a superset of the 47.6M
tokens per shard the pool read). Validation questions (what we score): SciQ 1 of 1000 (inside a
"if I tell you that ..., and ask you the question ..." template), QASC 0 of 926, ARC-Easy 0 of 570,
ARC-Challenge 0 of 299, OpenBookQA 3 of 500 (283 documents). Training-split questions (whether the
task itself is in FLAN): SciQ 5 of 11,645, QASC 3,336 of 7,466 (1,636 documents). QASC is therefore a
FLAN task in these shards; SciQ is not. SciQ is also one of the 51 OlmoBaseEval Easy tasks and the
evaluation on which the swarm predicted the low optimum before the sweep; QASC is in neither.

## Stage 3: unmatched proxy (no simulated epoching), submitted 23:03 PDT 19 September, collected 23:45 PDT

Iris parent `/calvinxu/tpp10-instruction-unmatched-sweep` (plan `plan_unmatched.json`, sha256 9aca28e0…): the
16.6M proxy trained on the full 190,316,544-token parent pool (no downsampling; at most 0.87 epochs of FLAN at
p=100), eight points including a retrained p=0, first launch with the early-release gate. Results in
`results_unmatched/`. On SciQ the unmatched curve has an interior minimum at p=20 (1.4566 BPB; 1.4813 at p=10,
1.4881 at p=30) driven by web displacement rather than repetition, while the matched proxy minimizes at p=10
(1.4703). Read on the target curve, the unmatched pick (p=20, 3.2 epochs) costs +4.0% on SciQ against 0 for the
simulated-epoching pick. At p=100 the unmatched proxy sits at 1.58 versus 2.04 for the matched proxy, the gap
that repetition harm opens.
