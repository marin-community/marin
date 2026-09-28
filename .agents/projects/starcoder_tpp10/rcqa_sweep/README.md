# Wikipedia-as-QA (wiki_to_rcqa) TPP-10 domain sweep (started 19 September 2026)

Purpose: the same-evaluation counterpart of the FLAN sweep. FLAN minimizes at 1.6 epochs on the
five QA-format evaluations (proxy, 19 September); in the Delphi swarm fits the synthetic-QA bucket
keeps helping the same evaluations up to the 64-epoch boundary with 10 to 18 percent benefit. Two
domains with optima of about 1.6 and 11 or more epochs on identical evaluations is the cleanest
form of "an epoch cap does not transfer across domains". The domain also pairs with the existing
Wikipedia prose sweep (same source, different form).

Everything except the domain text is identical to the other sweeps: the frozen TPP-10 design, the
190,316,544-token parent pool with the frozen parent permutation, one nested 10,485,760-token
matched subset (seed 20260912), the frozen allocator and Nemotron-CC blend, trainer seed 20260910,
and the grid p = 0, 5, 10, 20, 30, 50, 70, 100.

## Domain source

Dolma 3 Dolmino pool components `wiki_to_rcqa-part1/2/3` (22,399 zstd shards of about 0.25 MB,
5.6 GB; documents are `Passage: ... Question: ... Answer: ...` renderings of Wikipedia). The seeded
rule of the earlier domains (seed 20260916, ascending sha256 of "seed:uri") over all three parts
selects 1400 files for the pool, packed as four consecutive groups of 350 (one part each; measured
yield 0.86 tokens per compressed byte gives about 75M tokens per group against a 47.6M quota),
and the next 32 files as the held-out stream. Pins in
`experiments/domain_phase_mix/tpp10_rcqa_sweep_assets/sources.json`. The grouped reader
(`prepare_tpp10_rcqa_sweep.prepare_grouped_zstd`) streams a group's files in order through the
frozen tokenizer until the group's quota; each file keeps the pinned generation, size and CRC32C
checks of the zstd reader.

## Evaluations at every point (final step, full sets)

- Seven Uncheatable components (Wikipedia English is the prose-side reference) and Paloma
  programming languages, as before.
- The instruction sweep's five QA-format document sets (ARC-Easy, ARC-Challenge, OpenBookQA,
  QASC, SciQ), reused from `qa/<name>-tpp10/2026.09.19`.
- The held-out rcQA stream, first 4,194,304 tokens of the 32 pinned files.

On-target candidates: the five QA sets, the held-out rcQA stream, Uncheatable Wikipedia English.
Expectation: an optimum at 11 epochs or more on the QA sets with 10 percent or more benefit.

## Stage 1: proxy sweep (submitted 21:43 UTC, 19 September)

- Iris parent `/calvinxu/tpp10-rcqa-proxy-sweep`, us-central1-a, 2 CPU / 8 GB coordinator,
  non-preemptible, interactive, 48 h timeout, no automatic retries; preparation tasks at
  2 CPU / 4 GiB; children inherit the interactive band and request v5p-8. Bundle 21.2 MB.
- Plan `plan_proxy.json` (sha256 41a8f6ed…), release `release_proxy.json`, exact command
  `submit_proxy.sh`, log `submission_proxy.log`. Durable copies under
  `gs://marin-us-central1/experiments/tpp10_rcqa_sweep/<plan_sha256>/`.
- Order: raw pool (four grouped parts), parent, matched subset, held-out cache, then the p=100
  execution check, then the remaining seven points. Training compute 1.99e17 FLOPs.
- Monitor: `monitor_proxy.sh` (nohup) writes `monitor_results.log` every 15 minutes and collects
  into `results/` when the parent is terminal. Manual collection:

```bash
MARIN_PREFIX=gs://marin-us-central1 uv run --offline --no-sync python -m experiments.domain_phase_mix.launch_tpp10_rcqa_sweep --plan .agents/projects/starcoder_tpp10/rcqa_sweep/plan_proxy.json --collect .agents/projects/starcoder_tpp10/rcqa_sweep/results/measurements.csv
```

## Code

- `experiments/domain_phase_mix/prepare_tpp10_rcqa_sweep.py`: grouped zstd pool, held-out stream, evaluation caches.
- `experiments/domain_phase_mix/launch_tpp10_rcqa_sweep.py`: plan, submission, collection (pins the instruction module and its assets too, since the QA caches and zstd reader are shared).
- `tests/test_tpp10_rcqa_sweep.py`: grouping of the pool recipe, stream equality with the frozen launcher, evaluation tags, metrics.

## Recovery, 19 September 21:57 UTC: two pinned shards have unterminated zstd frames

The pool step (`prepare_grouped_zstd`, all four groups in one task) failed 13 minutes in with
`json.decoder.JSONDecodeError: Unterminated string` and no automatic retry, so the parent
`/calvinxu/tpp10-rcqa-proxy-sweep` failed; the held-out stream cache had already succeeded.
Streaming every pinned file through the sweep's reader (1432 files, 360 MB, from the laptop)
found exactly two whose zstd frame never terminates, both from the `part2/00038_*` batch that the
Dolmino tokenization also found corrupt in March: `00038_f285` (rank 829, group 3 of 4; 904
complete records, then a partial line) and `00038_f271` (rank 1387, group 4 of 4; 958 records).
Every other file decodes to 999 complete records, including the one undersized file
(`00033_f364`, 429 records, intact).

Fix: data only. `sources.json` now lists the two files under `excluded` with the reason and
replaces each at its own position with the next intact file in the seeded ranking after the
held-out block (rank 1433 `part1/00010_f270`, rank 1434 `part3/00051_f226`), so the other 1398
pool files and the held-out stream are unchanged. The reader code is untouched (the instruction
module is pinned by the running FLAN target sweep). Plan rebuilt as `plan_proxy_r2.json`
(sha256 c8051ae1…): the only differences from the failed plan are the sources.json pin and the
run fingerprints. `release_proxy_r2.json` is written with `approved: false` and
`submit_proxy_r2.sh` targets job `tpp10-rcqa-proxy-sweep-r2`; both wait for explicit approval.
The old `plan_proxy.json` and `release_proxy.json` stay as the record of the failed attempt.

Resubmitted with your approval at 23:05 UTC as `/calvinxu/tpp10-rcqa-proxy-sweep-r2` (log `submission_proxy_r2.log`, bundle 21.2 MB); monitor restarted against `plan_proxy_r2.json`, Fieldbook and the hourly cron repointed.

## Recovery, 19 September 23:07 UTC: the first attempt's pool parts block the repinned recipe

`-r2` failed after 3 minutes with `ValueError: Completed part has a different recipe:
.../wiki_to_rcqa/raw/2026.09.19/train/parts/000`. The pool step writes the four groups as parts
under one versioned path and identifies each completed part by the sha256 of the whole recipe
(all 1400 sources), so the two parts the first attempt completed (000, 001, whose own sources
are unchanged) and the partial part 002 are rejected by the frozen `write_part` guard once any
source changes. Nothing else was reached. Resolution options: delete the three stale parts and
resubmit the same r2 plan and release (`submit_proxy_r3.sh`, job `-r3`), or bump `VERSION`,
which also moves and rebuilds the valid held-out cache.

Resolved by option 1 at 23:10 UTC (Calvin: "do what you deem best practice"): the 16 objects under `raw/2026.09.19/train/parts/000-002` (160 MB) were deleted and the same r2 plan and release resubmitted as `/calvinxu/tpp10-rcqa-proxy-sweep-r3` (log `submission_proxy_r3.log`); monitor, Fieldbook and cron repointed.

## Proxy results (collected 17:25 PDT, 19 September; parent -r3 succeeded)

On the five QA-format evaluations the minimum sits at p = 20 percent, 3.2 epochs (ARC-Challenge
at 0.8 with a 0.2 percent margin), against 1.6 epochs for FLAN on the same evaluations; the rcQA
basins are shallower (3 to 6.5 percent benefit against FLAN's 8 to 12) and flatter around the
minimum. The in-domain held-out rcQA stream and Uncheatable Wikipedia English both minimize at
7.9 epochs with proper U shapes. The swarm's expectation of an optimum at 11 epochs or more on
the QA sets did not hold. Excess over the grid minimum:

| Evaluation | optimum (ep.) | depth (p=0) | 0.8 ep. | 1.6 ep. | 3.2 ep. | 4.8 ep. | 7.9 ep. | 11.1 ep. | 15.8 ep. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SciQ | 3.2 | 6.5% | +2.4% | +3.0% | 0 | +1.5% | +3.1% | +6.2% | +28.5% |
| ARC-Easy | 3.2 | 4.0% | +1.0% | +1.5% | 0 | +1.1% | +2.1% | +4.6% | +28.9% |
| ARC-Challenge | 0.8 | 2.9% | 0 | +0.9% | +0.2% | +0.8% | +1.8% | +4.8% | +28.9% |
| OpenBookQA | 3.2 | 3.7% | +0.6% | +0.6% | 0 | +0.4% | +1.1% | +3.9% | +34.2% |
| QASC | 3.2 | 5.8% | +1.8% | +3.4% | 0 | +1.5% | +3.0% | +6.8% | +33.0% |
| rcQA held-out stream | 7.9 | 17.1% | +7.1% | +5.0% | +2.5% | +1.2% | 0 | +1.0% | +9.8% |
| Uncheatable Wikipedia EN | 7.9 | 6.7% | +4.4% | +3.2% | +1.6% | +0.8% | 0 | +1.0% | +23.7% |

Files: `results/measurements.csv`, `results/analysis.json`, `results/rcqa_vs_flan_proxy_on_target.png`
(rcQA and FLAN side by side on the shared evaluations). The retrained p=0 point reproduces the
FLAN and arXiv sweeps' p=0 (same lock).
