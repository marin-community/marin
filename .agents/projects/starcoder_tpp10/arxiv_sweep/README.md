# arXiv-papers TPP-10 domain sweep (started 18 September 2026)

Purpose: a third domain for Figure 4 whose on-target evaluation should have a sharp, smooth
U-shaped optimum in epochs, to show that no single epoch cap transfers across domains. The
Wikipedia curves are flat on their left side on every logged evaluation; FineMath on GSM8K and
StarCoder on code are U-shaped. arXiv papers are scarce in the Nemotron-CC web blend and have
three in-distribution held-out evaluations already in the harness.

Everything except the domain text is identical to the Wikipedia and FineMath sweeps
(`experiments/domain_phase_mix/launch_tpp10_domain_sweeps.py`): the frozen TPP-10 design
(16.6M-parameter proxy at 165.9M tokens, 301M-parameter target at 3.01B tokens, TinyLlama
tokenizer, MuonH, cosine decay), the 190,316,544-token parent pool with the frozen parent
permutation, one nested 10,485,760-token matched subset (seed 20260912), the frozen allocator
and Nemotron-CC blend, trainer seed 20260910, and the grid p = 0, 5, 10, 20, 30, 50, 70, 100.
The pool size is fixed on purpose: the comparison isolates the data distribution's effect on
the optimal epoch count from the pool's size.

## Domain source

`common-pile/arxiv_papers_filtered@033cf7f` in `gs://marin-us-central1/raw/common_pile/`
(8 gzip JSON shards, 6.0 GB, markdown-converted full papers, public-domain licenses). The two
shards were chosen by the same seeded rule as FineMath's parquet files (seed 20260916, ascending
sha256 of "seed:uri"): `arxiv-papers-0006.json.gz` and `arxiv-papers-0004.json.gz`, pinned by
generation, size and CRC32C in `experiments/domain_phase_mix/tpp10_arxiv_sweep_assets/sources.json`.

## Evaluations at every point (final step, full sets)

- Seven Uncheatable components (existing TPP-10 caches of 11 September).
- Paloma programming languages (the frozen primary metric; p=0 reproduces the web-only run).
- Paloma `m2d2_s2orc_unsplit`, complete validation split (167 files, 22.8 MB gzip), tokenized
  with the frozen preparer into `paloma/m2d2_s2orc_unsplit-tpp10/2026.09.18`.

On-target candidates: Uncheatable arXiv CS, Uncheatable arXiv physics, Paloma S2ORC. The
p=0 point is retrained (not reused) so that S2ORC is scored on all eight points; its Paloma
code BPB should reproduce the archived web-only run within float noise (1.7713 BPB).

## Stage 1: proxy sweep (submitted 19:05:57 UTC, 18 September)

- Iris parent `/calvinxu/tpp10-arxiv-proxy-sweep`, us-central1-a, 2 CPU / 8 GB, non-preemptible,
  interactive, 48 h timeout, no automatic retries; children inherit the interactive band and
  request v5p-8 (8 CPU / 64 GB). Bundle 21.0 MB.
- Plan `plan_proxy.json` (sha256 d7f79004…), release `release_proxy.json`, exact command
  `submit_proxy.sh`, log `submission_proxy.log`. Durable copies under
  `gs://marin-us-central1/experiments/tpp10_arxiv_sweep/<plan_sha256>/`.
- Order: raw pool, parent, matched subset and S2ORC cache on the coordinator's CPU steps
  (hours; Wikipedia's took about eight with recoveries), then the p=100 execution check, then
  the remaining seven points with max_concurrent 7. Training compute 1.99e17 FLOPs, one
  eighth of the Wikipedia/FineMath survey's proxy half.
- Monitor: `monitor_proxy.sh` (nohup) writes `monitor_proxy.log` every 15 minutes and collects
  into `results/` when the parent is terminal. Manual collection:

```bash
MARIN_PREFIX=gs://marin-us-central1 uv run --offline --no-sync python -m experiments.domain_phase_mix.launch_tpp10_arxiv_sweep --plan .agents/projects/starcoder_tpp10/arxiv_sweep/plan_proxy.json --collect .agents/projects/starcoder_tpp10/arxiv_sweep/results/measurements.csv
```

Promising means: on at least one of the three on-target evaluations the proxy minimum is
interior with the neighbours at least a few percent worse on both sides. The Wikipedia proxy on
Wikipedia English (minimum 7.9 epochs, neighbours +0.7% and +1.0%) is the bar to beat.

## Stage 2: target sweep (not authorized yet)

`--stage target` builds the same eight points for the 301M target (6.7e18 FLOPs each, 5.3e19
total). Decide then whether to retrain p=0 or reuse the archived web-only target with a post-hoc
S2ORC evaluation.

## Code

- `experiments/domain_phase_mix/prepare_tpp10_arxiv_sweep.py`: data steps and the S2ORC cache.
- `experiments/domain_phase_mix/launch_tpp10_arxiv_sweep.py`: plan, submission, collection.
- `experiments/domain_phase_mix/analyze_tpp10_arxiv_sweep.py`: per-evaluation minima and excess.
- `tests/test_tpp10_arxiv_sweep.py`: stream equality with the frozen launcher, S2ORC tag, metrics.
- `experiments/domain_phase_mix/launch_starcoder_tpp10.py`: `runtime_versions()` now resolves
  the lock's two jax entries (0.11.0 default, 0.11.1 TPU extra, since the 15 September lock
  update) to the Levanter TPU-extra version; the training path is unchanged. Without this every
  TPP-10 plan build failed with "Ambiguous lock version: jax".

## Recovery, 19 September 04:19 UTC: 4 GiB preparation workers

The first coordinator queued its two preparation children for 1 h 50 min: the us-central1-a
on-demand CPU pool (six 16 GB workers) was full of other users' coordinators, and each
preparation task asked for 2 CPU / 8 GiB. With the user's authorization the tree
`/calvinxu/tpp10-arxiv-proxy-sweep` was canceled (all three jobs killed, no preparation had
started, only RUNNING status files and 90-second leases were left behind) and resubmitted as
`/calvinxu/tpp10-arxiv-proxy-sweep-cpu4g` through `resume_tpp10_arxiv_sweep.py`, which lowers
only the preparation CPU tasks (`prepare_raw`, `prepare_parent`, `prepare_subset`) to
2 CPU / 4 GiB and refuses any other CPU task. Plan, release, recipes, identities and every TPU
request are unchanged; the coordinator itself keeps 2 CPU / 8 GB. Exact command
`submit_proxy_cpu4g.sh`, log `submission_proxy_cpu4g.log`. The 190M-token tokenization has not
been run below 8 GiB before; an out-of-memory failure would surface as a failed preparation child.

## Recovery 2, 19 September 19:39 UTC: bundle must keep .html

Once the central1-a CPU pool freed up (the 24 leaked CI-smoke Zephyr coordinators were removed
around 19:00 UTC), the arXiv preparation task's cache-consolidation probe started and crashed:
`FileNotFoundError: /app/lib/zephyr/src/zephyr/dashboard/index.html`. Levanter's
`consolidate_shard_cache_ledgers` launches a nested Zephyr coordinator (`levanter-cache-probe`),
and the coordinator serves that dashboard file at startup. Both 18-19 September submissions used
`--exclude '\.(png|html|pkl|npz|parquet|pdf)$'`, copied from the v6e ladder launches, which
never start Zephyr coordinators. Both trees were cancelled (finished tokenized parts are reused
through their receipts) and resubmitted with `html` dropped from that exclude:
`/calvinxu/tpp10-arxiv-proxy-sweep-r3` (log `submission_proxy_r3.log`) and
`/calvinxu/tpp10-instruction-proxy-sweep-r2`. Plans, releases and resources are unchanged.

## Recovery 3, 19 September 20:05 UTC: the branch's Iris bundler excluded .html

Dropping `html` from the submission's own exclude was not enough: this branch's
`lib/iris/src/iris/cluster/client/bundle.py` carries extra default excludes added in April
(commit b33dc8fa9a: png, jpg, pkl, html, zip, csv.gz, joblib) that origin/main does not have, and
`--bundle-include` cannot re-add a file the default exclude rejects. The Zephyr dashboard (merged
15 September) reads `lib/zephyr/src/zephyr/dashboard/index.html` at every coordinator start, so
every nested cache-probe coordinator died. Fix: removed `\.html$` from the branch's DEFAULT_EXCLUDE
(adds at most 324 KB of tracked html; bundle 22.2 MB). Resubmitted as
`/calvinxu/tpp10-arxiv-proxy-sweep-r4` (log `submission_proxy_r4.log`) and
`/calvinxu/tpp10-instruction-proxy-sweep-r3`; plans, releases and resources unchanged.

## Proxy results (collected 20:40 UTC, 19 September; parent -r4 succeeded)

All three on-target evaluations have an interior minimum at p = 50 percent, 7.94 epochs. Excess
over the minimum on the grid 0 / 5 / 10 / 20 / 30 / 50 / 70 / 100 percent (0 to 15.8 epochs):

| Evaluation | benefit depth (p=0) | 4.8 ep. | 7.9 ep. | 11.1 ep. | 15.8 ep. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncheatable arXiv physics | 13.6% | +1.6% | 0 | +0.1% | +16.2% |
| Uncheatable arXiv CS | 9.2% | +0.5% | 0 | +0.7% | +19.9% |
| Paloma S2ORC | 9.9% | +1.1% | 0 | +0.9% | +24.2% |

Off-target: Wikipedia English, BBC and AO3 minimize at 0 to 0.8 epochs; GitHub, Paloma code and the
Uncheatable mean at 4.8 epochs. The retrained p=0 run scores 1.7792 on Paloma code against the
archived web-only run's 1.7713 (same trainer seed); the two archived trainer seeds differ by 0.023,
so this is within run-to-run noise. Plot: `results/arxiv_proxy_on_target.png`.
