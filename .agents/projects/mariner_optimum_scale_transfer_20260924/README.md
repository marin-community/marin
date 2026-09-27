# Four Uncheatable optima at the Llama proxy settings (2026-09-24)

Calvin asked (2026-09-24 ~18:50 PDT) to train MARINER's Uncheatable optimum at the two swarm settings where it had
not been trained, so the scale-transfer figure (paper Figure 28, `r3_scale_transfer_all_pairs`) can show a deployed
marker in every pair, ideally far into the bottom-left of each panel; after the first submission failed he asked
(19:20 PDT) to also train the Olmix, tuned-RegMix and released-RegMix Uncheatable optima "for good measure". The
39-bucket pools exist only in us-east5 (and Europe), so the runs are v5p-8 jobs in us-east5-a; the Olmix 1e21 runs
use the v6e-64 pool in us-east5-b and share only the CPU parent pool. Calvin approved that at 19:00 PDT.

## Launch

- Launcher: `experiments/domain_phase_mix/launch_mariner_optimum_scale_transfer.py` (new): the four trained
  proposals of the paper's Table 5 in their runtime (1/2048-grid) form, each read from its candidate table under
  `exploratory/two_phase_many/reference_outputs/` (added to the bundle with `--working-dir-include`):
  MARINER `lwspu_u_snc_cap06` (run name `mariner_u`, TV 0.515 from proportional), matched Olmix
  `olmixq_u_kl0p05_cap04` (`olmix_u`, 0.284), tuned RegMix `cmp_u_lgbm_cap08` (`regmix_tun_u`, 0.387) and released
  RegMix `rgref_u_endpoint_cap64` (`regmix_rel_u`, 0.397). Each is trained through the qsplit240 replay experiment
  of the proportional-perturbation scale-transfer launcher at `REGMIX_60M_1P2B` (Llama 160M/1.2B, 4,577 steps) and
  `REGMIX_300M_6B` (Llama 200M/6B, 22,888 steps), batch 128 x 2048, simulated epoching to 6.33T, run ids / data
  seeds 795000-795003, lm-eval harness skipped (Uncheatable is an in-training validation set). Dry-run manifests:
  `reference_outputs/mariner_optimum_scale_transfer_20260924/`.
- First submission `/calvinxu/dm-mariner-optimum-scale-transfer-20260924` (19:16 PDT, MARINER only, double-wrap
  form: the top-level task ran a nested `iris job run`) failed after 47 s: the nested submission was refused by the
  IAP edge with `connectrpc.errors.ConnectError: Forbidden` (HTTP 403), although the identical form launched the
  UniMax sweeps on 2026-09-22. Fieldbook job marked failed.
- Retries, all in the direct form (the executor is the top-level job and launches the eight v5p-8 children through
  the in-task client, `--max-concurrent 8`; east5 guard passed on the equivalent `iris job run` command; bundle
  18.4 MB; Fieldbook experiment `exp_01m3b5jaqsb0eq3wtz0armaq79`; watch: `watch.sh`, detached, 10-minute ticks
  -> `watch.log`):
  - retry1 (19:36 PDT) failed at import: `exploratory/two_phase_many/two_phase_many.csv` is read at import time by
    the qsplit240 launcher chain and the exploratory tree is excluded; found in one pass by staging the bundle
    locally with the wrapper's own filter and running the dry-run inside it (scratch `staged_dry_run_20260924.py`).
  - retry2 (19:43 PDT) raised under `MARIN_EXECUTOR_STRICT=1`: the launcher built its steps outside
    `executor_context()`; fixed like `launch_delphi_baseline_mixtures.py` (strict dry run passes).
  - retry3 (19:53 PDT) ran the executor, and both `cache_eval_datasets` steps failed: the branch-only
    `marin.evaluation.eval_dataset_cache._cache_eval_datasets` returned the cache path as a str, and the executor's
    `ArtifactRecord.result` now requires a mapping; it returns `{"gcs_path": ...}` now. Every other step function in
    the pipeline returns None.
  - retry4 `/calvinxu/dm-mopt-scale-transfer-20260924-retry4`, submitted 20:18 PDT: both cache steps succeeded at
    20:20 PDT and the executor launched all eight training children through the in-task client (three 60M runs
    running at once, the rest pending v5p-8 slots), so the direct form works where the nested submission did not.
  - retry5 `/calvinxu/dm-mopt-scale-transfer-20260924-retry5`, submitted 2026-09-25 16:52 PDT: retry4 trained 7 of 8 runs,
    but the tuned-RegMix 200M/6B run stalled near step 18k through 17 preemptions (attempts of 2-20 min with ~4 min of
    compilation each and 10-minute temporary checkpoints, so short attempts banked nothing; some attempts were stamped
    batch while the user was over the Iris budget and were evicted by interactive jobs, including this project's own
    per-bucket-threshold validation). retry4 was cancelled and retry5 relaunches that run with `--checkpoint-minutes 3`;
    output paths are unchanged, so the executor skipped the seven finished runs and resumed the eighth from its
    temporary checkpoint.
- Outputs: `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/mopt_<scale>/<run>-<hash>/` with
  `<scale>` in `60m_1p2b`, `300m_6b` and `<run>` in `mariner_u`, `olmix_u`, `regmix_tun_u`, `regmix_rel_u`
  (Uncheatable in `checkpoints/eval_metrics.jsonl`); the executor's `collect_results` and `fit_dataset_export`
  steps write RESULTS_CSV / FIT_DATASET_CSV per scale under the `mopt_<scale>` prefixes.
- Pitfalls met: the perturbation launcher's `_configure_training_step` passes `job_name`, which
  `TrainLmOnPodConfig` no longer accepts (the new launcher sets env vars only); W&B truncation of long run names
  dropped the scale token and gave both scales the same checkpoint path (fixed with the short `mopt` prefix and
  scale-tagged step names; no name needs truncation now); excluding `infra/` wholesale from the bundle breaks the
  install plan (`infra/deploy` is a workspace member), so the exclude list is the UniMax cap sweep's; a nested
  `iris job run` inside a task is now refused (see above), so executors are submitted directly.

## When it lands

Collect the byte-weighted Uncheatable macro BPB of each run (as the ladder collect procedure does) and compare with
the swarm's best runs at each setting; then extend `plot_scale_transfer_results_figure_20260905.py` to draw a
deployed marker per method and panel (x = source-setting BPB, y = target-setting BPB of the same mixture; Qwen3
3e18 values MARINER 0.982, Olmix 1.002, tuned RegMix 1.0003, released RegMix 1.0271) with labels, replace the "no
deployed markers" caption note, rebuild `r3_scale_transfer_all_pairs`, update Figure 28's caption and the outline,
and push.
