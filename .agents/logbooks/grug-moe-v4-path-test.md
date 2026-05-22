# Grug-MoE v4 Path Test: Research Logbook

## Scope
- Goal: evaluate the one-dimensional path from the proportional Grug-MoE mixture to the v4 mixture.
- Path: `w(t) = (1 - t) * w_proportional + t * w_v4`, with `t in {0.25, 0.50, 0.75}`.
- Existing endpoints: `t=0` is the completed `grug_moe_mix` proportional track; `t=1` is the completed `grug_moe_mix_v4` track.
- Scales: d512/2.19e17, d768/1.70e18, d1024/9.00e18, d1280/2.83e19, d1536/9.00e19.
- Primary readout: Grug validation/perplexity during training, followed by the Grug logprob eval suite used by `grug_moe_mix_dashboard_20260517`.

## Baseline
- Existing dashboard: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/dashboard.html`.
- Endpoint data source: `gs://marin-us-east5/grug/{grug_moe_mix,grug_moe_mix_v4}_d*-*`.
- Existing eval source: `gs://marin-us-east5/evaluation/grug_logprob`.
- Known t=1 coverage issue: `grug_moe_mix_v4_d1536-9.00e+19` has executor-successful but metric-empty `logprob_gsm8k_5shot` and `logprob_humaneval_10shot` results. These need a targeted eval retry before treating t=1 as full-metric complete.

## Experiment Log

### 2026-05-20 17:35 - Training Dry Run
- Hypothesis: intermediate path points will reveal whether v4’s gains/tradeoffs are monotone and whether a trust-region interpolation beats both endpoints on the aggregate.
- Command:
  `uv run python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5 --dry_run True`
- Config:
  15 training specs, `max_concurrent=8`, `v5p-8`, `us-east5-a`, `TARGET_BUDGET=6325183647689`, `TARGET_STEPS=2**14`.
- Result:
  Dry run inspected 15 Grug path training steps and emitted expected output roots:
  `gs://marin-us-east5/grug/grug_moe_mix_v4_path_{t025,t050,t075}_d*-*`.
- Validation:
  Local artifacts under `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_v4_path_test_20260520/` have 15 manifest rows and 234 weight rows. Each candidate phase sums to 1.0.
- Interpretation:
  The launcher is ready for live parent submission. The data config uses the current canonical top-level domain runtime-cache builder with phase weights recovered from the existing GCS endpoints.
- Next action:
  Submit the 15-run training parent, then run the same Grug logprob eval suite on the 15 new checkpoints plus the two malformed t=1 v4 d1536 eval cells.

### 2026-05-20 17:36 - Training Parent Submitted
- Command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-grug-moe-v4-path-train-20260520-1736 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 8GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5`
- Job:
  `/calvinxu/dm-grug-moe-v4-path-train-20260520-1736`
- Expected children:
  15 Grug training steps, with executor max concurrency 8. Each child dispatches a `v5p-8` training worker in `us-east5-a`.
- Initial status:
  Parent accepted by Iris; first summary showed the parent task in `building`.
- Next action:
  Recheck parent logs/summary until the 15 child training jobs have either completed or surfaced actionable failures. After all checkpoints land, launch the Grug logprob suite for the path rows and the two malformed v4 d1536 endpoint eval cells.

### 2026-05-20 17:40 - Parent Relaunch After GCS Client Fix
- Failure:
  `/calvinxu/dm-grug-moe-v4-path-train-20260520-1736` failed during parent startup before dispatching children because the Iris CPU parent image does not provide `gcloud`.
- Fix:
  Replaced launcher GCS reads with `fsspec.open` and hardcoded the two endpoint executor roots needed for proportional/v4 d512 endpoint weights:
  `gs://marin-us-east5/grug/grug_moe_mix_d512-2.19e+17-e6a48f` and
  `gs://marin-us-east5/grug/grug_moe_mix_v4_d512-2.19e+17-6aa7c8`.
- Validation:
  Re-ran dry-run successfully with:
  `uv run python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5 --dry_run True`.
- Relaunch command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-grug-moe-v4-path-train-20260520-1740 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 8GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5`
- Job:
  `/calvinxu/dm-grug-moe-v4-path-train-20260520-1740`
- Initial status:
  Parent accepted by Iris; first summary showed the parent `pending`.

### 2026-05-20 17:54 - Claude Code Launcher Review
- Command:
  `env -u ANTHROPIC_API_KEY claude --model claude-opus-4-7 --effort max --permission-mode dontAsk ...`
- Auth preflight:
  Local Claude OAuth account was `plambdafour@proton.me`, billing type `stripe_subscription`, and `ANTHROPIC_API_KEY` was removed from the Claude environment.
- Verdict:
  No hard blockers. Claude flagged one main comparability caveat: this path launcher uses the current canonical `TokenizedMixtureGroup` / `lm_varying_mixture_data_config` wiring, while the original v4/proportional endpoint branch used a `ConcatDatasetComponent`-style construction. These should be statistically close for size-proportional internal partition sampling, but they are not bit-identical data loaders.
- Other review notes:
  The path launcher also uses `shuffle=True`, whereas the endpoint path likely used the default `BlockShuffleConfig`, and it does not explicitly set `auto_build_caches=False`. Both are non-blocking for final metrics but should be documented when interpreting endpoint-vs-path loss curves.
- Decision:
  Do not stop the already-dispatched training parent. Treat this as a documented confound. If the path results look inconsistent, run a small t=0 calibration through the same path launcher wiring before drawing strong endpoint-relative conclusions.
- Eval follow-up requirements:
  The full-metrics eval launcher should set Grug eval `capacity_factor=8.0` explicitly and must not treat the two malformed v4 d1536 GSM8K/HumanEval `results.json` files as complete. Force a new eval hash or use a fresh output prefix for those two endpoint retry cells.

### 2026-05-20 18:03 - Training Relaunch After Missing `asyncio` Fix
- Failure:
  The `17:40` parent dispatched children, but several failed at dataset construction with:
  `NameError: name 'asyncio' is not defined` from `lib/levanter/src/levanter/data/text/datasets.py` in `LazyAsyncDataset.__init__`.
- Fix:
  Added the missing `import asyncio` to `lib/levanter/src/levanter/data/text/datasets.py`.
- Validation:
  `uv run python -m py_compile lib/levanter/src/levanter/data/text/datasets.py experiments/grug/moe/model.py experiments/grug/moe/launch_v4_path_test.py`

  `uv run python - <<'PY' ... LazyAsyncDataset(lambda: None, finite_length=1) ... PY`

  `uv run python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5 --dry_run True`

  The retry dry-run emits 15 fresh `grug_moe_mix_v4_path_r1_*` steps. Local artifacts still have 15 manifest rows and 234 candidate-weight rows; every candidate phase sums to 1.0.
- Recovery:
  Stopped `/calvinxu/dm-grug-moe-v4-path-train-20260520-1740` to avoid wasting TPU time on the known-bad image. Submitted a fresh parent:
  `/calvinxu/dm-grug-moe-v4-path-train-r1-20260520-1803`.
- Current status:
  Initial summary shows the retry parent running/assigned with no failures yet. First status sweep shows the first 8 retry executor steps as `RUNNING` and the remaining 7 not yet dispatched, matching `max_concurrent=8`.
