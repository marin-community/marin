# Recipe: Optimize Gated DeltaNet TPU Kernels

## Overview
Use this recipe when iterating on `lib/levanter/src/levanter/layers/gated_deltanet.py` for TPU performance.

This recipe standardizes the loop:
1. pick one optimization hypothesis,
2. run correctness on TPU,
3. run a small profiled training job,
4. inspect the trace,
5. commit one validated optimization.

The current baseline bottlenecks (from issue [#1884 comment 3714287157](https://github.com/marin-community/marin/issues/1884#issuecomment-3714287157), updated January 6, 2026):
- strict lower-triangular inversion is expensive on TPU; sequential dependencies hurt MXU occupancy,
- dynamic slicing inside Pallas TPU kernels is not available, forcing static indexing and segmented loop structures.

## Infra Added For This Loop
- `scripts/gdn/gdnctl.py`: one CLI for tests, profile submission, Ray wait/logs, HF trace downloads, and unattended Codex loops.
- `experiments/speedrun/hackable_transformer_gdn/tiny_profile.py`: short profiled training run using the same GDN model code path.
- `scripts/gdn/codex_iteration_prompt.md`: prompt template for unattended Codex hill-climbing.
- `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`: append-only experiment log.

## Prerequisites
- Ray auth/token configured (`make get_ray_auth_token` if needed).
- `HF_TOKEN` and `WANDB_API_KEY` available for profile artifact access.
- TPU capacity available in `us-central1` or `us-east5-a`.

## Standard Loop Commands

### 1) Correctness on TPU

Ray path:
```bash
uv run python scripts/gdn/gdnctl.py ray-test \
  --cluster us-central1 \
  --tpu auto \
  --tests both
```

Dev TPU path:
```bash
uv run python scripts/gdn/gdnctl.py dev-tpu-allocate \
  --cluster us-central1 \
  --tpu-name "$USER-gdn"

uv run python scripts/gdn/gdnctl.py dev-tpu-test \
  --cluster us-central1 \
  --tpu-name "$USER-gdn" \
  --tests both
```

### 2) Submit lightweight profile run

```bash
uv run python scripts/gdn/gdnctl.py ray-profile \
  --cluster us-central1 \
  --tpu v5p-8 \
  --size 130m \
  --num-steps 20 \
  --profile-start-step 2 \
  --profile-num-steps 6 \
  --batch-size 8 \
  --no-wait
```

If Ray queueing is unstable, run the same profile loop on an allocated dev TPU:
```bash
uv run python scripts/gdn/gdnctl.py dev-tpu-profile \
  --cluster us-central1 \
  --tpu-name "$USER-gdn" \
  --tpu v5p-8 \
  --size 130m \
  --num-steps 20 \
  --profile-start-step 2 \
  --profile-num-steps 6 \
  --batch-size 8 \
  --no-sync
```

### 3) Wait and inspect logs

```bash
uv run python scripts/gdn/gdnctl.py ray-wait \
  --cluster us-central1 \
  <job_id> \
  --show-logs \
  --tail 400
```

```bash
uv run python scripts/gdn/gdnctl.py ray-logs \
  --cluster us-central1 \
  <job_id> \
  --tail 400 \
  --grep "profiler|trace|ERROR|Traceback"
```

### 4) Download trace artifacts from HF

```bash
uv run python scripts/gdn/gdnctl.py hf-download-trace \
  --repo-id <org/repo> \
  --repo-type dataset \
  --path-prefix <run_or_trace_path> \
  --output-dir .profiles/hf
```

## Unattended Codex Loop

Run multiple autonomous iterations:

```bash
uv run python scripts/gdn/gdnctl.py codex-loop \
  --iterations 10 \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --prompt-file scripts/gdn/codex_iteration_prompt.md \
  --post-check "uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both"
```

Notes:
- Use `--codex-profile <profile-name>` if you have a Codex CLI profile for `gpt-5.3-codex` / high reasoning settings.
- By default, `codex-loop` stops if an iteration does not create a new commit.
- Use `--allow-dirty` or `--allow-no-commit` only when intentionally debugging the loop harness.

## Logging Expectations
After each meaningful iteration, append:
- hypothesis,
- exact command(s),
- test pass/fail,
- profile job id and trace path,
- key hotspot findings,
- next hypothesis,

to `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`.
