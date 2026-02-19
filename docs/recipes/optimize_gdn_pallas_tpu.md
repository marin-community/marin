# Recipe: Optimize Gated DeltaNet TPU Kernels

## Overview
Use this recipe when iterating on `lib/levanter/src/levanter/layers/gated_deltanet.py` for TPU performance.

This recipe standardizes the loop:
1. pick one high-impact optimization hypothesis,
2. run correctness on TPU,
3. run a small profiled training job,
4. inspect the trace,
5. commit one validated optimization.

Current phase goal:
- Push MFU toward ~50% with large kernel-level wins.
- Treat small tuning wins as secondary; prioritize architecture/kernel redesigns first.

The current baseline bottlenecks (from issue [#1884 comment 3714287157](https://github.com/marin-community/marin/issues/1884#issuecomment-3714287157), updated January 6, 2026):
- strict lower-triangular inversion is expensive on TPU; sequential dependencies hurt MXU occupancy,
- dynamic slicing inside Pallas TPU kernels is not available, forcing static indexing and segmented loop structures.

## Optimization Policy (Aggressive Phase)
- Every iteration must target a meaningful reduction in dominant hotspot cost, not only parameter retuning.
- Prefer changes that reduce Pallas custom-call launch count, increase work per call, improve tiling/layout, or remove serial dependencies.
- Equivalent mathematical reformulations are encouraged when they preserve model semantics and remove expensive operations (including explicit triangular inversion).
- Do not run standalone iterations that only tweak scalar constants (`unroll`, `chunk`, `segment`, `batch`) unless paired with a structural kernel/dataflow change.
- If an iteration delivers <3% MFU gain and hotspots are unchanged, the next iteration must escalate to a more radical design.
- Use FlashLinearAttention and Pallas TPU docs as design references before implementing.
- Failed or regressive attempts must not leave uncommitted working tree changes that block the next loop iteration.

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

To include XProf `xplane.pb` payloads only when needed:
```bash
uv run python scripts/gdn/gdnctl.py hf-download-trace \
  --repo-id <org/repo> \
  --path-prefix <run_or_trace_path> \
  --include-xplane
```

## Unattended Codex Loop

Run multiple autonomous iterations:

```bash
uv run python scripts/gdn/gdnctl.py codex-loop \
  --iterations 10 \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --directive-preset triangular-inversion \
  --dirty-policy stash \
  --no-commit-policy count-failure \
  --hold-dev-tpu \
  --dev-tpu-cluster us-central1 \
  --dev-tpu-fallback-cluster us-east5-a \
  --dev-tpu-name "$USER-gdn" \
  --dev-tpu-type v5p-8 \
  --dev-tpu-allocate-attempts 2 \
  --dev-tpu-allocate-retry-sleep 20 \
  --prompt-file scripts/gdn/codex_iteration_prompt.md \
  --post-check "uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name $USER-gdn --tests both" \
  --post-check "uv run python scripts/gdn/gdnctl.py lint-log"
```

Notes:
- Use `--codex-profile <profile-name>` if you have a Codex CLI profile for `gpt-5.3-codex` / high reasoning settings.
- By default, `codex-loop` stops if an iteration does not create a new commit.
- Use `--allow-dirty` or `--allow-no-commit` only when intentionally debugging the loop harness.
- The default iteration prompt (`scripts/gdn/codex_iteration_prompt.md`) is intentionally aggressive; keep it aligned with this policy.
- `--directive`, `--directive-file`, and `--directive-preset` inject per-session guidance (for example triangular inversion focus) without editing prompt files.
- Preset directives are stored as markdown docs under `scripts/gdn/session_directives/` (`triangular-inversion` maps to `scripts/gdn/session_directives/triangular-inversion.md`).
- Prefer `--dirty-policy stash --no-commit-policy count-failure` for unattended long runs so failed attempts do not permanently block progress.
- `--dirty-policy stash` restores the stashed tree automatically after each iteration.
- If stash restore conflicts with edits produced in the iteration, default `--stash-restore-policy warn-keep` keeps the stash and continues; use `--stash-restore-policy fail` for strict stop-on-conflict.
- Add `--hold-dev-tpu --dev-tpu-name <name>` to make `codex-loop` allocate/hold/release a dev TPU allocation for the entire loop session.
- In managed dev TPU mode, use `dev-tpu-test`/`dev-tpu-profile` (not Ray TPU test/profile commands) for loop validation/profiling.
- Keep managed-mode `--post-check` commands aligned with the held allocation `--cluster` and `--tpu-name`.

## Logging Expectations
After each meaningful iteration, append:
- hypothesis,
- exact command(s),
- test pass/fail,
- profile job id and trace path,
- key hotspot findings,
- next bold hypothesis,

to `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`.

Log hygiene:
- Do not leave `Commit: (pending)` in new entries. Use a concrete status such as a SHA, `this commit`, or `none (failed attempt)`.
