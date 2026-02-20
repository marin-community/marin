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

### TPU-specific gotchas that frequently cause 10-100× slowdowns

These are easy to miss and can dominate performance even when the math is optimal:

1) **Vector register layout: the last two axes are special.**
   TPU vector registers are organized as `(sublanes=8, lanes=128)`.
   Any elementwise operation is implicitly padded to these tile sizes.

   Practical implication:
   - A tensor shaped like `(..., 128, 1)` is extremely wasteful (the trailing `1` sits on the *lane* axis).
   - Prefer `(..., 1, 128)` when you need a broadcastable extra dimension.

   If you see shapes like `(Ct, 1)` or `(..., Ct, 1)` in Pallas kernels, assume it is a performance bug
   unless proven otherwise.

2) **Avoid explicit transposes of the last two axes.**
   Transposes/reshapes that touch the last two axes tend to lower to expensive XLU ops.
   For matmuls, use `lax.dot_general` dimension numbers to *fuse* the transpose.

3) **Default matmul precision & dtypes matter.**
   On TPU, the fast path is typically BF16 inputs with FP32 accumulation.
   Casting everything to FP32 inside the kernel can add conversions without actually buying precision
   unless FP32 matmul precision is explicitly requested.

4) **Use pipelining instead of Python loops for long sequential dimensions.**
   When you want a single kernel to “loop over” many blocks without unrolling, use
   `pltpu.emit_pipeline` with dynamic `pl.ds(start, size)` slicing.

## Optimization Policy (Aggressive Phase)
- Every iteration must target a meaningful reduction in dominant hotspot cost, not only parameter retuning.
- Prefer changes that reduce Pallas custom-call launch count, increase work per call, improve tiling/layout, or remove serial dependencies.
- Equivalent mathematical reformulations are encouraged when they preserve model semantics and remove expensive operations (including explicit triangular inversion).
- Do not run standalone iterations that only tweak scalar constants (`unroll`, `chunk`, `segment`, `batch`) unless paired with a structural kernel/dataflow change.
- If an iteration delivers <3% MFU gain and hotspots are unchanged, the next iteration must escalate to a more radical design.
- Use FlashLinearAttention and Pallas TPU docs as design references before implementing.
- Failed or regressive attempts must not leave uncommitted working tree changes that block the next loop iteration.

## The “Macro Move” Menu (pick one per iteration)

Most local minima come from repeatedly tweaking `segment_size`, `unroll`, and small algebra.
The loop should instead cycle through large, architectural moves.

Pick **exactly one** of the following per Codex iteration and push it to a fully tested, benchmarked state:

### A) Fix vector-layout pathologies
Target: eliminate any `(..., 128, 1)` / `(..., Ct, 1)` patterns in Pallas kernels.

Typical changes:
- reshape `g_cum` / `beta` blocks as `(..., 1, Ct)` instead of `(..., Ct, 1)`
- keep `g`/`beta` as rank-4 inputs (`(..., Ct)`) instead of rank-5 with trailing singleton
- ensure gradient outputs (`dg`, `db`) use the same non-pathological layout

### B) Replace `jnp.matmul(..., x.T)` with `lax.dot_general`
Target: fuse transposes and control dtype/precision explicitly.

Typical changes:
- introduce a single helper `mxu_dot(a, b, transpose_b=True, preferred_dtype=f32, precision=...)`
- systematically migrate all matmuls in both fwd and bwd kernels

### C) Switch the kernel math to BF16 inputs + FP32 accumulation
Target: reduce conversion overhead and increase MXU throughput.

Typical changes:
- keep `q/k/v` as bf16 in VMEM
- use `preferred_element_type=jnp.float32` in dot_general
- avoid eager `.astype(jnp.float32)` unless numerically required

### D) Use `pltpu.emit_pipeline` to fuse across chunk/segment loops
Target: remove the segmentation hierarchy (or make segments much larger) without VMEM blowups,
by loading one chunk at a time from HBM.

Typical changes:
- outer `pallas_call` grid over `NH`
- inside kernel: `emit_pipeline` with a sequential stage axis over chunks
- use `pl.ds(chunk_idx * Ct, Ct)` dynamic slicing in `BlockSpec` index maps
- keep `S_prev` in scratch across stages; write outputs per stage

### E) Tile the state/output along V
Target: shrink per-program state from `K×V` to `K×Vb` (e.g. `128×32`) so more programs can
co-reside in VMEM and reduce spill risk.

Typical changes:
- add a `vblock` grid axis
- make state scratch `K×Vb`
- ensure any `K×K` intermediates are shared or recomputed cheaply

### F) Match FlashLinearAttention’s kernel decomposition
Target: split the fused kernel into 2-4 kernels (A-build, solve/invert, recurrence, output)
so each kernel has a simple performance profile and lower register pressure.

## Measurement: avoid “trace-only” optimization

XProf traces are essential for hotspot attribution, but the loop also needs a **stable numeric score**
that can be used for automated selection.

Recommended scoring stack:
1) **Microbench**: `chunk_gated_delta_rule` forward+backward for one realistic shape (e.g. Qwen3 Next).
2) **Tiny profile**: the existing `dev-tpu-profile` / `ray-profile` run for end-to-end validation.

If you add a new microbenchmark script/command, keep it:
- deterministic (fixed PRNG),
- short (<30s wall),
- printing a single parseable line like `GDN_BENCH p50_ms=... mean_ms=...`.

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
  --resilient \
  --directive-preset tpu-layout-and-dtypes \
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
- Use `--resilient` for unattended loops to keep running through transient failures (network, connectivity, allocation).
- `--directive`, `--directive-file`, and `--directive-preset` inject per-session guidance (for example triangular inversion focus) without editing prompt files.
- Preset directives are stored as markdown docs under `scripts/gdn/session_directives/` (`triangular-inversion` maps to `scripts/gdn/session_directives/triangular-inversion.md`).
  Useful presets:
  - `tpu-layout-and-dtypes`: avoid TPU register-layout cliffs (singleton last-axis, transpose fusion, BF16/F32 policy).
  - `emit-pipeline-fullseq`: design sketch for collapsing segmentation using `pltpu.emit_pipeline`.
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
