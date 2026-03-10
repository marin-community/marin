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
- Optimize training throughput first; prioritize `chunk_gated_delta_rule` hotspots over decode-only paths.

## Current diagnosis (March 2026)

Recent evidence changes the optimization target:

- Multiple structurally different variants reduced train-path forward/backward `shard_map/pallas_call`
  closed-call buckets by roughly `40-52%`.
- End-to-end training still regressed because a device-side `while` bucket around `31.5-31.7 ms`
  appeared or became dominant.
- That means the limiting factor is now the lowered train-path control structure around the kernels,
  not only the math inside the kernels.

What to do differently now:

- Treat `closed-call` reductions as **necessary but not sufficient**.
- Evaluate every candidate in terms of:
  - `Kernel budget = forward_closed_call_ms + backward_closed_call_ms`
  - `Control budget = while_ms + conditional_ms`
  - `Train-path budget = kernel_budget + control_budget`
- A candidate is promising only if train-path budget improves or if there is a strong end-to-end MFU win
  that clearly dominates any control-flow regression.
- Do not spend primary iteration budget on more kernel-local H/J/E/I/G-style work unless it is nested
  inside a new outer train-path structure or explicitly removes the `while`/`conditional` bottleneck.
- After Iterations 64-66, also treat fused cross-entropy backend selection as a first-class bottleneck split.
- If residual `while` is still CE/XLA-attributed, do not keep spending mainline budget on standalone GDN-local work.
- Track these derived metrics for every profiled run:
  - `step_duration_ms = throughput/duration * 1000`
  - `remainder_budget_ms = step_duration_ms - train_path_budget_ms`
- If train-path budget improves but step duration does not, classify the candidate as `off-critical-path` / `overlap-loss`.

## CE-first pivot (March 2026)

Latest evidence:

- Iteration 64 produced a real GDN-side structural win:
  - forward closed-call `22.228 -> 20.661 ms`
  - backward closed-call `15.621 -> 13.130 ms`
  - kernel budget `37.849 -> 33.791 ms`
  - train-path budget `69.475 -> 65.401 ms`
- Iteration 65 then produced only `+0.26%` MFU with residual `while` staying flat around `31.6 ms`.
- Iteration 66 changed the outer train shell and still regressed slightly while the same `while` remained.
- Recent validated runs explicitly logged:
  - `Fused cross-entropy selected implementation: xla`

Interpretation:

- the remaining wall is no longer primarily inside the GDN train shell,
- the residual `while` is likely at least partly a CE/XLA backend-selection issue,
- the next budget should go first to CE/backend A/B and Macro O/M style control-arm work.

## Post-CE regime (March 2026, after Iterations 67-69)

The regime changed again once CE moved to TPU Pallas by default:

- Iteration 67 forced TPU fused CE to `pallas_tpu` and cut CE-attributed `while` from about `31.6 ms`
  to about `10.1 ms`, producing the first large post-review end-to-end win.
- Iteration 68 then reduced GDN forward closed-call and total tracked train-path budget materially while
  MFU still fell slightly.
- Iteration 69 confirmed that a full-XLA chunk control arm is diagnostic but clearly regressive as a
  deployable path.

Interpretation:

- CE implementation choice was the last giant false wall, and removing it was first-order.
- The next CE-specific target is the residual CE backward/custom-VJP shell.
- GDN remains expensive, but `GDN budget down` is no longer a reliable proxy for `step faster`.
- The loop must now track step remainder explicitly instead of inferring everything from GDN-labeled buckets.

Current mainline priority order:
1. CE backward mode A/B with `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu` fixed.
2. Remainder-budget instrumentation and promotion policy based on `step_duration_ms`.
3. Macro O as a diagnostic control arm only.
4. Macro M only after CE backward and remainder-budget work.

## Post-upper-bound regime (March 2026, after the attention-only control)

The practical ceiling is now measured on the same benchmark family:

- Hybrid validated regime: about `6.09 MFU`, `~166.3 ms` step time.
- Attention-only control (`gdn_layers_per_block = 0`): about `21.09 MFU`, `~57.86 ms` step time.

Interpretation:

- The hybrid-vs-attention gap is now the dominant fact.
- The gap is too large to treat as a few missing milliseconds inside the current GDN train shell.
- More than half of the gap is outside the currently tracked GDN closed-call + CE while budget.
- Therefore, same-boundary GDN Pallas train-path hillclimbing is no longer the mainline strategy.

Mainline priorities in this regime:
1. Full-step remainder attribution:
   - `upper_bound_gap_ms = hybrid_step_ms - attn_only_step_ms`
   - `gap_explained_by_train_path = train_path_budget_ms / upper_bound_gap_ms`
   - `remainder_topk`
2. Model-boundary sweep:
   - measure throughput penalty as `gdn_layers_per_block` varies with `gdn_block_size` fixed.
3. Bounded CE side-arms only if they improve end-to-end step time materially.
4. Treat Macro M/N/O/R style same-boundary GDN changes as research/diagnostic branches, not the mainline.

Required metrics from now on:
- `gdn_layer_fraction`
- `step_duration_ms`
- `train_path_budget_ms`
- `remainder_budget_ms`
- `upper_bound_gap_ms`
- `gap_explained_by_train_path`
- `remainder_topk`

Promotion policy in this regime:
- Reject candidates whose `step_duration_ms` does not improve, unless they are explicitly diagnostic.
- Reject candidates whose `remainder_budget_ms` grows, unless they are explicitly diagnostic.
- Classify `train_path_budget_ms down, step not faster` as `off-critical-path`.

## ejkernel / EasyDeL takeaways (March 2026)

Local reference repos:
- `~/Projects/Work/Marin/ejkernel`
- `~/Projects/Work/Marin/EasyDeL`

Most relevant takeaway from those references:
- the strongest TPU-relevant difference is not a brand-new training algorithm,
  but a different backward tradeoff:
  - smaller saved residual contract,
  - recompute-heavy backward from raw inputs plus chunk-start state,
  - simpler chunk-level control surface.

How to use that evidence:
- do not blindly port the entire stack;
- build a control arm that steals the backward/tape idea first;
- treat it as a chunk-first experimental branch:
  - minimize saved residuals,
  - recompute prepare/intermediate tensors in backward,
  - prefer a plain chunked path before reintroducing segment hierarchy,
  - try chunk sizes `{32, 64}` if the arm changes geometry.

What not to over-interpret:
- the wrapper spans multiple backends,
- inference fast paths are not direct evidence about the TPU training path,
- benchmark inference and training separately if you explore those paths.

What not to do:
- Do not spend mainline budget on forward-only GDN wins that do not move `step_duration_ms`.
- Do not promote candidates whose train-path budget improves but whose `remainder_budget_ms` grows.
- Do not treat Macro O as a near-champion if it only shifts work out of the tracked GDN path.

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
- For bottleneck attribution probes, temporarily approximate triangular solves is allowed if explicitly marked as probe-only and never promoted as champion code.
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

## New high-leverage macro moves (post-Iteration 32)

### G) Eliminate Ct^2 exponentials in `exp_diff` via centered outer-product exp
Target: replace Ct x Ct elementwise exponential work with O(Ct) vector exponentials plus
outer-product + clamp.

Why:
- Current flash prepare/recurrent/backward paths repeatedly build
  `exp_diff = exp(clip(g_i - g_j, -clip, clip))` as Ct x Ct exponentials.
- On TPU Pallas, `exp` is expensive. Reducing the count of exponentials can move the dominant
  custom-call critical path.

Implementation sketch:
- Add helper `_exp_diff_and_mask_from_g(g, clip=80.0)` returning `(exp_diff, mask)`.
- Fast path when `range(g) <= 2 * clip`:
  - `center = 0.5 * (g_max + g_min)`
  - `er = exp(g - center)`, `ec = exp(center - g)`  (vector exponentials)
  - `exp_factor = er[:, None] * ec[None, :]`
  - `exp_diff = clip(exp_factor, exp(-clip), exp(clip))`
  - `mask = (exp_factor > exp(-clip)) & (exp_factor < exp(clip))`
- Fallback path: current `diff/clip/exp` implementation for exactness in extreme ranges.

Success signal:
- Fewer `exponential` ops in the train-path custom call IR and reduced forward/backward
  `shard_map/pallas_call` wall time in traces.

### H) Batch matmuls by stacking
Target: reduce the number of separate `dot_general` calls in hot kernels.

Common opportunities:
- `QK` and `KKT` share `k^T`:
  - stack `[q; k_beta] @ k^T`, then split.
- `inter` and `v_prime` share `S`:
  - stack `[q_scaled; k_cumdecay] @ S`, then split.

Goal:
- Lower per-step dot launch overhead and improve MXU utilization without changing math.

### I) Fuse segmented forward prepare + recurrent only after G/H
Target: avoid recomputing expensive chunk-local factors and reduce intermediate traffic.

Why:
- Prior full-sequence fusion attempts were often low-impact because they fused launch structure
  without removing duplicated heavy work.
- After G/H, fusion can become meaningfully additive.

Constraint:
- Keep forward tape contract stable initially (`v_pseudo`, `k_cumdecay`, `solve_transform`) to
  avoid full backward rewrites in first attempts.

### J) Sweep `Ct`/`Seg` explicitly
Target: identify a better operating point once kernel structure changes.

Required sweep:
- `Ct in {64, 96, 128}`
- `Seg in {8, 16, 32}`

Record:
- Forward/backward train-path `shard_map/pallas_call` wall times.
- End-to-end MFU/tokens-per-second.

### K) Chunk-level affine associative scan (longer-term)
Target: reduce strict serial dependence across chunks.

Idea:
- Express chunk update as `S_out = M_chunk * S_in + U_chunk`.
- Compose chunks associatively and use prefix-scan style state propagation.

Risk:
- High algorithmic complexity. Attempt only after G/H/I are fully explored.

### R) ejkernel-style minimal-tape backward recompute arm
Target: validate whether a smaller residual contract plus recompute-heavy backward produces a better train-step tradeoff on TPU than the current save-heavy path.

Core idea:
- save raw inputs plus the minimum chunk-start state,
- do not carry large per-chunk tapes such as `v_pseudo`, `k_cumdecay`, or `solve_transform` unless measurement says they are still necessary,
- recompute prepare intermediates in backward instead,
- keep the experimental control surface chunk-first and simple.

Expected upside:
- lower tape traffic, smaller custom-VJP residuals, and a train-shell boundary that better matches TPU tradeoffs.

Implementation notes:
- this is a control arm, not a blind port;
- compare against the current deployable head under fixed CE settings;
- if the arm changes chunk geometry, sweep chunk sizes `{32, 64}` first;
- if train-path budget drops but step duration does not, classify it as `off-critical-path` and revert.

## Current high-priority macro moves (post-control-flow diagnosis)

These moves supersede the previous priority order for upcoming iterations.

### P) CE backward-mode A/B on the real train run
Target: determine whether the remaining CE-attributed control cost is best reduced by switching the TPU CE backward mode.

Required matrix:
- current deployable head + `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu` + Pallas CE backward
- current deployable head + `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu` + XLA-streaming CE backward
- optional sanity run: explicit `xla` CE when bottleneck attribution is unclear

Implementation note:
- keep the CE implementation fixed at `pallas_tpu` while toggling the backward mode env:
  - `LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH=0`
  - `LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH=1`
- profile the exact same train run; do not rely on microbenchmarks only.

### L) Associative chunk summaries / affine scan reformulation
Target: stop expressing training as a large serial scan over chunk kernels plus tapes.

Idea:
- Summarize each chunk as an affine update on recurrent state, e.g. `S_out = A_chunk(S_in) + B_chunk`.
- Compose chunk summaries associatively and compute chunk-start states with a prefix-style scan
  rather than a large lowered train-path `WhileOp`.

Why:
- The repeated `while ~31.6 ms` regressions indicate the serial train shell is now the wall.
- If chunk summaries can be composed in parallel, that attacks the actual bottleneck layer.

### M) XLA-first outer train path with Pallas only as leaf kernels
Target: move outer train-path orchestration out of custom-VJP/scan-heavy Pallas scaffolding.

Idea:
- Keep Pallas for chunk-local dense work where it clearly helps.
- Express outer state propagation, orchestration, and backward structure in XLA/JAX arrays.
- Avoid making Pallas/custom VJP responsible for the whole train-path program skeleton.

Why:
- Current evidence suggests the train-shell lowering is the bottleneck, not just the chunk microkernels.

### N) Backward tape-contract redesign
Target: reduce scanned residual state in reverse mode.

Idea:
- Replace large per-segment tape tuples with compressed chunk summaries, recomputation/remat,
  or a mathematically equivalent backward contract that threads less state through the reverse scan.

Constraint:
- Do not do a tape-only bandwidth tweak in isolation. Pair it with a real outer control-structure change.

### O) Reduced-Pallas / XLA control arm
Target: test whether the current train-path abstraction boundary is fundamentally wrong.

Idea:
- Build a control branch that uses fewer or no custom Pallas calls in the training path.
- Compare end-to-end train MFU directly against the current champion using the same validation profile.

Why:
- This is the quickest way to validate whether the pure-Pallas train shell is boxing the search in.

## Current priority order

1. `P` CE backend forcing / A-B benchmark
2. `O` reduced-Pallas / XLA control arm
3. `M` XLA-first outer train path
4. `N` backward tape-contract redesign inside `M`/`O`
5. `L` associative summaries only when paired with `M`/`O` or when CE is no longer the dominant unresolved bottleneck
6. `E/H/G/I/J` only when nested inside one of the above or when they explicitly suppress
   the train-path `while` / `conditional` buckets

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

### Structured hotspot reporting

Every iteration writeup should include these exact categories when available:

- `Forward closed-call`
- `Backward closed-call`
- `while`
- `conditional`
- `Kernel budget`
- `Control budget`
- `Train-path budget`

The unattended harness now parses these fields from the hillclimb log and uses them for
promotion/rejection policy. If they are missing, the harness loses visibility into the
real train-path bottleneck.

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
  --model gpt-5.4 \
  --reasoning-effort xhigh \
  --resilient \
  --directive-preset training-chunk-kernel-focus \
  --directive-preset ce-backend-priority \
  --directive-preset control-structure-pivot \
  --directive-preset macro-coverage-pivot \
  --directive-preset remainder-attribution-mainline \
  --directive-preset model-boundary-sweep \
  --validation-profile-ce-implementation pallas_tpu \
  --validation-profile-ce-bwd-mode pallas \
  --perf-upper-bound-step-ms 57.860499 \
  --dirty-policy stash \
  --no-commit-policy count-failure \
  --hold-dev-tpu \
  --dev-tpu-cluster us-east5-a \
  --dev-tpu-fallback-cluster us-central1 \
  --dev-tpu-name "$USER-gdn" \
  --dev-tpu-type v5p-8 \
  --dev-tpu-allocate-attempts 2 \
  --dev-tpu-allocate-retry-sleep 20 \
  --prompt-file scripts/gdn/codex_iteration_prompt.md \
  --post-check "uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name $USER-gdn --tests both" \
  --post-check "uv run python scripts/gdn/gdnctl.py lint-log"
```

Notes:
- Use `--codex-profile <profile-name>` if you have a Codex CLI profile for `gpt-5.4` / high reasoning settings.
- `codex-loop` defaults to `codex exec --ephemeral` so long unattended runs do not persist thousands of loop-created Codex sessions/threads into the app database. Opt out with `--no-codex-ephemeral` only when you explicitly need saved sessions for debugging or resume.
- By default, `codex-loop` stops if an iteration does not create a new commit.
- Use `--allow-dirty` or `--allow-no-commit` only when intentionally debugging the loop harness.
- The default iteration prompt (`scripts/gdn/codex_iteration_prompt.md`) is intentionally aggressive; keep it aligned with this policy.
- Use `--resilient` for unattended loops to keep running through transient failures (network, connectivity, allocation).
- `--directive`, `--directive-file`, and `--directive-preset` inject per-session guidance without editing prompt files.
- Preset directives are stored as markdown docs under `scripts/gdn/session_directives/`.
- For the current phase, prioritize:
  - `remainder-attribution-mainline`
  - `model-boundary-sweep`
  - `control-structure-pivot`
  - `ce-backend-priority`
  - `tpu-layout-and-dtypes`: avoid TPU register-layout cliffs (singleton last-axis, transpose fusion, BF16/F32 policy).
  - `emit-pipeline-fullseq`: design sketch for collapsing segmentation using `pltpu.emit_pipeline`.
- Use `--validation-mode profile-only` when running intentional ablation probes that may fail correctness tests.
- Use `--validation-profile-env KEY=VALUE` (repeatable) to inject profile-only environment switches such as:
  - `--validation-profile-env GDN_TRIANGULAR_SOLVE_PROBE=identity`
  - `--validation-profile-env GDN_TRIANGULAR_SOLVE_PROBE=first_order`
- Prefer `--dirty-policy stash --no-commit-policy count-failure` for unattended long runs so failed attempts do not permanently block progress.
- `--dirty-policy stash` restores the stashed tree automatically after each iteration.
- If stash restore conflicts with edits produced in the iteration, default `--stash-restore-policy warn-keep` keeps the stash and continues; use `--stash-restore-policy fail` for strict stop-on-conflict.
- Add `--hold-dev-tpu --dev-tpu-name <name>` to make `codex-loop` allocate/hold/release a dev TPU allocation for the entire loop session.
- In managed dev TPU mode, use `dev-tpu-test`/`dev-tpu-profile` (not Ray TPU test/profile commands) for loop validation/profiling.
- Keep managed-mode `--post-check` commands aligned with the held allocation `--cluster` and `--tpu-name`.
- `ray-test` and `dev-tpu-test` now install both `torch` and `transformers` before running `tests/test_gdn_kernels.py` and `tests/test_gdn_layer.py`. Treat `49 passed, 40 skipped` as an incomplete dependency-starved result; the full HF parity slice should look like `87 passed, 2 skipped`.
- Performance governance is enabled by default:
  - `--perf-mode required` ensures each validated iteration has parseable performance metrics.
  - `--perf-metric throughput/mfu` controls champion comparisons.
  - `--perf-min-improvement-pct` and `--perf-max-regression-pct` define promotion/regression thresholds.
  - `--perf-regression-policy revert-count-failure` (default) auto-reverts regressing commits with non-destructive `git revert --no-edit`.
  - `--perf-state-file` overrides the default champion-state file `<log-dir>/perf_state.json`.

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
- Do not leave `Commit: (pending)` or `Commit: this commit` in new entries. Use an exact SHA or `none (failed attempt)`.
