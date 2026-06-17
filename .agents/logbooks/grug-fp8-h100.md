# Grug FP8 on H100: Research Logbook

Append-only research logbook for the **FP8 quantized training of the Grug MoE LM on NVIDIA H100s**
work-trial project. Companion to the experiment issue (link added at kickoff) and the spec on
`codex/grug-fp8-h100-spec` (`.agents/projects/grug_fp8_h100/{spec,design,research}.md`).

Experiment ID prefix: **GFP8** (e.g. `GFP8-001`). Use these IDs in logbook entries, W&B run names,
and issue comments.

## Scope
- **Goal:** Enable FP8 quantized training of the Grug MoE LM on H100, flag-gated via
  `GrugFp8Config(enabled=False)` (default ⇒ behavior unchanged). Recipe: TE-style per-tensor
  delayed scaling — E4M3 fwd activations/weights, E5M2 output grads, FP32 accumulation. Quantize
  attention Q/K/V/O projections, dense/shared MLP, and routed-MoE expert GEMMs; keep router logits,
  top-k, softmax, norm, QB beta, attention-score softmax, and loss in full precision.
- **Primary metric(s):** Training MFU (BF16 GPU baseline ~15% → FP8 target **20–25%**, per David's
  day-1 orientation). Secondary: tok/s, step-time breakdown, peak HBM, compile time, loss/NaN-freeness,
  finite router metrics. Kernel-level GEMM throughput tracked separately if comm-bound.
- **Constraints:** Respect dependency direction `haliax → levanter → experiments/grug`. No backward
  compat shims. TPU-int8 naming ("quantized dot", not "fp8 dot"). Pin versions (JAX 0.10 / XLA / CUDA —
  scaled-dot behavior is version-sensitive); fix seeds/data; laptop↔H100 parity. Work in the open
  (keep the issue current); don't disclose compute donors. PRs/reviews via Discord `#code-review`.
- **Out of scope:** Blackwell MXFP8 / blockwise grouped-MoE (Hopper blockwise is a stretch goal),
  FP8 optimizer states, FP8 router/softmax/loss/norm, portable checkpoint export, Grug→Haliax-module
  rewrite, full TE API.

## Three lowering paths (Phase-1 decides on HLO + profiler evidence; dense and MoE may differ)
Hypotheses to test, **not** a ranking — spec says "do not assume A, B, or C is the right target."
- **Dense A** — Delayed-scaling Q/DQ around `dot_general` (Flax/Haliax style). Closest to existing
  Haliax state model; risk: relies on XLA pattern-matching, can silently miss H100 FP8 kernels — must
  verify `__cublas$lt$matmul$f8` + tensor-core counters.
- **Dense B** — Qwix `dot_general_qt` (MaxText `fp8_full` style). Covers dense + ragged uniformly;
  risk: likely framework mismatch with Haliax's functional pytree state (assess, don't assume).
- **Dense C** — `jax.nn.scaled_dot_general` (explicit XLA scaled-dot / cuDNN). Cleanest IR; risk: its
  config advertises mxfp8/nvfp4 (Blackwell-oriented), no ragged analog so MoE still needs A or B.
- **MoE candidates:** Pallas-Triton ragged-dot, XLA `ragged_dot_general(precision=?)`, Qwix
  `ragged_dot_qt`. Prior: dense `Fp8Dot` presumed **wrong** for ragged until proven otherwise.

## Baseline
- **Date:** TBD (R4 — reproduce BF16 Grug MoE baseline; in progress).
- **Reference recipe (May Recipe, marin#6044):** d2560, 26 layers, GQA 4:1, 256 experts top-4 + shared
  expert, GatedNorm, XSA, half-RoPE, PKO (7/26 layers), ring dispatch, MuonH optimizer; precision
  `params=fp32 / compute=bf16 / output=bf16` (the BF16 baseline FP8 must beat). TPU v4-1024 ref run.
- **FP8 benchmark target shape (David, day-1):** dense MLP at **hidden dim 3072 & 4096**, seq 4096.
  d3072 matches Russell's real 90B-scale H100 run.
- **Adjacent reference runs:** Russell's 90B-total/5.3B-active Grug MoE (128 experts top-4, d3072 L48)
  across 256 H100 on `cw-us-east-02a` at ~3% MFU (the gap FP8 + other opt closes). BF16 MFU push
  tracked in issue #6367 (`.agents/logbooks/grug-moe-d2560-mfu.md`).
- **Baseline numbers:** TBD.

## Environment / repro
- **Cluster:** CoreWeave H100 via Iris (Pod-per-task, no SSH). Live cluster `cw-us-east-02a`
  (checked-in `coreweave-*.yaml` US-WEST-04A configs are stale for this account).
- **Submit:** `uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 --enable-extra-resources --extra gpu -- <cmd>`
- **Artifact retrieval (no `job cp`, only `job logs`):**
  (A) print `jax.jit(f).lower(*args).compile().as_text()` to stdout, grep logs for
  `__cublas$lt$matmul$f8` (operands `f8e4m3fn`/`f8e5m2`) — zero storage.
  (B) `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text"` (set before backend init) +
  `jax.profiler.trace("<dir>")`, upload to `s3://marin-na/…` (R2). Creds via
  `-e AWS_ACCESS_KEY_ID/SECRET/ENDPOINT_URL`, `AWS_REGION=auto`.
- **HLO-dump helper pattern:** `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py`.
- **Local GPU (laptop, satori 3090):** GPU JAX works with `LD_LIBRARY_PATH=/run/opengl-driver/lib`;
  GPU deps via `uv sync --package marin-levanter --extra gpu`. Run the 3 acceptance test files as
  **separate** pytest invocations (single-command form hits a conftest package collision).

## Stop criteria
- **Phase-1 (decision gate):** dense + MoE lowering paths selected with HLO/profiler evidence;
  MoE viability verdict (implement vs documented-blocker, criterion-9 hatch); perf targets derived.
  Report signed off by team review.
- **Ship:** acceptance criteria 1–12 green (pre-commit + 3 test files; FP8 histories update under JIT
  with clean optimizer/EMA state; H100 synthetic profiles show FP8 matmuls for dense + expert GEMMs
  or documented blocker; NaN-free real-data run; BF16-vs-FP8 benchmark beats baseline toward 20–25% MFU).

## Plan (streamlined spine — maps 1:1 to spec phases; full ~1-day breakdown in memex grug-fp8-task-dag)

Three ramp/spike tasks all emit "HLO/profiler output on an H100" and are easy to conflate — they have
distinct purposes:
- **R3** (✅ 6/15): prove the *profiling pipeline* works — HLO + trace + TC counters for a **toy** FP8
  dot. Synthetic data, 1×H100.
- **R4** (⬜ next): prove the **BF16 Grug *model*** runs → establish the baseline — synthetic step +
  short real-data run (~200–500 steps SlimPajama-6B), loss finite, baseline profiler trace. Reduced
  single-node config first, then EP mesh.
- **S1** (⬜ Phase 1): reusable **dense-dot microbench rig** that paths A/B/C plug into — configurable
  shapes/dtypes, BF16 dense-dot timing + HLO. Synthetic data, 1×H100.

**David's "first concrete deliverable"** (linear-FP8 benchmark harness @ d3072/d4096, then `gmm` +
permute) **= S1 rig + S2 path-A FP8, then S5 + S6** — *not* R4. It depends only on R3 (done), so it
can run **in parallel with** R4.

- **Phase 0 — Ramp to green baseline** (days 1–3): R1 ✅ · R2 ✅ · R3 ✅ · **R4 model baseline** ⬜ ·
  R5 code-read map ⬜ → Gate G0.
- **Phase 1 — Lowering-path spike → decision gate** (days 4–8): dense **S1→S2/S3/S4** · ragged
  **S5→S6a/S6b** · perf targets **S7** (uses R4 profile) · report **S8** → **team review S9 = hard gate**.
- **Phase 2a — Dense FP8** (spec Wk1): Haliax helper → **PR1**; Grug dense wiring + OWG train step → **PR2**.
- **Phase 2b — MoE FP8** (spec Wk2): ragged `precision`, `QuantizedRaggedDot`, wire experts → **PR3**
  (or criterion-9 documented-blocker hatch).
- **Phase 3 — Validate + benchmark** (spec Wk3–4): V1 synth NaN-free · V2 real-data · V3 BF16-vs-FP8
  benchmark · V4 vs targets · V5 final lint + 3-file pytest suite.

## Experiment Log

### 2026-06-17 — Kickoff (research thread set up)
- Created research branch `research/grug-fp8-h100` and this logbook from the existing memex plan.
- Adjacent issues reviewed: #6367 (BF16 d2560 H100 MFU push — baseline + branch/logbook convention),
  #5816 (FP8 recipe on **B200/Blackwell** — sibling, different hardware). This thread is Hopper/H100
  per-tensor FP8.
- Next action: file the dedicated experiment issue (cross-linking #6367, #5816), then start R4/R5.

### 2026-06-17 — Plan streamlining pass (de-drift R4 vs S1/S5)
- Reconciled this logbook against the source plan (spec branch `codex/grug-fp8-h100-spec` + memex
  task DAG). No structural drift; one labeling artifact fixed.
- **Fix:** the task-DAG R4 line had stapled the dense-MLP/d3072 *benchmark harness* (David's "first
  deliverable") onto the BF16 *model* baseline. Peeled apart — R4 = model baseline only; the harness
  = S1 (dense rig) + S2 (path-A FP8) + S5 (ragged rig) + S6. S1 depends only on R3, so the first
  deliverable can start in parallel with R4.
- Updated the Plan section above; mirrored the same fix into memex `grug-fp8-task-dag.md` (R4/S1/S5).

<!-- New entries below this line. Template:
### YYYY-MM-DD HH:MM - <GFP8-NNN short label>
- Hypothesis:
- Command:
- Config:
- Result:
- Interpretation:
- Next action:
-->
