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

## Task DAG status (see memex grug-fp8-task-dag for full ~1-day breakdown)
- **Done:** W0 (pre-start), R1 (accounts/comms), R2 (local dev env), R3 (H100 cloud env + profiling).
- **Next:** R4 (reproduce BF16 baseline @ d3072/d4096), R5 (code-read map incl. team's *previous* FP8
  work), then Phase-1 spike S1–S7 → S8 report → S9 review gate.
- **First concrete deliverable (David):** a linear (dense MLP) FP8 benchmark harness at d3072/d4096,
  then analogous for `gmm` + permute ops.

## Experiment Log

### 2026-06-17 — Kickoff (research thread set up)
- Created research branch `research/grug-fp8-h100` and this logbook from the existing memex plan.
- Adjacent issues reviewed: #6367 (BF16 d2560 H100 MFU push — baseline + branch/logbook convention),
  #5816 (FP8 recipe on **B200/Blackwell** — sibling, different hardware). This thread is Hopper/H100
  per-tensor FP8.
- Next action: file the dedicated experiment issue (cross-linking #6367, #5816), then start R4/R5.

<!-- New entries below this line. Template:
### YYYY-MM-DD HH:MM - <GFP8-NNN short label>
- Hypothesis:
- Command:
- Config:
- Result:
- Interpretation:
- Next action:
-->
