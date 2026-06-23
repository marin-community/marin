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
- **Date:** TBD (R4 — reproduce BF16 Grug MoE baseline; **not started**).
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
- **R4** (⬜ in progress): prove the **BF16 Grug *model*** runs → establish the baseline.
  **Real data, small slice** (SlimPajama-6B cache confirmed complete on R2 — cache hit, no tokenize):
  reduced single-node config, ~10-step smoke → BF16 throughput/MFU baseline + profiler trace.
  Synthetic harness dropped (2026-06-17 — net complexity for no benefit once cache confirmed).
  The FP8 *real-data NaN/router* de-risk remains separate at **V2**.
- **S1** (⬜ Phase 1): reusable **dense-dot microbench rig** that paths A/B/C plug into — configurable
  shapes/dtypes, BF16 dense-dot timing + HLO. Synthetic data, 1×H100.

**David's "first concrete deliverable"** (linear-FP8 benchmark harness @ d3072/d4096, then `gmm` +
permute) **= S1 rig + S2 path-A FP8, then S5 + S6** — *not* R4. It depends only on R3 (done), so it
can run **in parallel with** R4.

- **Phase 0 — Ramp to green baseline** (days 1–3): R1 ✅ · R2 ✅ · R3 ✅ · **R4 model baseline** ⬜ ·
  R5 code-read map ✅ (parallel session) → Gate G0 (blocked on R4).
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

### 2026-06-17 — R5 done: code-read map (via parallel session)
- R5 completed in the parallel session (memex `b6e5dfb`, note `2026-06-17-grug-fp8-r5-code-map.md`);
  key findings folded here so the logbook stands alone.
- **Existing FP8 path = fragile QDQ pattern-matching.** `dot_general_with_precision` (haliax
  `fp8.py:121-133`) forces `Precision.DEFAULT` and *discards* the precision arg, leaning on XLA's
  GemmRewriter to emit `__cublas$lt$matmul$f8`. Near-verbatim Flax port, algorithm unchanged since the
  2025-11 ingest.
- **"Slower today" explained.** No `scaled_dot_general` / preset / `kScaledDot` usage anywhere in
  marin — the QDQ code stood still while XLA gained explicit `kScaledDot` / `ScaledDotRewriter`, so the
  fragile match can silently fall back to bf16.
- **Head start.** David prototyped dense wiring on unmerged `origin/codex/grug-generic-fp8`
  (`9af5f5d09`: `levanter/grug/linear.py` + grug model/train wiring + `test_grug_variant_contracts`
  FP8 cases) — useful for H1/C1/C2, but it reuses the suspect `Fp8DotGeneralOp`.
- **Reframes S2:** path A is a *re-measurement* of an existing-but-maybe-broken QDQ path, not a fresh
  experiment.

### 2026-06-17 — R4 start: dry-run validates reduced config + surfaces tokenize dependency
- **Vehicle:** `experiments/grug/moe/launch_cw_scale.py` — env-parameterized H100 launcher, BF16
  policy `params=float32,compute=bfloat16,output=bfloat16`, profiler support, executor-submitted.
- **Dry-run (free):** `SCALE_GPU_REPLICAS=1 SCALE_HIDDEN_DIM=3072 SCALE_NUM_LAYERS=4 SCALE_NUM_EXPERTS=8
  SCALE_EXPERT_AXIS=8 SCALE_BATCH=16 SCALE_SEQ_LEN=2048 SCALE_STEPS=10 SCALE_TRACKER=json_logger
  SCALE_CHECKPOINTS=local uv run python experiments/grug/moe/launch_cw_scale.py --dry_run true`
  → reduced single-node config builds; mesh divisibility OK (num_experts 8 % expert_axis 8 = 0;
  batch 16 % batch_shards 8 = 0). Step graph = **2 steps**: a SlimPajama-6B **tokenize** + training.
- **Cost-critical finding:** the real-data run triggers a 6B-token SlimPajama tokenize unless the
  tokenized cache already exists **in-region** — must verify before any live real-data run (AGENTS.md
  cross-region guardrail). Motivates doing the **synthetic-data** smoke first (no tokenize dependency).
- **GPU floor:** one 8×H100 node without editing the launcher (`GPUS_PER_NODE=8` hardcoded);
  single-GPU needs a file edit. Synthetic data available via `DirectDatasetComponent` (small harness).
- **Next:** confirm R4 sequence (synthetic smoke → real-data baseline) + scale with Matt before spend.

### 2026-06-17 — R4 scope decision: synthetic-only (real-data → V2)
- **Decision:** R4 reproduces the BF16 *model* baseline on **synthetic random tokens only**; the
  short real-data run moves to **V2** (criterion 10).
- **Why:** synthetic is cleaner for the BF16 throughput/MFU baseline (no data-loader variance), and
  real data's genuine de-risking is *FP8-phase* — (1) FP8 amax/range stability (E4M3 overflow on
  real-data activation outliers) and (2) router/MoE sanity + realistic ragged group-size
  distributions. BF16's wide exponent range makes a real-data NaN-check low-value at the baseline.
- **Bonus:** removes the SlimPajama-6B tokenize + cross-region exposure from the ramp; V2 will verify
  the in-region cache before the one real-data run.
- Mirrored to memex `grug-fp8-task-dag.md` (R4 synthetic-only, V2 absorbs the real-data de-risk).
- **Next:** kick off R4a — synthetic `DirectDatasetComponent` + reduced single-node BF16 smoke on `cw-us-east-02a`.

### 2026-06-17 — R4 data source resolved: real-data slice (SlimPajama-6B cache confirmed on R2)
- **Reversed the synthetic-only call.** Matt pushed on whether a small real-data slice beats a
  synthetic harness; investigation confirms it does, and cheaply.
- **Tokenize cost:** eager batch job (`marin/processing/tokenize/_core.py:307` `take_per_shard`), but
  bounded by `sample_count`; moot here because the cache already exists.
- **Cache CONFIRMED complete on R2** (read-only `aws s3 ls`, endpoint
  `74981a43…r2.cloudflarestorage.com`, prefix `s3://marin-na/marin`):
  `tokenized/slimpajama-6b-cw-31d2b0/` has `.executor_status = SUCCESS`, train+validation splits,
  `train/.stats.json = {total_tokens: 5,393,440,120, total_elements: 5,489,000}`, 41 shards
  (materialized 2026-03-12). Hash `31d2b0` matches the dry-run's expected output path → **executor
  skips tokenize (cache hit); zero tokenize, zero cross-region cost.**
- **R4 plan:** reduced-config `launch_cw_scale.py` with `slimpajama_6b_data()` as-is, ~10 steps, BF16,
  json_logger, on `cw-us-east-02a`. Synthetic harness dropped. Launch command pending; **submit on
  Matt's go-ahead only.**

### 2026-06-17 — R4 submission mechanism verified + draft launch command (NOT run)
- **Mechanism (verified):** `launch_cw_scale.py` uses `executor_main` + `ResourceConfig.with_gpu` —
  identical shape to `experiments/ferries/canary_ferry.py`. `get_iris_ctx()`
  (`lib/iris/src/iris/client/client.py:1072`) detects Iris via `get_job_info()` (pod env): laptop ⇒
  `LocalClient` (the dry-run), inside an Iris pod ⇒ auto-binds to the controller. So the pattern is
  the canary runbook: wrap the launcher in a small CPU `iris job run`; the launcher pod detects Iris
  and submits the H100 training step.
- **Confirmed config (Matt):** `SCALE_GPU_REPLICAS=1` (one 8×H100 node), d3072 / L4 / 8 experts /
  EXPERT_AXIS=8 / TOP_K=4 / batch 16 / seq 2048 / 10 steps. Divisibility OK (dry-run validated).
- **Draft submit (NOT run; awaiting go-ahead):**
  ```
  direnv exec . uv run iris --cluster=cw-us-east-02a job run \
    --job-name grug-r4-bf16-smoke-<ts> --cpu 2 --memory 8GB --disk 16GB --extra cpu \
    -e MARIN_PREFIX s3://marin-na/marin \
    -e AWS_ACCESS_KEY_ID "$R2_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY "$R2_SECRET_ACCESS_KEY" \
    -e AWS_ENDPOINT_URL https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com \
    -e AWS_REGION auto -e AWS_DEFAULT_REGION auto \
    -e SCALE_GPU_REPLICAS 1 -e SCALE_HIDDEN_DIM 3072 -e SCALE_NUM_LAYERS 4 \
    -e SCALE_NUM_EXPERTS 8 -e SCALE_EXPERT_AXIS 8 -e SCALE_TOP_K 4 \
    -e SCALE_BATCH 16 -e SCALE_SEQ_LEN 2048 -e SCALE_STEPS 10 \
    -e SCALE_TRACKER json_logger -e SCALE_CHECKPOINTS local \
    -- python experiments/grug/moe/launch_cw_scale.py
  ```
- **Unknowns to resolve before running:** (1) `NCCL_SOCKET_IFNAME` — canary sets it for multi-GPU
  NCCL; cw-us-east-02a interface value TBD; (2) HF_TOKEN likely unneeded (cache hit) — include if
  graph-build complains; (3) confirm the spawned GPU step inherits the correct GPU image/extra.

### 2026-06-17 — R4 launch fully de-risked (3 unknowns resolved; command finalized, NOT run)
- **NCCL_SOCKET_IFNAME:** cluster-injected via `cw-us-east-02a.yaml` `task_env: NCCL_SOCKET_IFNAME:
  =enp157s0np0` (+ `host_network: true`). Don't pass it.
- **HF_TOKEN:** not needed (cache hit ⇒ executor skips tokenize ⇒ no HF download). `create_environment`
  also auto-forwards it from the launcher env if set (`fray/types.py`). Omit.
- **GPU image/deps:** spawned step auto-derives deps via `dependency_groups_for_resources(resources,…)`
  (`step_runner.py:382`) from its `with_gpu("H100",…)` ResourceConfig; base image =
  `default_task_image: ghcr.io/marin-community/iris-task:latest`. No manual `--extra gpu`.
- **R2 creds → child step (new risk, resolved):** auto-injected into **all** worker/task pods
  (`coreweave.md:594-606`): `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` from the `iris-s3-credentials`
  Secret (created from `R2_*` at `iris cluster start`), `AWS_ENDPOINT_URL` + `FSSPEC_S3` from config,
  `MARIN_PREFIX` from `task_env`. So no storage env needs passing — cluster manages it.
- **Also:** `buffer_slices: 32` ⇒ 32 H100 nodes pinned warm (fast scheduling, no ~20-min provision).
- **Final command (NOT run; submit only on Matt's go-ahead) — only SCALE_* + resources:**
  ```
  direnv exec . uv run iris --cluster=cw-us-east-02a job run \
    --job-name grug-r4-bf16-smoke-<ts> --cpu 2 --memory 8GB --disk 16GB --extra cpu \
    -e SCALE_GPU_REPLICAS 1 -e SCALE_HIDDEN_DIM 3072 -e SCALE_NUM_LAYERS 4 \
    -e SCALE_NUM_EXPERTS 8 -e SCALE_EXPERT_AXIS 8 -e SCALE_TOP_K 4 \
    -e SCALE_BATCH 16 -e SCALE_SEQ_LEN 2048 -e SCALE_STEPS 10 \
    -e SCALE_TRACKER json_logger -e SCALE_CHECKPOINTS local \
    -- python experiments/grug/moe/launch_cw_scale.py
  ```
  (`direnv exec .` is only for KUBECONFIG so the CLI reaches the controller; storage creds are
  cluster-side.) Final pre-launch check: `direnv exec . uv run iris --cluster=cw-us-east-02a cluster status`.

### 2026-06-17 — R4 write-up (motivation, goals, parameters)
*Motivation & goals.* Russell's 256-H100 run already shows the full-recipe BF16 Grug MoE trains on
H100, but it's a one-off production bring-up — too expensive to re-run per experiment and not set up
as an FP8 control. R4 instead establishes a **cheap, single-node (8×H100), reproducible BF16 reference
on the exact launcher and flag-gated config we'll flip FP8 on**, so the eventual FP8 claim is a clean
apples-to-apples delta (same harness, seeds, shape, profiler) rather than a comparison across differing
runs. This first run is a deliberately small **correctness + plumbing smoke**: confirm the BF16 model
compiles and runs forward+backward with finite loss while exercising the real expert-parallel MoE ring
dispatch, and validate the full path — executor→Iris submission, SlimPajama cache hit, profiler/metric
capture — at minimal cost (~$20-30 on a warm node). It yields the first throughput datapoint and
profiler trace feeding the Phase-1 perf-target derivation (S7), and clears Gate G0 before any FP8
lands. The full benchmark-shape baseline (deeper/wider, longer profiler window) follows once this smoke
confirms the path.

*Test parameters.*
- **Hardware:** 1 node = 8xH100 on `cw-us-east-02a` (`SCALE_GPU_REPLICAS=1`), pure intra-node expert
  parallelism (`EXPERT_AXIS=8`, `REPLICA_AXIS=1`).
- **Model (reduced):** hidden 3072, 4 layers, 8 experts top-4, seq 2048.
- **Precision:** BF16 recipe — `params=fp32 / compute=bf16 / output=bf16`.
- **Run:** batch 16, 10 steps; `json_logger` tracker; local (disposable) checkpoints.
- **Data:** SlimPajama-6B real slice — tokenized **cache hit** on R2 (`slimpajama-6b-cw-31d2b0`), so no
  tokenization and no cross-region cost.

### 2026-06-17 — Decision: first public push after R4 baseline
- **When to go public** (per run-research conventions + David's "work in the open"): post the
  experiment issue + push the research branch **after the R4 baseline smoke yields a datapoint**, so
  the issue opens with a concrete result rather than pure plan. (Convention biases earlier — issue is
  a kickoff artifact — but holding until R4 is a deliberate "open with a number" call.)
- **Sequence:** R4 smoke (held for Matt's go-ahead) → post issue (`grug-fp8-h100-issue-draft`) + push
  `research/grug-fp8-h100` so logbook permalinks resolve → issue updates per milestone.
- **Code** stays separate: clean PR-candidate branch, PRs at PR1/PR2/PR3 checkpoints via Discord
  `#code-review` (two-branch convention). All public pushes confirmed with Matt first (standing rule).

<!-- New entries below this line. Template:
### YYYY-MM-DD HH:MM - <GFP8-NNN short label>
- Hypothesis:
- Command:
- Config:
- Result:
- Interpretation:
- Next action:
-->

### 2026-06-22 — GFP8-001: S1 dense microbench rig
- **Goal:** Build the dense-dot microbench rig (S1) — the BF16 baseline arm of David's
  linear-FP8 harness. Shapes/dtypes/timing/HLO/profiler hooks; FP8 paths (S2) extend the same file.
- **Code:** `lib/levanter/scripts/bench/bench_dense_fp8.py`, 3 commits on `research/grug-fp8-h100`
  (8aa4dcda9 core rig · d9b88150a HLO-dump/profiler hooks · 325927f42 roofline). Single dense
  projection `[M,K]@[K,N]` fwd+bwd via `lax.dot_general` with shared `_DIMENSION_NUMBERS` (so S2
  swaps only dtype/precision). Emits compiled HLO to stdout (path-A `iris job logs` read),
  `--xla-dump-dir`/`--profiler-dir` for the R2-upload path B, and one `result_json` line with
  achieved TFLOP/s + %-of-H100-peak (peak shown only when an H100 is detected).
- **Command (local verify):** `uv run python lib/levanter/scripts/bench/bench_dense_fp8.py
  --m 1024 --k 1024 --n 1024 --steps 5 --warmup 2`
- **Result:** Runs on CPU (JAX 0.10.0); emits forward HLO (`dot_general` op visible) + fwd/bwd
  timing + achieved TFLOP/s in `result_json`. `./infra/pre-commit.py` green. **S1 acceptance met**
  (BF16 dense-dot baseline runs; emits timing + HLO). Not yet run on H100 — %-peak and the real
  d3072/d4096 baseline numbers land on the first `cw-us-east-02a` run.
- **Interpretation:** Rig is the reusable measurement substrate for S2–S4 (dense lowering paths);
  the shared `dimension_numbers` + dtype/precision seam is the extension point for the Path-A FP8 dot.
- **Next action:** S2 — lower an `Fp8DotGeneral` at d3072/d4096 on H100 through this rig and grep the
  HLO for `__cublas$lt$matmul$f8` (+ `_FAST_ACCUM`?); same H100 run yields the BF16 baseline datapoint.

### 2026-06-22 — GFP8-002: S1 H100 BF16 baseline datapoint (job /matt/iris-run-job-20260622-185707)
- **Hypothesis:** The S1 rig runs on a real H100 and emits the BF16 dense-dot baseline (timing + HLO),
  with the HLO grep-able for the cuBLASLt matmul custom-call — the exact mechanism S2 reuses for `$f8`.
- **Command:** `uv run iris --cluster=cw-us-east-02a job run --no-wait --cpu 8 --memory 64GB
  --gpu H100x1 --enable-extra-resources --extra gpu -- bash -c 'python -u
  lib/levanter/scripts/bench/bench_dense_fp8.py --k 3072 --n 3072 && python -u
  lib/levanter/scripts/bench/bench_dense_fp8.py --k 4096 --n 4096 --no-print-hlo'`
- **Config:** 1×H100 (`CudaDevice(id=0)`), M=4096 (seq 4096), BF16, steps 20 / warmup 5, fwd+bwd.
- **Result (succeeded):**
  - d3072 `[4096,3072]@[3072,3072]`: fwd **435 TFLOP/s = 44.0%** of H100 BF16 peak (0.178 ms);
    bwd **600 TFLOP/s = 60.7%** (0.257 ms).
  - d4096 `[4096,4096]@[4096,4096]`: fwd **549 TFLOP/s = 55.5%** (0.250 ms);
    bwd **682 TFLOP/s = 68.9%** (0.403 ms).
  - HLO forward dot lowers to `custom_call_target="__cublas$lt$matmul"`, operands+output `bf16`,
    `precision_config.operand_precision=["DEFAULT","DEFAULT"]`, `scale_mode:0`, **no `__cublas$lt$matmul$f8`**
    (grep count 0) — the expected BF16 signature. Peak ref = H100 SXM BF16 989.5 TFLOP/s.
- **Interpretation:** S1 fully validated on hardware: rig runs, emits timing + HLO, %-of-peak fires on
  H100. The captured BF16 HLO is the reference S2 diffs against — the FP8 path must flip operands to
  `f8e4m3fn`, add scale operands, and set a nonzero `scale_mode`. Larger K (d4096) sits closer to peak,
  as expected (more arithmetic to amortize launch/IO). These are the controlled BF16 numbers the FP8
  arm must beat (~2x is the rough ceiling: H100 FP8 peak ~1979 TFLOP/s).
- **Ops notes:** (1) needed `uv pip install kubernetes` for the laptop-side controller tunnel. (2) Iris
  bundles the working tree to `/app` (12.8 MB) — the committed script runs at its repo path, no inlining
  needed. (3) **First submit failed** (`ModuleNotFoundError: jax`): wrapping in `bash -lc` (login shell)
  re-sourced profile and reset PATH off the Iris-activated `/app/.venv`; fix = `bash -c` (non-login) or
  run `python <path>` directly, as R3's hello-world did. (4) Retrieval was path A (HLO + result_json via
  `iris job logs`, zero storage); `python -u` for unbuffered logs.
- **Next action:** S2 — lower `Fp8DotGeneral` (manual `dq(op(q,q))` per David's 6/22 steer) at
  d3072/d4096 through this same rig; diff the HLO against this baseline for `$f8` + `_FAST_ACCUM`.

### 2026-06-22 — GFP8-003: S2 method — replicate the existing QDQ path under *delayed scaling*
- **Goal:** the R5 prior is that the existing `Fp8DotGeneralOp` QDQ path "silently falls back to BF16"
  — but the hypothesis is specifically about the **real delayed-scaling setting** (the MoE training
  path), where the per-tensor scales are *live runtime scalars* threaded from amax-history state. So S2
  replicates that path faithfully and asks: does XLA's GemmRewriter still fuse `__cublas$lt$matmul$f8`
  when the scales are live? Fixes (manual `dq(op(q,q))`, `scaled_dot_general`, fast-accum) come after.
- **Code:** `lib/levanter/scripts/bench/bench_dense_fp8.py`, 3 commits on `research/grug-fp8-h100-s2`
  (realistic bwd cotangent · print backward HLO · `--path qdq` arm). The `qdq` arm calls
  `haliax.quantization.Fp8DotGeneralOp` verbatim (E4M3 fwd weights/acts, E5M2 output grad). **Two
  faithfulness points that decide the result:**
  - *Delayed scaling, not constant scaling:* the op state (scales + amax histories) is passed as a
    **runtime jit arg**, not closed over. Closing it folds `compute_scale(history)` to a compile-time
    constant and bakes scale=const into the f8 call — a trivially-easier rewriter case that does **not**
    represent training. Histories are seeded from the tensors so the scales are realistic non-unit live
    scalars.
  - *Realistic backward cotangent:* a `sum` loss gives an all-ones cotangent, and `dequant(quant(1.0))`
    is an identity XLA folds — which would erase the E5M2 output-grad QDQ and stop a bwd f8 matmul from
    lowering. A random cotangent keeps it non-trivial.
- **Command (CPU verify):** `uv run python lib/levanter/scripts/bench/bench_dense_fp8.py --m 256 --k 256
  --n 256 --steps 3 --warmup 1 --path qdq` (+ a jaxpr probe).
- **Result (CPU):** both paths run fwd+bwd; `result_json` gains `path` + `peak_tflops_per_s`. jaxpr is
  recipe-faithful (`float8_e4m3` ×2 + `float8_e5m2` ×1); with the op as a runtime arg the fwd jaxpr
  carries the state inputs and computes the scale **in-graph** (`reduce_max` over the history) — i.e.
  the scales are live, not folded. `./infra/pre-commit.py` green. CPU can't render the `$f8` verdict
  (`__cublas$lt$matmul$f8` is a GPU-only custom-call) — that's H100.
- **Next action:** run `--path bf16` (baseline) and `--path qdq` at d3072/d4096 on H100; grep for
  `__cublas$lt$matmul$f8` in the fwd and bwd HLO separately.

### 2026-06-22 — GFP8-004: S2 H100 result — QDQ *forward* does NOT fire `$f8` under delayed scaling
- **Hypotheses (Matt):** (1) the QDQ path fails to emit `__cublas$lt$matmul$f8` in the real delayed-
  scaling setting; (2) it is then ≤ bf16 (bf16 matmul + QDQ overhead). **Confirmed for the forward.**
- **Jobs (1×H100, `CudaDevice(id=0)`, M=4096/seq 4096, steps 20/warmup 5, random cotangent):** bf16
  baseline + qdq at d3072/d4096 (`/matt/grug-s2-fp8-qdq-20260622-124154`); qdq with live delayed-scaling
  scales (`…-qdq-livescale-…125923`), re-captured `grep`-filtered to dodge the `iris job logs` ~1000-line
  tail cap (`…-qdq-live2-…130401`, `…-qdq-live3-…130651`).
- **Result — GEMM lowering (live delayed-scaling scales; every `custom_call_target` captured):**
  | shape | forward | backward dx/dw |
  |-------|---------|----------------|
  | d3072 | bf16 Triton `gemm_fusion_dot_general` (dequant operands) — **no `$f8`** | `__cublas$lt$matmul$f8` ×2 (live `%loop_convert_fusion` scales) |
  | d4096 | bf16 `__cublas$lt$matmul` (dequant `%loop_multiply_fusion` operands) — **no `$f8`** | `__cublas$lt$matmul$f8` ×2 |
  Both forwards are **bf16 fallbacks** (no FP8); they differ only in XLA's GEMM-backend autotune — Triton
  fusion at d3072 vs cuBLASLt at d4096 (job `…-fwdhlo-150727`, forward-only full HLO). Not FP8-relevant.
- **Result — throughput (qdq %-peak vs FP8 1978.9; bf16 vs 989.5):**
  | shape | bf16 fwd | qdq fwd | bf16 bwd | qdq bwd |
  |-------|----------|---------|----------|---------|
  | d3072 | 410 | **357 (18.0%) −13%** | 523 | **557 (28.2%) +6%** |
  | d4096 | 531 | **454 (23.0%) −14%** | 633 | **725 (36.6%) +15%** |
  (TFLOP/s; bwd counts dx+dw = 2× fwd flops.)
- **Verdict:**
  1. **The existing QDQ *forward* does NOT lower to `$f8` under live delayed scaling** — bf16 fallback
     (d4096) or a plain non-cuBLAS dot (d3072), and **slower than bf16** (−13–14%). This is the
     previously-reported "fails to fire / slower today" — **R5 prior confirmed for the forward.**
  2. **The *backward* does fire `$f8`** (both dims, live scales folded into the call) and beats bf16
     (+6–15%). XLA matches the bwd dot shape with a runtime scale but not the fwd one.
  - *Sanity:* a control with the op **closed over the jit** (constant scales) flips the fwd to `$f8` —
    confirming the failure is specific to *live* delayed-scaling scales, the training-relevant case.
- **Interpretation (SUPERSEDED — see GFP8-005):** the original read here was a TN-layout / `kScaledDot`
  /`ScaledDotRewriter` version-drift story. Both the layout-causal claim and the "rewriter declined"
  claim were later **refuted** (primary-source research + a pass-level HLO dump + an independent Codex
  review). The defensible statement is narrower: *the QDQ forward is not captured by any FP8 rewrite
  before XLA's `float_normalization`/`simplify-fp-conversions` strips its f8 representation; the backward
  is captured and fires `$f8`.* Root-cause discriminator still open — tracked in GFP8-005. Practical
  takeaway unchanged: don't rely on the QDQ→f8 pattern for the forward as it stands.
- **Reproduced on the rewritten branch** (`234fddd23`, job `/matt/grug-s2-verify-20260622-150148`):
  identical lowering — qdq fwd no `$f8` (d3072 bf16 Triton gemm fusion, d4096 bf16 `$lt$matmul`), bwd `$f8` ×2
  both dims with live `%loop_convert_fusion` scales; perf within run-to-run variance (qdq fwd 341/451,
  bwd 539/712 TF/s; bf16 fwd 405/532, bwd 521/630).
- **Ops notes:** (1) `iris job logs` tail-caps at ~1000 lines → pipe the remote bench through `grep` for
  `result_json|=== |custom_call_target` to capture complete, untruncated HLO evidence. (2) iris CLI via
  `--project /Users/matt/projects/marin` (main venv's `iris[controller]`) while cwd stays the S2 worktree
  (bundle carries the qdq code); `KUBECONFIG=~/.kube/coreweave-iris-gpu` (`use coreweave` direnv target).
- **Next action:** root-cause the forward non-capture first (GFP8-005), then build the manual
  `dq(dot(q,q))` fix arm. Dropped from the earlier list: `jax.nn.scaled_dot_general` (path C — Blackwell-
  only mxfp8, dead on Hopper) and fast-accum (out of scope until `$f8` fires at all). Bar to clear:
  fire fwd `$f8` with live scales and beat the bf16 baseline.

### 2026-06-22 — GFP8-005: S2 root-cause — forward f8 stripped pre-rewrite; layout/"declined" claims refuted
- **Goal:** find *why* the QDQ forward never emits `$f8` (GFP8-004), to a primary-source standard. Three
  causal stories were tested and the first two **refuted**.
- **Refuted — TN-vs-NN layout (was the leading hypothesis):** cuBLASLt FP8 on Hopper is genuinely TN-only,
  but XLA's GPU `GemmRewriter` *inserts transposes* to canonicalize FP8 operands (OpenXLA PR #59515, Feb
  2023; JAX #22313 shows a plain NN `[1]×[0]` forward fuses to `$f8`). So TN-vs-NN is a **perf** concern
  (avoids an inserted transpose — what TransformerEngine precomputes via a column-major FP8 copy), **not**
  a correctness gate. If layout were the cause we'd see `$f8` + a transpose, not a pure bf16 GEMM with the
  f8 converts gone. (Primary-source research, 4 claims, 3-0 each.)
- **Refuted — target-independent simplification:** CPU HLO dump (`--xla-dump-dir`, `--m 256 --k 256 --n
  256 --path qdq --forward-only`): the forward module carries **2 `f8e4m3` converts in
  `before_optimizations` and still 2 in `cpu_after_optimizations`**, feeding the dot. Nothing
  target-independent removes them → the collapse is GPU-pipeline-specific.
- **Eliminated for the bench — operand rank:** the bench forward dot is 2-D (one contracting + one free
  dim), so the f8 rewriter's dimension-count precondition is satisfied. (Still live for the real model's
  3-D attention projections — separate concern.)
- **Decisive evidence — H100 pass-level dump** (`--xla_dump_hlo_pass_re=.*`, forward-only d4096, job
  `/matt/grug-s2-passdump`; analysis script `lib/levanter/scripts/bench/_s2_passdump.sh`): f8e4m3 count
  per pass on the forward module (`module_11249.jit__lambda`, 83 pass files):
  - `2` from pass `0000` through `0028`; `0` after pass `0029` **`simplify-fp-conversions`** (the
    `float_normalization` pipeline).
  - `__cublas$lt$matmul$f8` appears in **none** of the 83 files; a **bf16** `__cublas$lt$matmul` is
    created later at `0031` (post-layout-assignment / autotuner).
- **Independent review (Codex, gpt-5.5, read the source) — refined the deduction:** my step "the rewriter
  *inspected and declined* the forward" is **not** established by the forward dump. Proven only:
  *the forward f8 representation is destroyed at `simplify-fp-conversions` and no `$f8` survives.* Whether
  an FP8 rewrite ran-and-missed vs. never-reached-it is **unresolved** (unchanged passes may emit no dump,
  so a missing filename ≠ a pass that didn't run; `ScaledDotRewriter` is also a *separate* pass from the
  FP8 path of `GemmRewriter`). Codex also down-ranked precision as the discriminator (XLA emits E4M3²
  `DEFAULT` NN f8 dots — issue #17276; our backward accepts `HIGHEST`); stronger suspects: **scale
  representation** (state inits scales as `f32[1]`, not scalar `f32[]` — `quantization.py:163`), the exact
  syntactic dequant DAG, and the number of runtime-scaled operands.
- **Defensible conclusion:** *the forward QDQ graph is not captured by an FP8 rewrite before
  float-normalization eliminates its f8 representation; the backward is captured and fires `$f8`.* The
  fwd/bwd discriminator is open.
- **Next action:** a confirm/refute research agent is dispatched (claims C1 pass-order, C2
  captured-vs-never-seen, C3 true discriminator, C4 does manual `dq(dot(q,q))` fire on 0.10/H100; methods
  = XLA 0.10 source + two experiments: backward pass dump, forward-only factorial probe over
  precision×dtype×scale-count×scale-shape with direct `convert(f8)*scale` operands). Hold local
  experiments for its verdict; then implement the indicated fix (manual arm, or a one-line QDQ change if
  the discriminator is scale-shape/precision).

### 2026-06-23 — GFP8-006: S2 — research verdict + manual direct-f8 forward arm
- **Confirm/refute research verdict** (the agent dispatched in GFP8-005; 94 agents, 12 sources, 25 claims
  adversarially verified, 21 confirmed / 4 killed):
  - **H0 outcome CONFIRMED, mechanism partial:** the fwd→bf16 / bwd→`$f8` split matches a real,
    compiler-attributable regression — JAX #24051 documents 0.4.30 fusing QDQ→`$f8` vs 0.4.31+ splitting
    it into an FP32/bf16 dot + separate requant; Flax docs call the operand-QDQ pattern-match "brittle."
    Our "stripped before float-normalization" wording is consistent-but-unproven at the 0.10 tag.
  - **C3 — scale representation REFUTED 0-3:** `f32[1]` vs scalar `f32[]` is **not** the discriminator (no
    `IsScalar` bailout in the matcher). **Do not chase scale shape.** Precision-per-se also down-ranked
    (matcher DAG is precision-agnostic). Live suspects narrowed to: exact syntactic dequant DAG + number of
    runtime-scaled operands.
  - **C1 (0.10 pass order) and C2 (declined-vs-stripped) UNDETERMINED:** no source read `gpu_compiler.cc`
    at the 0.10 tag; C2 leans "never-seen/stripped" (bf16 matmul created at ~0031 *after* converts gone at
    ~0029 ⇒ consistent with `IsF8Type==false`, not the `TurnF8DotIntoF16Dot` decline branch). Matcher locus
    on `main`: `gemm_rewriter.cc` `MatchFp8Param` → `MultiplyAnyOrder(Convert(fp8_input), Broadcast(scale))`,
    `num_dequant_ops=2` (note: 0.10 path is `xla/service/gpu/gemm_rewriter.cc`, not `xla/backends/...`).
  - **C4 (direct-f8 fix) — research over-reached; corrected by reading the threads.** The agent claimed the
    `quantize → dot → dequant` form is "Flax's endorsed migration"; **it is not.** Reading JAX #24051 and
    XLA #17887 directly: neither recommends q-dot-dq. The form Flax/haliax *ship* is the operand-QDQ that
    regressed. The **documented** workarounds are epilogue nudges — a `relu` before scaling, and/or an
    **output abs-max capture** (`jnp.max(jnp.abs(out))`) — which the reporter says bring the fusion back.
    q-dot-dq is **our** hypothesis, unproven on 0.10/H100. Both issues read open/dormant, no maintainer fix.
  - **Strong secondary lead from #24051 — missing forward output capture.** TE-style delayed scaling
    captures the forward *output's* abs-max each step; our bench forward's `out_qdq` is **identity** (no
    output capture/requant). So a live alternative discriminator: *the forward fails for lack of the output
    abs-max-capture/requant epilogue the rewriter keys on.* Cheap second arm if the manual arm misses.
- **Bottom line:** web research nailed the *what* (real 0.4.31+ XLA QDQ-pattern regression) but cannot pin
  the 0.10-tag line refs or run the decisive experiments — those are exactly C1/C2/C3/C4.
- **Built — `--path manual` (forward-only), commit `3e58f5a55`:** the candidate fix as a bench arm. Feeds
  genuine E4M3 operands straight into `dot_general` (`dot(f8,f8)->f32`, `preferred_element_type=f32`) and
  dequantizes only the f32 accumulator by `x_scale * w_scale` — so **no operand f8→bf16 round-trip exists**
  for `simplify-fp-conversions` to strip. Scales computed live from the same delayed-scaling histories as
  qdq (apples-to-apples). Scoped forward-only on purpose: the whole open question is forward emission; the
  e5m2 backward is deferred until the forward is shown to fire.
- **CPU verification:** unoptimized StableHLO shows `dot_general` consuming `f8E4M3FN` operands directly
  (`(f8E4M3FN, f8E4M3FN) -> f32`), contrast qdq's bf16 operands. Numerics match qdq (rel err vs an fp32
  reference: manual 0.0367, qdq 0.0370). On CPU the operands upcast to f32 (no f8 matmul there) — the f8
  emission question is H100-only.
- **Next action:** run `--path manual --forward-only` on H100. **Measure throughput + the operand dtype
  feeding whatever GEMM XLA picks — not only a `$f8` grep.** There is no pattern-match-free f8 on
  Hopper-via-XLA (the rewriter or a Triton f8 fusion must fire; else f8 operands upcast to bf16 at the dot,
  correct but slow), and a Triton **f8** fusion would be a win the `$f8` grep alone misses (d3072's forward
  already lands in a Triton fusion — a bf16 one, GFP8-004). Decision tree:
  - f8 tensor-core GEMM (cuBLASLt `$f8` *or* Triton f8 fusion), TFLOP/s > bf16 ⇒ fix works; promote to a
    real haliax op with the e5m2 backward, wire into grug dense.
  - upcast to bf16 ⇒ direct-f8 alone is insufficient. Then test the **abs-max-capture arm** (add forward
    output abs-max capture/requant, per #24051's documented workaround) before escalating to the C1/C2
    0.10-source read (`gpu_compiler.cc` pass order + `gemm_rewriter.cc` matcher) and the factorial probe
    over the surviving suspects (dequant-DAG shape, # runtime-scaled operands, epilogue consumer).
  - Command: `uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 --enable-extra-resources --extra
    gpu -- python lib/levanter/scripts/bench/bench_dense_fp8.py --k 4096 --n 4096 --path manual
    --forward-only` (read `fwd_tflops_per_s` vs the bf16 baseline; inspect the forward HLO for the GEMM's
    operand dtype; no storage needed).

### 2026-06-23 — GFP8-007: S2 H100 result — manual direct-f8 forward FIRES `$f8` (no speedup at d4096)
- **Hypothesis:** the manual arm `dq(dot(q(x), q(w)))` (genuine E4M3 operands into the dot, dequant on the
  output) fires `__cublas$lt$matmul$f8` on the forward, where operand-QDQ (transient bf16→f8→bf16
  round-trip, stripped at `simplify-fp-conversions`) does not.
- **Command:** iris `cw-us-east-02a` H100x1, `bash lib/levanter/scripts/bench/_s2_manual_validate.sh`
  (manual `--forward-only` + bf16 baseline, d4096). NOTE: finelog log server was down (`StatsError`,
  cluster Workers 0/0) — output retrieved via a keep-alive pod + `iris task exec` (kubectl exec bypasses
  finelog).
- **Result — CONFIRMED, forward fires f8:**
  - `__cublas$lt$matmul$f8` present (1 match); `f8e4m3` **survives** into the optimized module (8
    occurrences) vs qdq's **0** (stripped at pass 0029).
  - HLO: `%cublas-gemm.1 = (f32[4096,4096], s8[...]) custom-call(%input_transpose_fusion.1, ...,
    %loop_multiply_fusion, %constant), custom_call_target="__cublas$lt$matmul$f8"`, `scale_mode:1` (live
    delayed-scaling scales as runtime operands), `operand_precision=["DEFAULT","DEFAULT"]`.
  - XLA inserts an f8 transpose (`%input_transpose_fusion.1 = f8e4m3fn[4096,4096]`) to satisfy cuBLASLt
    TN-only — confirms PR #59515 (perf cost, not correctness gate).
- **Result — throughput: NO speedup at d4096 (exploratory, single shape):**
  - manual: **543.8 TFLOP/s** (252.7 µs/step, 27.5% of f8 peak 1978.9).
  - bf16:   **533.8 TFLOP/s** (257.5 µs/step, 53.9% of bf16 peak 989.5).
  - Essentially tied (~2%). The f8 GEMM fires but per-call quantize (x *and* w) + the inserted TN-transpose
    (memory-bound bf16/f32) cancel the f8 compute win; the f8 path runs at only 27.5% of its peak.
- **Interpretation:** necessary milestone reached — the direct-f8 form is the structural fix that unblocks
  the forward rewrite (validates the GFP8-005/006 transient-round-trip-stripping diagnosis). But the 2× is
  **not free**: at d4096 the QDQ + transpose overhead ≈ the GEMM savings. Realizing it needs: amortize
  weight quantization (quantize once, not per forward), avoid the transpose (store f8 in TN layout / TE
  column-major copy), fuse quantize into producers, and/or larger compute-bound shapes.
- **Next action:** (a) shape sweep (larger M; d3072) to find where f8 wins; (b) measure with weight
  quantized once (closer to training); (c) read the deep-research verdict on what TE-JAX/MaxText do —
  likely cuBLASLt custom calls that avoid this overhead; (d) decide whether the XLA-fusion path can beat
  bf16 here or whether TE-JAX/Pallas is required. Ops: report the finelog `StatsError` to infra.

### 2026-06-23 — GFP8-008: S2 H100 — TN layout removes the transpose but does NOT change timing
- **Hypothesis:** the d4096 manual f8 forward ties bf16 (GFP8-007) because of the XLA-inserted TN
  transpose; storing the weight `[N,K]` (TN, K minor on both — mirrors haliax `Linear`) should drop the
  transpose and recover the f8 win.
- **Change:** `--layout {tn,nn}` (default tn) in `bench_dense_fp8.py` (GFP8-008 commit). A/B script
  `_s2_tn_vs_nn.sh`, job `/matt/grug-s2-tn-nn2`, d4096 forward-only.
- **Result:**

  | arm | layout | `$f8` | `input_transpose_fusion` | TFLOP/s | µs |
  |-----|--------|-------|--------------------------|---------|-----|
  | manual | tn | 1 | **0** | 544.5 | 252.4 |
  | manual | nn | 1 | **4** | 546.7 | 251.4 |
  | bf16   | tn | 0 | 0 | 534.1 | 257.3 |

  - TN **removes the transpose** (`input_transpose_fusion` 0 vs 4); cuBLASLt now consumes the quantized
    operands directly (`custom-call(%loop_convert_fusion.1, %loop_convert_fusion, ...)`) vs NN's
    `custom-call(%input_transpose_fusion.1, ...)`.
  - **Timing is unchanged**: TN ≈ NN ≈ bf16 (~252 µs). Removing the transpose changed nothing.
- **Interpretation (refines GFP8-007):** the transpose is **not** the bottleneck — XLA fuses/overlaps it,
  so it is effectively free. The entire f8 overhead is the **per-call quantize** of *both* operands
  (divide-by-scale, clip, convert-to-f8) + output dequant + scale reductions — memory-bound work that is
  identical under TN and NN and ≈ the f8 GEMM's compute savings, so f8 ties bf16 either way. GFP8-007's
  "+ transpose" attribution was wrong; it's quantize-bound.
- **Next action — find the real f8 win:**
  1. **Amortize weight quant** (biggest realistic lever): quantize `w` once (pre-quantized f8 input),
     quantize only `x` per call — mirrors training, where weights are quantized once per step.
  2. **Isolate the GEMM:** time `dq(dot(f8,f8))` on *pre-quantized* f8 inputs (no in-loop quantize) vs the
     bf16 GEMM — confirms the f8 GEMM itself is ~2× and the overhead is quantize.
  3. **Larger M sweep** (M=8192/16384): make the GEMM more compute-bound vs the O(MK+NK) quantize.

### 2026-06-23 — GFP8-009: S2 — pass ordering pinned: f8 GemmRewriter runs BEFORE simplify-fp-conversions
- **Question (the C1/C2 left open in GFP8-005):** does the f8 GemmRewriter run *before* `simplify-fp-conversions`
  (Story A: rewriter declines the forward, leaving orphans the strip removes) or *after* (Story B: the strip
  folds the forward's f8 first, so the rewriter never sees it)?
- **Method:** full `--xla_dump_hlo_pass_re=.*` dump of `--path qdq --layout nn` **fwd+bwd** at d4096; per
  f8-bearing module, the ordered pass schedule + a verdict comparing the `$f8`-creation index vs
  `simplify-fp-conversions`. Script `lib/levanter/scripts/bench/_s2_pass_order.sh`, job
  `/matt/grug-s2-pass-order`.
- **Result — Story A, definitively:**
  - **Backward** (`module_45633`, fires `$f8`): `$f8` created at `0029...after_cublas-gemm-rewriter`
    (dump index 30); `simplify-fp-conversions` runs **later** at `0034` (index 35). f8e4m3 count stays **3
    across the strip** — the operands are folded into the `$f8` call, so the strip can't touch them.
  - **Forward** (`module_11249`, never fires `$f8`): f8e4m3 = 2 through 0028, **stripped to 0 at
    `0029...after_simplify-fp-conversions`**; `$f8` created at **no** pass.
- **Mechanism (now fully pinned):** the cuBLAS f8 GemmRewriter runs first and **claims the backward**
  (`e5m2×e4m3` grad dots → `$f8`, operands folded in → survive the later strip) but **declines the forward**
  (`e4m3²` dot → no `$f8`, no module change → orphaned f8 round-trip → `simplify-fp-conversions` removes it).
  `simplify-fp-conversions` is **cleanup of orphans**, not the decider — the decider is the rewriter's matcher.
- **Observability note:** the forward's declined-rewriter pass produced **no dump** (no change → no file), as
  predicted; the backward (identical pass schedule) is what reveals the rewriter's position. Also corrects the
  earlier red herring (GFP8-005/discussion): the bf16 cuBLAS at forward-pass 0031 is a *separate, later*
  rewrite on the already-bf16 dot, not the f8 rewriter.
- **Resolves:** C1 (pass order) and C2 (declined-vs-never-seen) → **declined** (Story A). Still open: *why* the
  matcher claims `e5m2×e4m3`/HIGHEST but declines `e4m3²`/DEFAULT for the QDQ round-trip — a matcher rule
  (dtype/precision/DAG shape), not an ordering artifact. The manual fix sidesteps it: resident f8 is matched
  even at `e4m3²`/DEFAULT (GFP8-007/008).

### 2026-06-23 — GFP8-010: S2 — PRECISION is the fwd/bwd gate; one-line fix for the existing op
> **[corrected by GFP8-012]** Two claims below are overstated: (a) "precision **is the** gate" — precision
> gates only the *transient* operand-QDQ round-trip; materialized f8 fires at DEFAULT too. (b) "**one-line**
> fix" — the training forward value comes from the `custom_jvp` primal recompute (`:146`), not `:133`, so the
> fix needs **both** lines; and the within-job +38% is at only 31% of f8 peak (possible partial fusion). See GFP8-012.
- **Hypothesis (the open "why" from GFP8-009):** does forcing the `e4m3×e4m3` operand-QDQ forward to
  `precision=HIGHEST` make it fire `$f8` (like the backward grad dots), where `DEFAULT` declines?
- **Method:** new `--path qdq_prec` = `in_qdq` operands + `lax.dot_general(precision=...)` (haliax's
  `dot_general_with_precision` hardwires DEFAULT, so this rebuilds it at an explicit precision). d4096
  forward-only; script `_s2_precision.sh`; job `/matt/grug-s2-precision`.
- **Result — precision IS the gate:**

  | layout | precision | `$f8` | f8e4m3 | GEMM | TFLOP/s | µs |
  |--------|-----------|-------|--------|------|---------|-----|
  | nn | default | **0** | 0 | bf16 `__cublas$lt$matmul` | 442.6 | 310 |
  | nn | highest | **1** | 7 | `__cublas$lt$matmul$f8` | **610.9** | 225 |
  | tn | highest | **1** | 6 | `__cublas$lt$matmul$f8` | 590.2 | 233 |

  HIGHEST fires `$f8` (f8e4m3 survives); DEFAULT declines (stripped, bf16). The `e5m2` output-grad dtype is
  **not** what made the backward fire — precision is.
- **Root cause fully closed:** haliax `dot_general_with_precision` forces the **primal=DEFAULT**
  (`_src/fp8.py:133`) and **tangent=HIGHEST** (`:146-149`). That asymmetry (copied from Flax) is exactly why
  the forward declines and the backward fires. The f8 GemmRewriter (which runs before the strip, GFP8-009)
  only claims HIGHEST-precision f8 dots here.
- **Implication — one-line fix, no rewrite:** `qdq_prec` is an exact proxy for "the existing
  `Fp8DotGeneralOp` with its forward primal at HIGHEST." So flipping that one `DEFAULT`→`HIGHEST` makes the
  **existing** operand-QDQ op fire the forward — the manual custom_vjp arm (GFP8-006/007) is **not required**.
  Bonus: operand-QDQ folds the scales into the cuBLASLt epilogue (no explicit output rescale like manual),
  so within-job HIGHEST (610) beats the DEFAULT control (442) by +38% and beats the bf16 baseline (~534,
  cross-job) — a real speedup, unlike the manual arm's d4096 tie (GFP8-007/008).
- **Caveats / next:** (1) replicate the bf16-beating with a same-job bf16 baseline (cross-job here);
  (2) confirm flipping the haliax primal precision doesn't regress other call sites/numerics; (3) verify the
  full fwd+bwd op (primal HIGHEST) still fires `$f8` on both. Then the dense fix is a 1-line haliax change.
- (Ops: the job exited non-zero — a cosmetic script-tail artifact; all three cells produced complete data.)

### 2026-06-23 — GFP8-011: deep-research — provenance + landscape (operand-QDQ is DEPRECATED upstream)
- **Deep-research workflow** (102 agents, 86 claims → 25 adversarially verified, 25 confirmed / 0 killed; full
  report in job `wmkzpm1f8`). Verified findings:
  - **Provenance (high, 3-0):** operand-QDQ (`in_qdq`/`out_qdq`, `dot(dq(q(x)), dq(q(w)))` @ DEFAULT)
    originated in **Flax `flax/linen/fp8_ops.py`, PR #3322** (wenscarl/shuw, commit `b309c8f25b` 2023-09-11,
    merged 2023-09-27), deliberately targeting XLA's GemmRewriter (Philipp Hack, TF PR #58720, 2022-12-16,
    RFC openxla/xla#22). **Haliax `Fp8DotGeneralOp` is a documented copy.** (Answers David's "came from Flax?"
    — yes.)
  - **Flax DEPRECATED it (high, 3-0):** `Fp8DotGeneralOp.__post_init__` emits a DeprecationWarning; Flax docs
    call the XLA pattern-matching **"brittle, as the patterns could be easily broken by other XLA
    optimizations."** Replaced by explicit ops (`in_q`/`out_dq`, `Fp8DotGeneral`/`Fp8DirectDotGeneralOp`,
    `Fp8Einsum`) routing through `fp8_scaled_dot_general` with **materialized f8 operands**.
  - **Production stacks bypass XLA fusion (high, 3-0):** NVIDIA TransformerEngine-JAX routes fwd + both bwd
    GEMMs through its own cuBLASLt **FFI custom call** (`te_gemm_v2_ffi` / `nvte_cublas_gemm_v2`), NOT
    QDQ-around-dot. AQT (google/aqt) is `dot_general`-level and INT8-centric.
  - **Load-bearing distinction (high, 2-1):** the canonical XLA-targeted pattern uses **materialized f8
    leaves** (cast up to FP16, scaled, dot on the wide type) — persistent-leaf — vs Flax/Haliax's **transient
    bf16→f8→bf16 round-trip** that `simplify-fp-conversions` can strip.
  - **Regression boundary:** $f8 fusion broke at **JAX 0.4.30→0.4.31** (0.4.30 fires; 0.4.31+ falls to
    FP32/bf16). Output-requant non-fusion (#22313) is a *separate* earlier phenomenon.
- **Reconciliation — we are ahead of the survey on mechanism.** The research's materialized-vs-transient axis
  matches GFP8-007 (materialized/manual fires) vs GFP8-004 (transient qdq-fwd fails), but we proved MORE:
  GFP8-009/010 show the actual gate is **precision** — transient+DEFAULT declines, transient+**HIGHEST fires**,
  and materialized+DEFAULT also fires. So on 0.10 there are **two** independent ways to make the rewriter
  claim it (materialize the f8, OR use HIGHEST); the sole failing case is transient+DEFAULT — exactly haliax's
  forward primal.
- **Strategic implication (Phase-1 decision-gate material):** the one-line precision flip (GFP8-010) is a real
  short-term win but patches a mechanism Flax has **officially deprecated as brittle** — the very path that
  broke at 0.4.31. Durable directions: (a) **materialized-f8 explicit op** (Flax's non-deprecated
  `fp8_scaled_dot_general`; our manual arm is in this family) or (b) **TE-JAX cuBLASLt FFI** (production
  reference; heavier integration). If we ship the precision flip, treat it as interim + pin jaxlib + add the
  `$f8`-in-HLO regression test.
- **Still open (research could not resolve):** does `jax.nn.scaled_dot_general` support delayed scaling on
  Hopper or is it Blackwell-mxfp8-only (central to path C); FP8 mechanism in MaxText/Qwix/Praxis (uncovered).
- **Sources:** Flax PR #3322; `flax/linen/fp8_ops.py`; flax fp8_basics docs; JAX #24051, #22313; TF #58720;
  openxla/xla#22; NVIDIA/TransformerEngine (`te_gemm_v2_ffi`); google/aqt; NVIDIA/atex xla-fp8 tutorial.

### 2026-06-23 — GFP8-012: S2 — independent-review audit corrects GFP8-010 (gate wording + the fix is 2 lines)
- **Why:** sanity-checked the GFP8-010 conclusions with two independent agents (codex; opencode/GLM-5.2),
  each given the same neutrally-framed evidence brief + read access to `fp8.py`/`bench_dense_fp8.py`. Both
  independently returned **"partially holds."** Then verified the load-bearing catch on CPU.
- **Correction 1 — "precision is THE gate" overclaims.** Precision gates only the **transient**
  bf16→f8→bf16 operand-QDQ pattern; the **materialized**-f8 manual arm (GFP8-007) fires `$f8` at DEFAULT.
  Defensible restatement: *for the transient operand-QDQ round-trip, `precision=HIGHEST` is **necessary** to
  re-fuse the forward into `$f8`; at DEFAULT the round-trip is stripped → bf16. For materialized f8, precision
  is irrelevant.* (This is the "two independent triggers" point from GFP8-011, stated correctly.)
- **Correction 2 — the fix is NOT one line (verified).** `dot_general_with_precision` is a `custom_jvp`; under
  `value_and_grad` (Levanter training) the forward **value** is the jvp rule's primal **recompute** at
  `_src/fp8.py:146` (DEFAULT), **not** the standalone primal at `:133`. CPU jaxpr check
  (`value_and_grad(sum(Fp8DotGeneralOp(x,w)))`) shows the training trace's three dots =
  `[(DEFAULT,DEFAULT), (HIGHEST,HIGHEST), (HIGHEST,HIGHEST)]` — the lone DEFAULT is `:146`; `:133` is not even
  traced. So flipping only `:133` would green the **forward-only microbench yet do nothing in training.** The
  dense fix must flip **both `:146` (training fwd value) and `:133` (eval/inference fwd).**
- **Correction 3 — "numerically free" is unproven, and the win may be partial.** (a) On the **fallback** path
  (fusion declines), HIGHEST on bf16 operands is untested vs DEFAULT. (b) GFP8-010's +38% within-job is only
  **610/1979 ≈ 31% of f8 peak**, and f8/bf16 ≈ **1.38× not ~2×** — consistent with *partial* fusion (residual
  `convert` prologue not absorbed) or under-settled autotune (warmup=5). `$f8`-in-HLO is **necessary, not
  sufficient**, for "fully fused f8 tensor-core gemm."
- **Confound found:** GFP8-007-vs-010 ("materialized fires at DEFAULT") varies *two* things — `manual_fp8_dot`
  sets `preferred_element_type=f32`, `qdq_prec_dot` doesn't. Clean attribution needs manual-without-PET (or
  qdq_prec-with-PET).
- **Framing nit (GLM):** "undocumented heuristic" overstates — `precision` lowers to `precision_config`, a
  *documented* HLO field XLA may legitimately key gemm-algorithm choice on; the JAX-level docstring is just
  silent. Brittle/version-fragile stands (the f8-refusion policy is not a stable contract).
- **Mechanism (still undetermined) — cheap disambiguator:** `--path bf16 --precision default` vs `highest`
  (pure bf16, no QDQ). HIGHEST changes the bf16 lowering/throughput ⇒ bf16 rewriter preempts at DEFAULT (H1);
  identical ⇒ f8 rewriter independently declines DEFAULT (H2). Also measures the fallback-path risk above.
- **Actions:** added the `:133`+`:146` precision toggle behind an explicit `Fp8DotGeneralOp.forward_precision`
  field (default unchanged) + an HLO-`$f8` regression test (both reviewers recommended the canary). H100 batch
  queued: bf16 precision A/B (mechanism), profiler/HLO convert-check on the 1.38× (is it fully fused?), and the
  `preferred_element_type` de-confound.

### 2026-06-23 — GFP8-013: S2 H100 — fix FIRES the fwd $f8, but the real win over bf16 is ~+7% at d4096
- **Job:** `/matt/grug-s2-gfp8012-batch` (H100x1, d4096/TN, single-run). Validates the GFP8-012 fix end-to-end
  and closes the audit's open questions. Numbers are one-shot (no error bars).

  | arm | path / precision | `$f8` | fwd TFLOP/s | note |
  |-----|------------------|-------|-------------|------|
  | A3 | bf16 (real)              | 0 | **561** | true bf16 baseline (57% of bf16 peak) |
  | A1 | qdq / default (fwd)      | 0 | 459 | legacy fwd fallback: bf16 gemm **+ dead QDQ converts** |
  | A2 | qdq / **highest** (fwd)  | **1** | **602** | **THE FIX: real Fp8DotGeneralOp fwd fires `$f8`** (30% of f8 peak) |
  | C1 | qdq / highest, long warmup | 1 | 618 | +autotune settled → still ~31% of peak (not an autotune artifact) |
  | D1 | manual, PET=f32          | 1 | 528 | materialized f8 |
  | D2 | manual, PET=none         | 1 | 560 | materialized f8 |
  | B1 | bf16 / default           | 0 | 518 | `__cublas$lt$matmul` |
  | B2 | bf16 / highest           | 0 | 551 | `__cublas$lt$matmul` (same target) |

  fwd+bwd: A4 (qdq/highest) fwd 600 / bwd 665 / `$f8`=3; A5 (qdq/default) fwd 457 / bwd 723 / `$f8`=2. The bwd
  fires `$f8` in both (2 grad dots, always HIGHEST); the fix adds the +1 forward `$f8` (3 vs 2).
- **Fix validated (high):** the real `Fp8DotGeneralOp` at `forward_precision="highest"` emits
  `__cublas$lt$matmul$f8` on the forward (A2/C1/A4) — operands are genuine `f8e4m3fn[4096,4096]` leaves with
  `operand_precision:["HIGHEST","HIGHEST"]`, `scale_mode:1` (scaled f8 gemm, bf16 out). Backward unchanged.
- **CORRECTS GFP8-010's "+38% / beats bf16" claim.** The +38% was qdq/highest (602) vs the qdq/**DEFAULT
  fallback** (459) — but that baseline is *slower than real bf16* (561) because it still pays the QDQ convert
  round-trip then runs bf16. Against a **true bf16 dot the fwd f8 win is only ~+7%** (602 vs 561), at ~30% of
  f8 peak.
- **Why so small — quantization-convert overhead, NOT partial fusion (refines GLM's concern).** The HLO shows
  the f8 gemm consuming f8 operands *directly*; the QDQ is fully absorbed into f8 leaves. The cost is the **two
  standalone bf16→f8 `loop_convert_fusion` kernels** (one per operand, each 4096²) that quantize the operands
  and are **not amortized** by the d4096 gemm. Longer warmup (618) rules out autotune. This matches GFP8-007/008
  (manual f8 ties bf16 at d4096): the per-call per-tensor quantization is the ceiling, not the gemm.
- **Mechanism — H2 favored (medium):** on pure bf16, DEFAULT vs HIGHEST both lower to `__cublas$lt$matmul`
  (B1≈B2, 518 vs 551 = noise, same target) — precision does **not** change bf16 gemm selection. So the
  DEFAULT-vs-HIGHEST `$f8` difference is the f8 GemmRewriter's matcher independently keying on the dot's
  `precision_config` and declining DEFAULT (not the bf16 rewriter preempting).
- **PET de-confound — resolved clean (high):** manual fires `$f8` at DEFAULT with PET=f32 *and* PET=none
  (D1=D2=1). So materialization alone (genuine f8 operands) is what fires obs-4; `preferred_element_type` is
  irrelevant. GFP8-012 Correction-1's confound is closed.
- **Phase-1 implication:** the precision flip is **correct and confirmed** (forward `$f8` fires on the real op),
  but at d4096 it buys only ~+7% over bf16 — the headline MFU win must come from **amortizing the quantization**
  (larger GEMMs; keeping activations resident in f8 across consecutive matmuls / output-requant; or fusing the
  convert into the gemm prologue) rather than from merely making `$f8` appear. Re-measure at the real MoE expert
  GEMM shapes before sizing the win.
