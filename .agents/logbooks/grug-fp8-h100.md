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
  | d3072 | **no cuBLAS matmul** (plain `dot`/fusion) — **no `$f8`** | `__cublas$lt$matmul$f8` ×2 (live `%loop_convert_fusion` scales) |
  | d4096 | `__cublas$lt$matmul` (bf16, `%loop_multiply_fusion` operands) — **no `$f8`** | `__cublas$lt$matmul$f8` ×2 |
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
- **Interpretation:** the fwd is one DEFAULT-precision dot fed by `in_qdq` (dequant×live-scale) on both
  operands; XLA's current `kScaledDot`/`ScaledDotRewriter` doesn't match that shape with a runtime
  scale (the version-drift fragility R5 flagged). Don't rely on the QDQ→f8 pattern for the forward.
- **Reproduced on the rewritten branch** (`234fddd23`, job `/matt/grug-s2-verify-20260622-150148`):
  identical lowering — qdq fwd no `$f8` (d3072 no cuBLAS matmul, d4096 bf16 `$lt$matmul`), bwd `$f8` ×2
  both dims with live `%loop_convert_fusion` scales; perf within run-to-run variance (qdq fwd 341/451,
  bwd 539/712 TF/s; bf16 fwd 405/532, bwd 521/630).
- **Ops notes:** (1) `iris job logs` tail-caps at ~1000 lines → pipe the remote bench through `grep` for
  `result_json|=== |custom_call_target` to capture complete, untruncated HLO evidence. (2) iris CLI via
  `--project /Users/matt/projects/marin` (main venv's `iris[controller]`) while cwd stays the S2 worktree
  (bundle carries the qdq code); `KUBECONFIG=~/.kube/coreweave-iris-gpu` (`use coreweave` direnv target).
- **Next action:** build the fix arms and re-measure fwd `$f8` under **live** scaling — (i) David's manual
  `dq(op(q,q))` (genuine f8 operands + explicit scale → the kScaledDot shape directly), (ii)
  `jax.nn.scaled_dot_general` (path C), (iii) fast-accum. Bar to clear: fire fwd `$f8` with live scales
  and beat the bf16 baseline.
