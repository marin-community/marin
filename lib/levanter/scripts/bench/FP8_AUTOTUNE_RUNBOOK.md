# FP8 ragged-dot autotuning benchmark — runbook

The **metric harness** for the FP8 mixed-precision MoE ragged-dot optimization loop. It answers one
question defensibly: **how much faster is the best-tuned FP8 hybrid than the best-tuned bf16 baseline**,
on representative Grug MoE expert-MLP shapes, with a confidence interval rather than a single noisy mean.

Use this when iterating on the FP8 ragged-dot kernels (`haliax._src.fp8_ragged`,
`haliax._src.transposed_ragged_dot_mgpu`, `haliax.nn.ragged_dot`) — it is the number that decides
whether a change helped. For the fork/jaxlib rationale see [`MIXED_FP8_FORK.md`](MIXED_FP8_FORK.md);
for the genuine-mixed-fp8 correctness invariant see the `grug-fp8-ragged-mixed-required` memory.

## Files

| file | role |
|------|------|
| `bench_ragged_fp8_autotune.py` | single-GPU autotuner + `--worker` mode (one process, one GPU) |
| `orchestrate_fp8_autotune.py` | multi-GPU orchestrator: one worker subprocess per GPU, coordinate descent as parallel waves |
| `fp8_autotune_configs.py` | jax-free experiment design: shapes + block-config candidates + smem pruning (single source of truth) |
| `fp8_autotune_stats.py` | jax-free stats: bootstrap median CI, ratio-of-medians CI |
| `fp8_wheel_cache.py` | stash/fetch the forked jaxlib wheel in R2 so jobs skip the ~11 min build |
| `fp8_artifact_store.py` | shared R2/S3 plumbing + generic `cp` (ships profile-trace tarballs local<->cluster) |
| `mixed_fp8_fork_setup.sh` | builds/installs the forked jaxlib + jax overlay + Mosaic toolchain in the job container |
| `../../tests/test_bench_ragged_fp8_autotune.py` | jax-free unit tests (stats, smem pruning, orchestrator planning) |

## The metric

Headline per shape = **median(best-tuned bf16 fwd+bwd time) / median(best-tuned fp8 fwd+bwd time)** =
fp8 speedup over bf16, reported with a bootstrap 95% CI. Both approaches are tuned (best-vs-best):

- **fp8**: coordinate descent over the shared forward/dlhs `MosaicBlockConfig`, then the independent
  `WgradBlockConfig`, scored on the true e2e fwd+bwd time. Numerics-gated vs bf16 (recipe correctness
  is block-invariant, so checked once per shape + reconfirmed on the winner).
- **bf16**: swept over the Triton `RAGGED_DOT_BLOCK_*` / warps / stages space.

Config injection uses **zero production-code change**: `fp8_ragged` imports the Mosaic kernels by name,
so the harness swaps those module attributes for candidate-bound wrappers. `jax.clear_caches()` runs
before every compile — jit caches a trace by `(fn, avals)` and would otherwise serve the *previous*
candidate's kernel after the attribute swap (silently timing one config for the whole sweep).

Shapes (`fp8_autotune_configs.SHAPE_GRID`), spanning the experts/raggedness axis:

| bucket | D (hidden) | F (intermediate) | E (experts) | T (tokens) | tok/expert |
|--------|-----------|------------------|-------------|------------|------------|
| small  | 512  | 256  | 8   | 4096  | 512  |
| target | 2048 | 5632 | 8   | 8192  | 1024 | ← GFP8-029 headline regime |
| scale  | 3072 | 1536 | 128 | 16384 | 128  |

## Local (CPU) — validate mechanics without a GPU

```bash
# jax-free unit tests (stats, smem pruning, orchestrator planning) — ~0.1s
uv run --no-sync --with pytest python -m pytest \
  lib/levanter/tests/test_bench_ragged_fp8_autotune.py -o addopts="" --noconftest -q

# harness orchestration + statistics on CPU (bf16/xla only; no mosaic, no real speedup)
JAX_PLATFORMS=cpu uv run --no-sync python lib/levanter/scripts/bench/bench_ragged_fp8_autotune.py --smoke

# orchestrator multi-process plumbing on CPU (bf16-only, 2 simulated workers)
JAX_PLATFORMS=cpu uv run --no-sync python lib/levanter/scripts/bench/orchestrate_fp8_autotune.py \
  --simulate --shapes small --samples 6 --inner-steps 3 --warmup 1 --out-dir /tmp/sim
```

## Cluster (cw-us-east-02a H100) — the real numbers

Mosaic-GPU is H100-only and needs the forked jaxlib (mixed E4M3/E5M2 wgmma). Each GPU node is **8×H100**.

### One-time auth setup (per machine/worktree)

```bash
# 1. CoreWeave kubeconfig token from https://console.coreweave.com/tokens -> ~/.kube/coreweave-iris-gpu
export KUBECONFIG=~/.kube/coreweave-iris-gpu
kubectl cluster-info                        # sanity check

# 2. iris controller deps (NOT a root extra; install into the venv, then always run iris with --no-sync
#    so a uv re-sync doesn't wipe it):
uv pip install 'marin-iris[controller]'

# 3. (only if artifact/state ops fail) R2 creds, see lib/iris/docs/coreweave.md:
#    export R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
```

### Cached wheel (fast path — skip the ~11 min build)

The forked jaxlib build dominates job wall-clock (~11 min build vs ~1.5 min 8-GPU sweep). A prebuilt
wheel is stashed in the cluster R2 bucket; jobs `get` it and pass `JAXLIB_WHEEL=` so setup skips the
build, turning a run into ~3 min. Managed by `fp8_wheel_cache.py` (put/get/exists via s3fs + injected
R2 creds).

- **URI:** `s3://marin-na/marin/grug-fp8/wheels/jaxlib-mixfp8-0.10.0-cp312-cw.whl`
- **sha256:** `3b4f8a71efc0e4052262f291591428cf8cbef81fc25a23580962e75f71cb48a9` (85026111 bytes)
- Built from `mcwitt/jax@mixed-fp8-wgmma-0.10.0` for jaxlib 0.10.0 / cp312 / manylinux_2_27 x86_64.

Rebuild + re-stash when the fork or jaxlib version changes (one H100x1 job, `--disk 150GB`):

```bash
... -- bash -lc 'set -euo pipefail; nvidia-smi -L; \
  bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh; \
  uv run --no-sync python lib/levanter/scripts/bench/fp8_wheel_cache.py put "$(ls /root/jaxsrc/dist/jaxlib-*.whl | head -1)"'
# the FP8_WHEEL_PUT log line prints the new uri + sha256 — update this section.
```

### Submit — one 8-GPU job per shape (cached-wheel fast path)

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu
for SHAPE in target small scale; do
  uv run --no-sync iris --cluster=cw-us-east-02a job run \
    --gpu H100x8 --enable-extra-resources --extra gpu \
    --cpu 32 --memory 128GB --disk 64GB --no-wait \
    -- bash -lc "set -euo pipefail; nvidia-smi -L; \
       uv run --no-sync python lib/levanter/scripts/bench/fp8_wheel_cache.py get /tmp/wheels; \
       JAXLIB_WHEEL=\$(ls /tmp/wheels/*.whl | head -1) bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh; \
       uv run --no-sync python lib/levanter/scripts/bench/orchestrate_fp8_autotune.py \
         --shapes $SHAPE --num-gpus 8 --out-dir /app/scratch/fp8_sweep_$SHAPE --worker-timeout 900"
done
```

### Submit — from-scratch build per job (no cached wheel; needs `--disk 150GB`)

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu
for SHAPE in target small scale; do
  uv run --no-sync iris --cluster=cw-us-east-02a job run \
    --gpu H100x8 --enable-extra-resources --extra gpu \
    --cpu 32 --memory 128GB --disk 150GB --no-wait \
    -- bash -lc "set -euo pipefail; \
       bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh; \
       uv run --no-sync python lib/levanter/scripts/bench/orchestrate_fp8_autotune.py \
         --shapes $SHAPE --num-gpus 8 --out-dir /app/scratch/fp8_sweep_$SHAPE --worker-timeout 900"
done
```

Each job: forked-jaxlib build (~11 min) → 8-way sharded sweep for that shape (~5 min). The orchestrator
prints `result_json {...}` (the shape's headline + CI) to stdout, captured in the job logs. Within a job
the 8 GPUs parallelize that shape's ~43 config evals; the wave barriers stay in-process.

### Monitor + retrieve

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu
JOB=/matt/iris-run-job-YYYYMMDD-HHMMSS          # from the submit output

uv run --no-sync iris --cluster=cw-us-east-02a job list --json --prefix "$JOB"      # state
uv run --no-sync iris --cluster=cw-us-east-02a job summary --json "$JOB"            # exit code / postmortem
uv run --no-sync iris --cluster=cw-us-east-02a job logs --since-seconds 2400 "$JOB" \
  | grep -aE "result_json|>> HEADLINE|best (mosaic|wgrad|bf16)|numerics gate"
```

`result_json` / `--out-dir/summary.json` carries per-shape `bf16_best`, `fp8_best` (winning mosaic+wgrad
configs + median/CI/min + grad_rel_frob_vs_bf16), and `speedup_vs_bf16_best` (median + 95% CI). Raw
per-config rows are in `--out-dir/rows.jsonl` (skill-required machine-readable fields).

## Profiling capture (attribution, not a metric)

`--profile` answers *where* the fp8 fwd+bwd spends time (Mosaic GEMM kernels vs fp8 fixed overhead:
quantize/scale, cast-transpose, dequant) — which the headline speedup cannot. It writes jax profiler
traces for one shape at the default block config; analyze with the `profile-training` skill's
`lib/marin/tools/profile_summary.py`. It is **not** a timing metric (trace overhead perturbs absolute
times); use it only for the breakdown.

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu
uv run --no-sync iris --cluster=cw-us-east-02a job run \
  --gpu H100x1 --enable-extra-resources --extra gpu \
  --cpu 16 --memory 64GB --disk 64GB --no-wait \
  -- bash -lc "set -euo pipefail; nvidia-smi -L; \
     uv run --no-sync python lib/levanter/scripts/bench/fp8_wheel_cache.py get /tmp/wheels; \
     JAXLIB_WHEEL=\$(ls /tmp/wheels/*.whl | head -1) bash lib/levanter/scripts/bench/mixed_fp8_fork_setup.sh; \
     uv run --no-sync python lib/levanter/scripts/bench/bench_ragged_fp8_autotune.py \
       --profile --shapes d2560_e32_t1k --out-dir /app/scratch/fp8_profile --profile-steps 30; \
     for K in fp8 bf16; do \
       uv run --no-sync --with protobuf python lib/marin/tools/profile_summary.py summarize \
         --profile-dir /app/scratch/fp8_profile/profiler/\$K --output /tmp/sum_\$K.json || true; \
       echo \"===== TOP OPS \$K =====\"; \
       uv run --no-sync --with protobuf python lib/marin/tools/profile_summary.py query \
         --summary /tmp/sum_\$K.json --question 'What are the top 25 ops by exclusive time?' || true; \
     done; \
     tar czf /tmp/fp8_profile.tgz -C /app/scratch fp8_profile; \
     uv run --no-sync python lib/levanter/scripts/bench/fp8_artifact_store.py cp \
       /tmp/fp8_profile.tgz s3://marin-na/marin/grug-fp8/profiles/d2560_e32_t1k.tgz"
```

The top-ops tables print to the job logs (`job logs | grep -A30 'TOP OPS'`); the raw trace tarball is
stashed in R2 for richer local analysis (`fp8_artifact_store.py cp s3://.../d2560_e32_t1k.tgz /tmp/`).

## Gotchas (learned the hard way)

- **`--disk 150GB` is mandatory.** The default `disk=5GB` evicts the pod (exit 137, *"container could not
  be located"*) when the bazel jaxlib cache fills, ~99% into the build. Memory 128GB / cpu 32 are fine.
- **`uv run --no-sync iris`**, never `uv run iris` — a re-sync drops the manually-installed `[controller]`
  extra and bricks the CLI with `ImportError: Install iris[controller]`.
- **One iris tunnel at a time.** Concurrent `iris` calls fight over the controller port-forward; run a
  single poller, don't call `iris` in the foreground while a background monitor is polling.
- **Use the cached wheel** (see "Cached wheel" above) — rebuilding jaxlib per job is ~11 min of dead
  time; `fp8_wheel_cache.py get` + `JAXLIB_WHEEL=` skips it. This is what makes a per-change run ~3 min.
- The forked-jax setup is the `grug-fp8-fork` approach; `grug-fp8-shim` runs the same path on stock
  jaxlib via an import-time overlay (no build). Port wins between branches as needed.

## Tuning the run

| flag | default | when to change |
|------|---------|----------------|
| `--shapes` | `all` (orch) / `small,target,scale` (harness) | one shape per cluster job for fan-out |
| `--samples` | 40 | raise for a tighter headline CI; lower for a quick check |
| `--inner-steps` / `--warmup` | 10 / 5 | per-batch dispatch amortization / discarded warmup |
| `--mosaic-wgrad` | `fp8` | `bf16` for the simpler ~1.27× hybrid (skips the wgrad-stage sweep) |
| `--grad-dtype` | `e5m2` | `e4m3` for the all-E4M3 recipe (loses E5M2 gradient range) |
| `--numerics-tol` | 0.25 | max grad rel_frob vs bf16 to accept the fp8 path for a shape |
| `--max-reqs-per-worker` | 4 | configs per worker subprocess (startup vs balance trade-off) |
| `--worker-timeout` | 1200 | kill a hung worker (a stuck mosaic compile) and continue |

The candidate config space lives in `fp8_autotune_configs.py` (`_MOSAIC_RAW`, `_WGRAD_RAW`, `_BF16_RAW`),
curated around the GFP8-029 winner and smem-pruned. Edit there to widen/narrow the sweep — both the
single-GPU harness and the orchestrator read it, so they always sweep the same space.
