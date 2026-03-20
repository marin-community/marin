# Grug Muon Throughput: Research Logbook

## Scope
- Goal: Optimize Grug Muon optimizer-update throughput for a Grug MoE model matching the target W&B shape at approximately 24.16B parameters, without running forward/backward.
- Primary metric(s): Mean optimizer step time for `optimizer.update(...)` plus `optax.apply_updates(...)`; compile time as a secondary metric.
- Constraints: Focus on optimizer update only; use random but plausible gradients; bottleneck is expected to be Muon orthogonalization; prefer v5p-64 for final measurement, but start with local/v5p-8 microbench if needed.
- Requested target shape: `hidden_dim=4096`, `num_layers=27`, `E=64`, `EP=4`, `K=4`, routed expert width `1024`, shared expert width `1024`, `cf=1.25`.
- Code refs:
  - `experiments/grug/moe/model.py`
  - `lib/levanter/src/levanter/optim/grugmuon.py`
  - `lib/levanter/scripts/bench/bench_grug_muon.py`

## Baseline
- Date: 2026-03-19
- Fixed baseline case: Compare current replicated Grug Muon Newton-Schulz path against a sharding-preserving path on the same model shape and hardware.
- Baseline numbers: Pending first successful end-to-end optimizer microbench.
- Notes:
  - Current Grug MoE code path hard-codes `capacity_factor=1.25`, matching the requested target.
  - Rough parameter estimate for the requested shape is approximately 24.27B with `num_heads=32`, `num_kv_heads=8`, `head_dim=128`, `vocab_size=128256`.

## Experiment Log
### 2026-03-19 13:xx - Kickoff and harness setup
- Hypothesis: The dominant throughput bottleneck is the Grug Muon orthogonalization path, specifically full replication before Newton-Schulz on sharded TPU runs.
- Command:
```bash
sed -n '1,260p' lib/levanter/src/levanter/optim/grugmuon.py
sed -n '1,260p' experiments/grug/moe/model.py
sed -n '1,260p' docs/dev-guide/dev_tpu.md
sed -n '1,260p' scripts/ray/dev_tpu.py
```
- Config: Local code inspection only.
- Result:
  - Found Grug Muon in `lib/levanter/src/levanter/optim/grugmuon.py`.
  - Found Grug MoE model in `experiments/grug/moe/model.py`.
  - Confirmed dev TPU workflow via `scripts/ray/dev_tpu.py`.
- Interpretation: Existing Grug Muon code was the right optimization surface; dev TPU is the likely fast-iteration path before any larger run.
- Next action: Build a dedicated optimizer-only microbench that instantiates the Grug MoE model, creates optimizer state, synthesizes plausible gradients, and times update/apply.

### 2026-03-19 13:xx - Orthogonalization refactor for A/B benchmarking
- Hypothesis: Preserving useful sharding through Newton-Schulz should avoid unnecessary all-gathers and improve throughput relative to the legacy replicated path.
- Command:
```bash
uv run --package levanter --group test python -m pytest \
  lib/levanter/tests/test_grugmuon.py \
  lib/levanter/tests/test_optimizer_linear_like.py -q
```
- Config:
  - Added `_zeropower_via_newtonschulz_preserve_sharding(...)`
  - Kept `_zeropower_via_newtonschulz_replicated(...)` for A/B comparison
  - Added `lib/levanter/scripts/bench/bench_grug_muon.py`
  - Added focused unit coverage in `lib/levanter/tests/test_grugmuon.py`
- Result:
  - Tests passed: `8 passed`
- Interpretation: The basic optimizer refactor is mechanically sound and benchmarkable, but the optimizer microbench still needs an end-to-end sharding-clean update path.
- Next action: Run the microbench on a small local shape and identify any remaining update sharding mismatches.

### 2026-03-19 13:xx - Local microbench blocker: update sharding mismatch
- Hypothesis: A remaining explicit-mesh sharding mismatch is preventing `optax.apply_updates(...)` from accepting Muon-produced updates after the Newton-Schulz refactor.
- Command:
```bash
uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-heads 4 \
  --num-kv-heads 4 \
  --head-dim 32 \
  --num-experts 8 \
  --expert-axis-size 1 \
  --num-experts-per-token 2 \
  --routed-expert-width 64 \
  --shared-expert-width 64 \
  --vocab-size 1024 \
  --steps 2 \
  --warmup-steps 1
```
- Config: 1-device explicit-mesh smoke test with small model dimensions.
- Result:
  - Current failure:
```text
jax._src.core.ShardingTypeError: add got incompatible shardings for broadcasting:
('data', 'model'), (None, ('data', 'model'))
```
- Interpretation: The orthogonalization helper restores the expected sharding in eager spot checks, but at least one optimizer-produced update leaf still carries an extra leading replicated axis through the jitted update/apply path.
- Next action: Inspect optimizer update leaf shardings directly, fix the mismatch at the optimizer boundary, then re-run the local microbench before attempting TPU allocation.

### 2026-03-19 16:xx - Fix JIT sharding mismatch in Grug Muon updates
- Hypothesis: The remaining mismatch is introduced only under JIT, because explicit-mesh tracing loses the concrete `param.sharding` object used by the original fixup.
- Command:
```bash
uv run python - <<'PY'
# Probed update leaf shardings inside and outside eqx.filter_jit.
PY
uv run --package levanter --group test python -m pytest \
  lib/levanter/tests/test_grugmuon.py \
  lib/levanter/tests/test_optimizer_linear_like.py -q
```
- Config:
  - Switched sharding restoration from `jnp.asarray(..., out_sharding=...)` to `jax.sharding.reshard(...)`
  - Resolved target shardings via `jax.typeof(param).sharding` when tracing
- Result:
  - Inside JIT, Muon updates now match parameter shardings correctly.
  - Local smoke benchmark completes again.
  - Tests still pass: `8 passed`.
- Interpretation: The optimizer microbench harness is valid again; TPU execution can proceed.
- Next action: Check TPU launch paths and try `v5p-64`, with `v5p-8` as the fallback microbench target.

### 2026-03-19 16:10 - TPU launch sanity checks
- Hypothesis: Iris `job run` is the right path for multi-host TPU benchmarking because it auto-configures replicas/coscheduling for multi-host JAX jobs.
- Command:
```bash
uv run iris --config=lib/iris/examples/marin.yaml cluster status
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-jaxcheck-m64 \
  --cpu 16 \
  --memory 64GB \
  --disk 20GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  python -c "import jax; print({'process_index': jax.process_index(), 'process_count': jax.process_count(), 'local_device_count': jax.local_device_count(), 'device_count': jax.device_count()})"
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-64 \
  --zone us-east5-a \
  --job-name codex-grug-muon-v5p64-jaxcheck-m64 \
  --cpu 16 \
  --memory 64GB \
  --disk 20GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  python -c "import jax; print({'process_index': jax.process_index(), 'process_count': jax.process_count(), 'local_device_count': jax.local_device_count(), 'device_count': jax.device_count()})"
```
- Config:
  - `v5p-8` on `us-central1-a`
  - attempted `v5p-64` on `us-east5-a`
- Result:
  - `v5p-8` succeeded with `process_count=1`, `local_device_count=4`, `device_count=4`
  - first `v5p-8` attempt failed only because Iris defaulted to `1GB` task memory and the container was OOM-killed during setup
  - `v5p-64` submission auto-expanded to `replicas=8`, but remained pending with:
```text
Scheduler: Coscheduling: need 8 workers in 'tpu-name' group ...
```
  - terminated the pending `v5p-64` job afterward
- Interpretation: Multi-host launch plumbing works, but `v5p-64` capacity was not available in a full coscheduled slice during this session.
- Next action: Use `v5p-8` for microbenching, then interpret results carefully because they may not reflect multi-host `v5p-64` communication costs.

### 2026-03-19 16:15 - Full optimizer update benchmark on `v5p-8`, target-shape 1-layer slice
- Hypothesis: Even on `v5p-8`, a 1-layer target-shape model should provide a useful end-to-end optimizer-update measurement for the Grug Muon path.
- Command:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-preserve-l1 \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization preserve_sharding \
    --steps 3 \
    --warmup-steps 1 \
    --hidden-dim 4096 \
    --num-layers 1 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 128256
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-replicated-l1 \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization replicated \
    --steps 3 \
    --warmup-steps 1 \
    --hidden-dim 4096 \
    --num-layers 1 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 128256
```
- Config:
  - 1-layer target-shape slice
  - `mesh_shape={"data": 1, "expert": 4, "model": 1}`
  - `parameter_count=1,910,779,904`
- Result:
  - preserve:
    - `compile_s=7.6119886950`
    - `mean_step_s=0.0206826423`
  - replicated:
    - `compile_s=7.6460547800`
    - `mean_step_s=0.0206080063`
- Interpretation:
  - No meaningful throughput win on this layout; preserve was about `0.36%` slower on steady-state step time.
  - This is expected because the mesh had no nontrivial `data` or `model` axis on the Muon leaves (`expert=4, model=1`), so the sharding-preserving path could not avoid any cross-device gather on the relevant contractions.
- Next action: Force a model-parallel axis on `v5p-8` for an orth-only microbench to exercise the intended sharding pattern despite missing `v5p-64` capacity.

### 2026-03-19 16:30 - Orthogonalization-only microbench on `v5p-8` with model-parallel axis
- Hypothesis: A Muon-only microbench with `expert=1, model=4` on `v5p-8` will better expose whether preserving sharding helps once orthogonalized leaves are actually model-sharded.
- Command:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-orth-preserve-m4 \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon_orth.py \
    --orthogonalization preserve_sharding \
    --steps 5 \
    --warmup-steps 1 \
    --hidden-dim 4096 \
    --num-layers 1 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 1 \
    --model-axis-size 4 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 1024
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-orth-replicated-m4 \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon_orth.py \
    --orthogonalization replicated \
    --steps 5 \
    --warmup-steps 1 \
    --hidden-dim 4096 \
    --num-layers 1 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 1 \
    --model-axis-size 4 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 1024
```
- Config:
  - Muon-only microbench over the real Grug Muon leaves from a 1-layer target-shape model
  - `mesh_shape={"data": 1, "expert": 1, "model": 4}`
  - `muon_leaf_count=8`
  - `muon_parameter_count=54,788,096`
- Result:
  - preserve:
    - `compile_s=5.8726458480`
    - `mean_step_s=0.0149386358`
  - replicated:
    - `compile_s=10.4745315360`
    - `mean_step_s=0.0134636940`
- Interpretation:
  - Preserve compiled faster, but steady-state step time was about `10.96%` slower than the replicated path on this single-host `model=4` microbench.
  - Negative result: the sharding-preserving Newton-Schulz variant does not improve single-host `v5p-8` throughput, even once a nontrivial model axis is present.
  - Strong caveat: this does not settle the multi-host `v5p-64` question, where the replicated path may incur materially higher cross-host communication costs than seen here.
- Next action:
  - Keep the benchmark harnesses and sharding fixups.
  - Re-run the matched orth microbench and the full-step 1-layer slice on a real `v5p-64` allocation when a full 8-worker coscheduled slice is available.
  - If `v5p-64` also shows preserve slower, inspect the contraction layout itself rather than just the pre-orthogonalization sharding.

### 2026-03-19 16:40 - Deeper `v5p-8` full-step benchmark with `expert=4, model=1`
- Hypothesis: A deeper no-model-parallel `v5p-8` slice should better approximate the intended data+EP-only regime, while still fitting on a single host.
- Command:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-preserve-l8 \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization preserve_sharding \
    --steps 3 \
    --warmup-steps 1 \
    --hidden-dim 4096 \
    --num-layers 8 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --model-axis-size 1 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 128256
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-replicated-l8 \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization replicated \
    --steps 3 \
    --warmup-steps 1 \
    --hidden-dim 4096 \
    --num-layers 8 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --model-axis-size 1 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 128256
```
- Config:
  - `mesh_shape={"data": 1, "expert": 4, "model": 1}`
  - `parameter_count=7,931,498,496`
- Result:
  - preserve:
    - `compile_s=13.7122033720`
    - `mean_step_s=0.1357633603`
  - replicated:
    - `compile_s=13.6754369910`
    - `mean_step_s=0.1357625973`
- Interpretation:
  - Backend choice remained effectively flat at 8 layers; preserve was slower by only `0.00056%`, well within noise.
  - Under pure data+EP sharding on single-host `v5p-8`, the orthogonalization backend is not the throughput limiter.
- Next action: Probe how far this same setup fits on `v5p-8` without changing the sharding regime.

### 2026-03-19 16:52 - `v5p-8` 12-layer fit probe, same data+EP-only layout
- Hypothesis: The same no-model-parallel `v5p-8` setup may support materially more than 8 layers, which would give a better proxy for the intended workload even without reaching 24B.
- Command:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-preserve-l12-probe \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization preserve_sharding \
    --steps 1 \
    --warmup-steps 0 \
    --hidden-dim 4096 \
    --num-layers 12 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --model-axis-size 1 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 128256
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-replicated-l12-probe \
  --cpu 32 \
  --memory 128GB \
  --disk 50GB \
  --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization replicated \
    --steps 1 \
    --warmup-steps 0 \
    --hidden-dim 4096 \
    --num-layers 12 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --model-axis-size 1 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --vocab-size 128256
```
- Config:
  - `mesh_shape={"data": 1, "expert": 4, "model": 1}`
  - `parameter_count=11,371,909,120`
  - single-step fit probe only (`steps=1`, `warmup_steps=0`)
- Result:
  - preserve:
    - `compile_s=20.4734790260`
    - `mean_step_s=0.2033368770`
  - replicated:
    - `compile_s=20.8726508810`
    - `mean_step_s=0.2040673640`
- Interpretation:
  - 12 layers also fit on `v5p-8` in the same data+EP-only layout.
  - The measured step delta was about `-0.36%` in favor of preserve, but this was only a single measured step and is not strong evidence of a real backend difference.
  - Across 1, 8, and 12 layers in the no-model-parallel regime, the two orthogonalization backends remain effectively tied on steady-state throughput.
- Next action:
  - If the goal is a better `v5p-8` proxy for the target workload, keep increasing layer count until the first memory or compile-time failure.
  - If the goal is specifically to optimize orthogonalization, the next credible measurement still needs a real multi-host `v5p-64` slice.

### 2026-03-19 17:25 - Iris submission launcher for `v5p-64` and `v5p-32`
- Hypothesis: The benchmark thread needs a reusable launcher so `v5p-64` and `v5p-32` can be re-submitted without reconstructing the full Iris command matrix by hand.
- Command:
```bash
uv run python lib/levanter/scripts/bench/submit_grug_muon_iris.py
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /dlwh/codex-grug-muon-v5p-
```
- Config:
  - new launcher: `lib/levanter/scripts/bench/submit_grug_muon_iris.py`
  - defaults:
    - TPU types: `v5p-64`, `v5p-32`
    - orthogonalizations: `preserve_sharding`, `replicated`
    - target shape: `hidden_dim=4096`, `num_layers=27`, `E=64`, `EP=4`, `K=4`, widths `1024`, `vocab_size=128256`
    - sharding regime: `expert_axis_size=4`, `model_axis_size=1`
    - Iris resources: `cpu=32`, `memory=128GB`, `disk=50GB`
- Result:
  - submitted jobs:
    - `/dlwh/codex-grug-muon-v5p-64-preserve_sharding-l27-20260320-002529`
    - `/dlwh/codex-grug-muon-v5p-64-replicated-l27-20260320-002529`
    - `/dlwh/codex-grug-muon-v5p-32-preserve_sharding-l27-20260320-002529`
    - `/dlwh/codex-grug-muon-v5p-32-replicated-l27-20260320-002529`
  - immediate status after submission:
    - both `v5p-64` jobs pending with `need 8 workers in 'tpu-name' group`
    - both `v5p-32` jobs pending with `need 4 workers in 'tpu-name' group`
- Interpretation:
  - Submission is automated now; the remaining blocker is TPU slice availability, not launch mechanics.
  - The launcher is now the easiest path to retry or adjust target shape/resources without reconstructing raw Iris commands.
- Next action:
  - Re-check the four submitted jobs until one of the slices places.
  - If pending persists, consider zone pinning or reduced `num_layers` only as a capacity workaround, not as a correctness workaround.

## Open Questions
- When a full 8-worker `v5p-64` slice is available, does the replicated path degrade enough from cross-host communication that the preserve-sharding path becomes worthwhile?
- If preserve remains slower on `v5p-64`, should the next optimization target be the Newton-Schulz contraction schedule itself rather than the pre-contraction sharding policy?
- Is there a better contraction layout for Grug Muon on TPU that keeps sharding-friendly intermediates without paying the current preserve-path step-time penalty seen on `v5p-8`?

### 2026-03-19 14:52 - Submitted `v5p-32` and `v5p-64` jobs completed
- Hypothesis:
  - The real multi-host, no-model-parallel target-shape run might change the orthogonalization conclusion seen on single-host `v5p-8`, especially if replicated orthogonalization paid meaningful cross-host communication.
- Commands:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /dlwh/codex-grug-muon-v5p-
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-64-replicated-l27-20260320-002529 | tail -n 80
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-64-preserve_sharding-l27-20260320-002529 | tail -n 80
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-32-replicated-l27-20260320-002529 | tail -n 60
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-32-preserve_sharding-l27-20260320-002529 | tail -n 60
```
- Config:
  - all four submitted jobs reached `succeeded`
  - target shape:
    - `hidden_dim=4096`
    - `num_layers=27`
    - `num_experts=64`
    - `num_experts_per_token=4`
    - routed/shared widths `1024`
    - `vocab_size=128256`
  - sharding regime:
    - `expert_axis_size=4`
    - `model_axis_size=1`
  - representative mesh shapes from worker logs:
    - `v5p-64`: `{"data": 8, "expert": 4, "model": 1}`
    - `v5p-32`: `{"data": 4, "expert": 4, "model": 1}`
  - representative parameter count:
    - `24,273,448,960`
- Result:
  - `v5p-64 preserve_sharding`:
    - `compile_s=185.6730599110`
    - `mean_step_s=0.9888612680`
  - `v5p-64 replicated`:
    - `compile_s=210.4923184880`
    - `mean_step_s=0.3913710763`
  - `v5p-32 preserve_sharding`:
    - `compile_s=145.8529972600`
    - `mean_step_s=0.5897596603`
  - `v5p-32 replicated`:
    - `compile_s=99.5552359200`
    - `mean_step_s=0.3921823160`
- Interpretation:
  - The multi-host result decisively rejects the idea that preserve-sharding is helping throughput in the target data+EP-only regime.
  - On `v5p-64`, replicated is about `2.53x` faster in step time than preserve (`0.391s` vs `0.989s`).
  - On `v5p-32`, replicated is about `1.50x` faster in step time than preserve (`0.392s` vs `0.590s`).
  - The `v5p-64` replicated result is also essentially flat versus `v5p-32` replicated, which suggests the current bottleneck is not improving with the larger slice in this setup.
  - The likely optimization target is now the preserve-sharding Newton-Schulz path itself, or removing it entirely for this regime, rather than trying to preserve array sharding through orthogonalization.
- Next action:
  - If the goal is pure throughput, default to `replicated` for this benchmark configuration.
  - If preserve-sharding still matters semantically, profile the preserve path directly to find which collectives or contraction layouts are causing the `1.5x` to `2.5x` slowdown.

### 2026-03-19 15:09 - Fix Grug Muon semantics for expert-stacked matrices
- Hypothesis:
  - The prior benchmark was incomplete in an important way: Grug MoE expert weights are rank-3 stacks of matrices, so the existing `param.ndim == 2` mask skipped the bulk of the expert update path entirely.
  - The preserve-sharding variant was also not preserving anything meaningful, because it hard-coded `P(None, ("data", "model"))` instead of deriving a target from the original partition spec.
- Commands:
```bash
uv run --package levanter --group test python -m pytest lib/levanter/tests/test_grugmuon.py -q
uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
  --orthogonalization replicated \
  --steps 1 \
  --warmup-steps 0 \
  --hidden-dim 64 \
  --num-layers 1 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --head-dim 16 \
  --num-experts 8 \
  --expert-axis-size 1 \
  --model-axis-size 1 \
  --num-experts-per-token 2 \
  --routed-expert-width 32 \
  --shared-expert-width 32 \
  --vocab-size 512
```
- Config:
  - optimizer changes in `lib/levanter/src/levanter/optim/grugmuon.py`
  - test coverage in `lib/levanter/tests/test_grugmuon.py`
  - new semantics:
    - rank-3 `w_up_gate` and `w_down` leaves are now routed to Muon
    - Muon now applies Newton-Schulz to rank-3 expert stacks by vmapping over the leading expert dimension
    - preserve-sharding now derives its target pspec from the original sharding and clears `data`/`model` instead of hard-coding a fixed pspec
- Result:
  - `pytest`: `2 passed`
  - small bench smoke:
    - `compile_s=0.3561825000`
    - `mean_step_s=0.0018092501`
    - `parameter_count=133,824`
- Interpretation:
  - The benchmark semantics are now closer to what Grug Muon is actually supposed to optimize: per-expert orthogonalization of the stacked expert matrices, not just the rank-2 leaves.
  - The earlier `v5p-32` and `v5p-64` numbers are still useful as a baseline for the old implementation, but they are no longer the final answer for the corrected optimizer path.
  - Algorithmically, the expert axis is the only natural batch axis to preserve through Newton-Schulz. The inner matrix axes are repeatedly contracted, so trying to keep them sharded is unlikely to be beneficial unless a very specific contraction layout proves otherwise.
- Next action:
  - Re-run the `v5p-32` and `v5p-64` Iris benchmarks with the corrected Muon coverage over expert matrices.
  - Compare whether `replicated` still dominates once the expert-stack updates are actually included.

### 2026-03-19 15:11 - Submitted corrected multi-host reruns
- Hypothesis:
  - With Muon now covering the stacked expert matrices, the old multi-host baseline is stale and needs a fresh `v5p-32` / `v5p-64` rerun before drawing any throughput conclusion.
- Command:
```bash
uv run python lib/levanter/scripts/bench/submit_grug_muon_iris.py
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /dlwh/codex-grug-muon-v5p-64-
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /dlwh/codex-grug-muon-v5p-32-
```
- Config:
  - same target shape as prior multi-host runs:
    - `hidden_dim=4096`
    - `num_layers=27`
    - `num_experts=64`
    - `expert_axis_size=4`
    - `model_axis_size=1`
    - `num_experts_per_token=4`
    - routed/shared widths `1024`
    - `vocab_size=128256`
  - corrected optimizer semantics from the previous entry are now in the submitted workspace bundle
  - submitted jobs:
    - `/dlwh/codex-grug-muon-v5p-64-preserve_sharding-l27-20260320-051028`
    - `/dlwh/codex-grug-muon-v5p-64-replicated-l27-20260320-051028`
    - `/dlwh/codex-grug-muon-v5p-32-preserve_sharding-l27-20260320-051028`
    - `/dlwh/codex-grug-muon-v5p-32-replicated-l27-20260320-051028`
- Result:
  - all four jobs submitted successfully
  - immediate state:
    - both `v5p-64` jobs pending on `need 8 workers in 'tpu-name' group`
    - both `v5p-32` jobs pending on `need 4 workers in 'tpu-name' group`
- Interpretation:
  - The rerun matrix is live and waiting only on coscheduled TPU capacity.
  - No code or submission error is blocking the corrected comparison.
- Next action:
  - Poll until the slices place and then capture the new throughput numbers.

### 2026-03-19 15:25 - Corrected multi-host rerun results
- Hypothesis:
  - Once Muon actually covers the stacked MoE expert matrices, the multi-host throughput ranking between `preserve_sharding` and `replicated` may change substantially from the earlier, incomplete baseline.
- Commands:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /dlwh/codex-grug-muon-v5p-64-preserve_sharding-l27-20260320-051028
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /dlwh/codex-grug-muon-v5p-32-preserve_sharding-l27-20260320-051028
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /dlwh/codex-grug-muon-v5p-32-replicated-l27-20260320-051028
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-64-preserve_sharding-l27-20260320-051028 | tail -n 80
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-32-preserve_sharding-l27-20260320-051028 | tail -n 60
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-32-replicated-l27-20260320-051028 | tail -n 60
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-64-replicated-l27-20260320-051028 | tail -n 120
uv run iris --config=lib/iris/examples/marin.yaml job logs /dlwh/codex-grug-muon-v5p-64-replicated-l27-20260320-051656 | tail -n 80
```
- Config:
  - target shape unchanged:
    - `hidden_dim=4096`
    - `num_layers=27`
    - `num_experts=64`
    - `expert_axis_size=4`
    - `model_axis_size=1`
    - `num_experts_per_token=4`
    - routed/shared widths `1024`
    - `vocab_size=128256`
  - corrected Muon semantics:
    - expert-stack rank-3 matrices included via per-expert vmapped Newton-Schulz
  - job outcomes:
    - `/dlwh/codex-grug-muon-v5p-64-preserve_sharding-l27-20260320-051028`: `succeeded`
    - `/dlwh/codex-grug-muon-v5p-32-preserve_sharding-l27-20260320-051028`: `succeeded`
    - `/dlwh/codex-grug-muon-v5p-32-replicated-l27-20260320-051028`: `succeeded`
    - `/dlwh/codex-grug-muon-v5p-64-replicated-l27-20260320-051028`: `failed`
    - `/dlwh/codex-grug-muon-v5p-64-replicated-l27-20260320-051656`: `failed`
- Result:
  - `v5p-64 preserve_sharding`:
    - `compile_s=156.0996081320`
    - `mean_step_s=0.9769447960`
  - `v5p-32 preserve_sharding`:
    - `compile_s=153.8125933670`
    - `mean_step_s=0.9705001827`
  - `v5p-32 replicated`:
    - `compile_s=156.6890906630`
    - `mean_step_s=1.0253053377`
  - `v5p-64 replicated`:
    - no benchmark result yet
    - both attempts failed during container build while downloading the Astral Python standalone tarball from GitHub
- Interpretation:
  - Correcting the optimizer semantics changed the conclusion materially.
  - On `v5p-32`, `preserve_sharding` is now about `5.6%` faster than `replicated` in step time (`0.9705s` vs `1.0253s`).
  - The earlier result where `replicated` looked dramatically better was mostly an artifact of skipping the MoE expert-stack matrices.
  - `v5p-64 preserve_sharding` landed at `0.9769s`, essentially flat with `v5p-32 preserve_sharding`, so the corrected preserve path is not scaling meaningfully with the larger slice in this configuration.
  - The missing quadrant is `v5p-64 replicated`, but the blocker is infrastructure, not the benchmark itself.
- Next action:
  - Get one clean `v5p-64 replicated` run through container build to complete the matrix.
  - If the build failures persist, either reuse a warmed image/runtime path or pin to a zone/worker pool with the necessary Python artifact already cached.

### 2026-03-19 23:33 - Centralized final reshards and profiled the cleaned-up stack path
- Hypothesis:
  - The current Grug Muon update path is doing duplicated layout restoration: Newton-Schulz helpers reshard back to the original parameter layout, and then the optimizer chain reshards the updates again via `_match_update_sharding()`.
  - If that duplication is removed, the fair `stack_batch_sharded` vs `vmap_replicated` comparison on `v5p-8` should tighten materially, and any remaining bottleneck should mostly be the Newton-Schulz batched matmuls themselves.
- Commands:
```bash
uv run --package levanter --group test python -m pytest lib/levanter/tests/test_grugmuon.py -q

uv run iris --config=lib/iris/examples/marin.yaml job run --tpu v5p-8 --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-stack-bench-rerun \
  --cpu 32 --memory 128GB --disk 50GB --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization stack_batch_sharded \
    --steps 3 --warmup-steps 1 \
    --vocab-size 1024 \
    --hidden-dim 4096 \
    --num-layers 1 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --model-axis-size 1 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024

uv run iris --config=lib/iris/examples/marin.yaml job run --tpu v5p-8 --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-vmap-rerun \
  --cpu 32 --memory 128GB --disk 50GB --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization vmap_replicated \
    --steps 3 --warmup-steps 1 \
    --vocab-size 1024 \
    --hidden-dim 4096 \
    --num-layers 1 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --model-axis-size 1 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024

uv run iris --config=lib/iris/examples/marin.yaml job run --tpu v5p-8 --zone us-central1-a \
  --job-name codex-grug-muon-v5p8-stack-profile-rerun \
  --cpu 32 --memory 128GB --disk 50GB --extra tpu \
  -e LIBTPU_INIT_ARGS --xla_tpu_scoped_vmem_limit_kib=50000 -- \
  uv run python lib/levanter/scripts/bench/bench_grug_muon.py \
    --orthogonalization stack_batch_sharded \
    --steps 3 --warmup-steps 1 \
    --vocab-size 1024 \
    --hidden-dim 4096 \
    --num-layers 1 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-experts 64 \
    --expert-axis-size 4 \
    --model-axis-size 1 \
    --num-experts-per-token 4 \
    --routed-expert-width 1024 \
    --shared-expert-width 1024 \
    --profile-dir /tmp/grug-muon-stack-profile-rerun
```
- Config:
  - code change in `lib/levanter/src/levanter/optim/grugmuon.py`:
    - remove the extra `reshard(..., target_sharding)` at the end of `transform_array`
    - stop the Newton-Schulz helper functions from restoring original sharding internally
    - keep `_match_update_sharding()` as the single place that restores final parameter layout
  - benchmark shape:
    - `hidden_dim=4096`
    - `num_layers=1`
    - `num_experts=64`
    - `expert_axis_size=4`
    - `model_axis_size=1`
    - `num_experts_per_token=4`
    - routed/shared widths `1024`
    - `vocab_size=1024`
- Result:
  - tests:
    - `pytest`: `4 passed`
  - `v5p-8 stack_batch_sharded`:
    - `compile_s=10.0356450820`
    - `mean_step_s=0.0398159247`
  - `v5p-8 vmap_replicated`:
    - `compile_s=10.0847834780`
    - `mean_step_s=0.0398860120`
  - fair A/B delta:
    - `stack_batch_sharded` is faster by about `0.18%`
    - effectively a tie
  - profiled `v5p-8 stack_batch_sharded` rerun:
    - `compile_s=10.0203378860`
    - `mean_step_s=0.0416678067`
    - time breakdown:
      - compute share `62.76%`
      - host share `37.23%`
      - communication share `0%`
    - hottest ops still map directly to the batched Newton-Schulz contractions:
      - `grugmuon.py:423` (`...ik,...jk->...ij`)
      - `grugmuon.py:424` (`...ik,...kj->...ij`)
      - `grugmuon.py:425` (`...ik,...kj->...ij`)
    - top residual gap/copy signal:
      - largest pre-op gap is still before `copy.108`
      - `total_gap_duration=63079.09375`
      - copy family share is still visible, but much smaller than the total compute share
- Interpretation:
  - The meaningful optimization in this pass was not choosing between `stack_batch_sharded` and `vmap_replicated`; it was removing duplicated resharding and making final layout restoration happen exactly once.
  - After that cleanup, the two orthogonalization layouts are basically tied on single-host `v5p-8`.
  - The current stack-sharded path is already doing the “more aggressive” batch-axis sharding idea for rank-3 expert stacks: when the leading expert count is divisible by the product of nontrivial mesh axes, it shards the stack over all of them, yielding the moral equivalent of `P(("data", "expert", "model"), None, None)` when those axes exist.
  - The remaining stack-path cost is now mostly real Newton-Schulz batched matmul work, plus a smaller amount of copy / host-side overhead. There is no longer a large layout-restoration artifact dominating the comparison.
- Next action:
  - If more throughput is needed, the next likely targets are:
    - reducing the residual `dynamic_nodonate` copy traffic via donation / aliasing improvements
    - replacing the batched Newton-Schulz dot-generals with a more specialized kernel path
    - validating whether the same “single final reshard” cleanup changes the multi-host `v5p-32` / `v5p-64` ranking materially
