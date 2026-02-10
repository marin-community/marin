# MoE EP Int8: Research Logbook

## Scope
- Goal: evaluate int8 quantized MoE training-path throughput in the EP hillclimb harness on v5p.
- Primary metric(s): `forward_backward` TF/s and tokens/s, with delta vs `quant_mode=none`.
- Constraints: keep comparisons apples-to-apples (same shape, backend, impl, pass mode, routing pack strategy, and device count).

## Links
- Experiment issue: https://github.com/marin-community/marin/issues/2710
- Prior experiment issue: https://github.com/marin-community/marin/issues/2704
- Research recipe: `docs/recipes/agent_research.md`
- Harness: `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- Related EP logbook: `.agents/logbooks/moe_ep_benchmark.md`

## Baseline
- Date: 2026-02-09
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `.agents/logbooks/moe_ep_benchmark.md`
- Baseline numbers (quantization off, compact EP path):
  - shape: `tokens=32768 hidden=2048 mlp_dim=1408 experts=60 topk=2 shared=5632 backend=gmm impl=fused_w13 parallel=ep routing=argsort pass=forward_backward shared_fused`
  - TF/s: `239.150`
  - confidence: `replicated` (from prior sweep logbook)

## Stop Criteria
- We have direct `quant_mode=none` vs `quant_mode=int8` comparisons on v5p for at least one fixed baseline case.
- We have one top-k sweep (`1,2,4,6,8`) in int8 mode with identical non-quant baseline controls.
- We can state whether int8 is beneficial/neutral/regressive for this harness regime.

## Experiment ID Prefix
- `MOE-I8`

## Experiment Log
### 2026-02-10 10:40 - MOE-I8-000 Kickoff (confidence: exploratory)
- Hypothesis: int8 fake-quantized matmul path may improve effective throughput on v5p EP MoE shapes.
- Command: N/A (setup)
- Config:
  - branch: `codex/int8-moe-ep-v5p`
  - harness change: add `--quant-mode {none,int8}`
- Result:
  - Added int8 quant mode to benchmark harness and connected through non-EP, EP, and stage-timing paths.
  - Imported canonical research playbook from `docs/agent-research-logbook-recipe` as `docs/recipes/agent_research.md`.
- Interpretation: harness is ready for A/B runs.
- Next action: run int8 A/B on v5p dev TPU and append results.

### 2026-02-10 10:43 - MOE-I8-001 Local smoke tests (confidence: exploratory)
- Hypothesis: quantized path executes end-to-end without changing harness control flow.
- Command:
  - `uv run --package levanter python lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 128 --hidden 64 --mlp-dim 32 --experts 4 --topk 2 --distribution random --backend ragged_dot --impl baseline --bench-pass forward --parallel-mode none --quant-mode int8 --iters 1 --warmup 0 --dtype float32`
  - `uv run --package levanter python lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 128 --hidden 64 --mlp-dim 32 --experts 4 --topk 2 --distribution random --backend gmm --impl fused_w13 --bench-pass forward --parallel-mode none --quant-mode int8 --iters 1 --warmup 0 --dtype float32`
  - Negative control:
    - `uv run --package levanter python lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 128 --hidden 64 --mlp-dim 32 --experts 4 --topk 2 --distribution random --backend gmm --impl fused_w13 --bench-pass forward --parallel-mode none --quant-mode int8 --iters 1 --warmup 0 --dtype bfloat16`
- Config: CPU smoke only, tiny shape.
- Result:
  - `ragged_dot/int8`: successful run.
  - `gmm/int8/float32`: successful run.
  - `gmm/int8/bfloat16`: failed with existing harness/backend issue:
    - `TypeError: pad operand and padding_value must be same dtype: got bfloat16 and float32`
- Interpretation:
  - Int8 harness plumbing works.
  - Existing `gmm_sharded` pad literal dtype mismatch remains a known blocker for bf16 on this CPU smoke path.
- Next action: run on v5p where benchmark context is intended; keep bf16 pad mismatch in mind for local-only tests.

### 2026-02-10 10:51 - MOE-I8-002 Dev TPU launcher checks (confidence: exploratory)
- Hypothesis: dev TPU command path is immediately runnable for v5p A/B.
- Command:
  - `.venv/bin/python scripts/ray/dev_tpu.py --help`
  - `RAY_AUTH_MODE=token .venv/bin/python scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -- "echo codex-int8-check && hostname"`
  - `uv run --package marin python scripts/ray/dev_tpu.py --help`
- Config: local launcher validation only.
- Result:
  - Direct `.venv/bin/python` invocation failed due missing dependency:
    - `ModuleNotFoundError: No module named 'watchdog'`
  - `uv run --package marin ...` reached environment build, but was blocked by long Rust build path (`dupekit`) before launcher output.
- Interpretation:
  - No v5p measurement yet; current blocker is launcher/runtime environment readiness in this worktree.
- Next action:
  - Run `uv sync` (or equivalent package install path that includes `watchdog`) and retry `dev_tpu.py` allocation/execute.

### 2026-02-10 10:55 - MOE-I8-003 Planned first v5p matrix (confidence: exploratory)
- Hypothesis: first actionable int8 signal comes from strict A/B on one fixed shape before broader sweeps.
- Command (planned):
  - Baseline:
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name "$USER-moe-int8" execute -- uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode none --iters 3 --warmup 1`
  - Int8:
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name "$USER-moe-int8" execute -- uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode int8 --iters 3 --warmup 1`
- Config: fixed shape, fixed backend/impl, EP on.
- Result: pending.
- Interpretation: this is the minimum reproducible A/B bundle for first claim.
- Next action: execute commands once launcher environment is unblocked, then append measured deltas.

### 2026-02-10 11:01 - MOE-I8-004 First v5p A/B (shared=5632, EP) (confidence: exploratory)
- Hypothesis: int8 path can improve EP train-like throughput on a Qwen-like shape.
- Command:
  - `ssh dev-tpu-codex-int8 'cd /home/dlwh/marin && ... /home/dlwh/marin/.venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode "$q" --iters 3 --warmup 1'`
  - `q in {none,int8}`, `tk in {2,8}`
- Config:
  - devices: 4 (`v5p-8` host)
  - backend: `gmm`
  - impl: `fused_w13`
  - parallel_mode: `ep`
  - pass: `forward_backward`
  - routing pack: `argsort`
  - shared expert: `5632`, `shared_fused=true`
- Result:
  - `topk=2`: none `252.451 TF/s`, int8 `230.432 TF/s` (`-8.72%`)
  - `topk=8`: none `415.992 TF/s`, int8 `388.639 TF/s` (`-6.58%`)
  - remote log: `.agents/logbooks/moe_ep_int8_run1_20260210_190131.log`
- Interpretation:
  - No lift from current int8 path on this shape; regression is material.
- Next action:
  - Isolate whether shared path contributes disproportionate int8 overhead via `shared_expert_dim=0` run.

### 2026-02-10 11:05 - MOE-I8-005 Shared-off A/B (shared=0, EP) (confidence: exploratory)
- Hypothesis: int8 regression might be dominated by shared-expert quantized dense matmuls.
- Command:
  - `ssh dev-tpu-codex-int8 'cd /home/dlwh/marin && ... /home/dlwh/marin/.venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 0 --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode "$q" --iters 3 --warmup 1'`
  - `q in {none,int8}`, `tk in {2,8}`
- Config:
  - same as MOE-I8-004 except `shared_expert_dim=0`
- Result:
  - `topk=2`: none `181.036 TF/s`, int8 `175.093 TF/s` (`-3.28%`)
  - `topk=8`: none `444.455 TF/s`, int8 `429.591 TF/s` (`-3.34%`)
  - remote log: `.agents/logbooks/moe_ep_int8_run2_20260210_190553.log`
- Interpretation:
  - Regression persists even with shared path removed, but drops from `~7-9%` to `~3.3%`.
  - Shared path contributes additional int8 penalty, but is not the root cause.
- Next action:
  - Check a higher-arithmetic-intensity shape (`3072/3072`) to test if regression vanishes.

### 2026-02-10 11:11 - MOE-I8-006 Larger-shape A/B (3072/3072, EP) (confidence: exploratory)
- Hypothesis: int8 may recover at larger matmul sizes where quant/dequant overhead amortizes better.
- Command:
  - `ssh dev-tpu-codex-int8 'cd /home/dlwh/marin && ... /home/dlwh/marin/.venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk 8 --shared-expert-dim 0 --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode "$q" --iters 3 --warmup 1'`
  - `q in {none,int8}`
- Result:
  - none `491.551 TF/s`
  - int8 `475.733 TF/s` (`-3.22%`)
  - remote log: `.agents/logbooks/moe_ep_int8_run3_20260210_191144.log`
- Interpretation:
  - No lift at larger shape; regression remains ~3%.
- Next action:
  - Run stage-timing profile to attribute regression to routing vs expert matmul stages.

### 2026-02-10 11:13 - MOE-I8-007 Stage profile attribution (confidence: exploratory)
- Hypothesis: if int8 overhead is mostly from extra quant/dequant work, `up/down` stages should increase while `pack/combine` stay flat.
- Command:
  - shared off:
    - `ssh dev-tpu-codex-int8 'cd /home/dlwh/marin && ... /home/dlwh/marin/.venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --shared-expert-dim 0 --backend gmm --impl fused_w13 --parallel-mode none --routing-pack-strategy argsort --queue-mode full --bench-pass forward --quant-mode "$q" --iters 3 --warmup 1 --stage-timing'`
  - shared on:
    - `ssh dev-tpu-codex-int8 'cd /home/dlwh/marin && ... /home/dlwh/marin/.venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --parallel-mode none --routing-pack-strategy argsort --queue-mode full --bench-pass forward --quant-mode "$q" --iters 3 --warmup 1 --stage-timing'`
  - `q in {none,int8}`
- Result:
  - shared off (`topk=8`, forward):
    - none: `time=0.027644`, `tflops=238.640`, stage seconds: `pack=0.004189`, `up=0.013398`, `down=0.006262`, `combine=0.004739`
    - int8: `time=0.031711`, `tflops=208.036`, stage seconds: `pack=0.004181`, `up=0.015950`, `down=0.008137`, `combine=0.004759`
  - shared on (`topk=8`, forward):
    - none: `time=0.033482`, `tflops=264.767`, stage seconds: `pack=0.004184`, `up=0.013386`, `down=0.006215`, `combine=0.004761`
    - int8: `time=0.039705`, `tflops=223.267`, stage seconds: `pack=0.004185`, `up=0.015962`, `down=0.008136`, `combine=0.004747`
  - remote logs:
    - `.agents/logbooks/moe_ep_int8_stage_20260210_190843.log`
    - `.agents/logbooks/moe_ep_int8_stage_shared_20260210_191319.log`
- Interpretation:
  - `pack/combine` are unchanged.
  - Regression is concentrated in expert compute path (`up` and `down`), consistent with qdq overhead.
  - Current int8 implementation is not giving kernel-level int8 speedup in this harness.
- Next action:
  - Evaluate a true int8 compute path for expert matmuls (AQT / int8-native kernel integration) instead of qdq emulation around bf16 `gmm_sharded`.

## Current Status
- Across tested v5p EP shapes, current `quant_mode=int8` is consistently slower (`~3%` to `~9%`).
- No evidence of positive lift with the current implementation.

### 2026-02-10 11:40 - MOE-I8-008 Dense-only int8 sweep (shared path + dense dot, EP) (confidence: exploratory)
- Hypothesis: if routed experts stay on fast `gmm` and only dense/shared projections use int8 dot_general, total step throughput may improve.
- Command:
  - `RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name codex-int8 execute -- /bin/bash -lc 'set -euo pipefail; cd ~/marin; sudo rm -f /tmp/libtpu_lockfile; for q in none int8; do for tk in 1 2 4 6 8; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --parallel-mode ep --quant-mode "$q" --iters 3 --warmup 1; done; done'`
- Config:
  - v5p-8 host (4 local devices)
  - `tokens=32768 hidden=2048 mlp_dim=2048 experts=64 shared=5632 shared_fused`
  - backend=`gmm`, impl=`fused_w13`, pass=`forward_backward`, mode=`ep`
- Result:
  - topk=1: none `213.926`, int8 `218.259` (`+2.03%`)
  - topk=2: none `252.906`, int8 `259.338` (`+2.54%`)
  - topk=4: none `321.321`, int8 `326.563` (`+1.63%`)
  - topk=6: none `372.985`, int8 `380.450` (`+2.00%`)
  - topk=8: none `417.870`, int8 `423.142` (`+1.26%`)
- Interpretation:
  - This mode is consistently faster across the tested top-k sweep.
  - The lift is modest but reliable (`~1.3%` to `~2.5%`) for this shape.
- Next action:
  - Explore routed-expert int8 path with safer scaling policy (`per-token` activation scales + `per-expert` weight scales), as suggested by collaborator.

### 2026-02-10 11:52 - MOE-I8-009 Routed int8 with per-token/per-expert scaling (confidence: exploratory)
- Hypothesis: routed expert int8 can become net-positive if we avoid shared/global scaling and use:
  - per-token activation scales,
  - one scale per expert weight tensor,
  - dequantization once at expert output.
- Code change:
  - Added `quant_mode=int8_routed` in `lib/levanter/scripts/bench/bench_moe_hillclimb.py`.
  - Implementation details:
    - `_symmetric_int8_quantize` helper.
    - `_expert_matmul_int8_routed`: int8 `ragged_dot_general` with `preferred_element_type=int32` and row-wise dequant using token scale * expert scale.
    - Partial pack+quant fusion: quantize token activations once and gather pre-quantized routed rows for the first expert projection.
- Command:
  - `RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name codex-int8 execute -- /bin/bash -lc 'set -euo pipefail; cd ~/marin; sudo rm -f /tmp/libtpu_lockfile; for q in none int8 int8_routed; do for tk in 2 8; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --parallel-mode ep --quant-mode "$q" --iters 3 --warmup 1; done; done'`
- Config:
  - same fixed v5p shape as MOE-I8-008.
- Result:
  - topk=2:
    - none `252.506`
    - int8 `259.714` (`+2.85%`)
    - int8_routed `267.822` (`+6.07%` vs none, `+3.12%` vs int8)
  - topk=8:
    - none `416.677`
    - int8 `423.129` (`+1.55%`)
    - int8_routed `473.409` (`+13.85%` vs none, `+11.88%` vs int8)
- Numerical smoke check (CPU, small shape, ragged backend):
  - `int8 vs none`: rel-norm error `~0.0138`
  - `int8_routed vs none`: rel-norm error `~0.0126`
- Interpretation:
  - The per-token/per-expert-scale routed int8 path shows a much larger throughput gain than dense-only int8 on this benchmark shape.
  - This is promising but still exploratory; needs replication across additional shapes and stage-timing attribution.
- Next action:
  - Replicate `int8_routed` on shared-off and larger-shape regimes (`shared=0`, `3072/3072`) and run stage breakdown to confirm where the lift comes from.

### 2026-02-10 12:05 - MOE-I8-010 Routed-int8 replication on shared-off and larger shape (confidence: exploratory)
- Hypothesis: the `int8_routed` lift seen on shared-heavy shape should persist when shared experts are removed and at larger hidden/mlp dims.
- Command:
  - `ssh dev-tpu-codex-int8 'cd /home/dlwh/marin && ... /home/dlwh/marin/.venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 0 --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode "$q" --iters 3 --warmup 1'`
  - `q in {none,int8_routed}`, `tk in {2,8}`
  - larger-shape check:
    - `ssh dev-tpu-codex-int8 'cd /home/dlwh/marin && ... /home/dlwh/marin/.venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk 8 --shared-expert-dim 0 --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode "$q" --iters 3 --warmup 1'`
- Config:
  - v5p-8 host (4 local devices)
  - backend=`gmm`, impl=`fused_w13`, pass=`forward_backward`, routing=`argsort`
- Result:
  - `shared=0, topk=2`: none `181.533`, int8_routed `200.026` (`+10.19%`)
  - `shared=0, topk=8`: none `442.142`, int8_routed `536.599` (`+21.36%`)
  - `hidden=3072, mlp_dim=3072, shared=0, topk=8`: none `488.893`, int8_routed `669.610` (`+36.96%`)
- Interpretation:
  - Positive lift is not limited to shared-expert-heavy setup.
  - Lift increases with arithmetic intensity in this harness/configuration.
- Next action:
  - Gather stage-timing for `int8_routed` to identify where incremental speedup is coming from.

### 2026-02-10 12:25 - MOE-I8-011 Rerun blocker after re-allocation (confidence: operational)
- Hypothesis: rerunning MOE-I8-010 matrix after fresh `dev_tpu allocate` should reproduce the lift.
- Command:
  - `RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name codex-int8 allocate --tpu-type v5p-8`
  - follow-up rerun via `dev_tpu execute` and direct `ssh`.
- Result:
  - Bench startup failed repeatedly with:
    - `TPU initialization failed: open(/dev/vfio/0): Device or resource busy`
  - This happened before first benchmark step despite removing `/tmp/libtpu_lockfile`.
- Interpretation:
  - Repro run is currently blocked by host TPU device ownership contention.
  - Existing MOE-I8-010 numbers remain the latest completed replication set.
- Next action:
  - Retry on a clean allocation/host or after TPU VM reset, then compare against MOE-I8-010 deltas.

### 2026-02-10 13:04 - MOE-I8-012 Hybrid mode (`int8_all`) on shared-heavy shape (confidence: exploratory)
- Hypothesis: adding dense/shared int8 (`Int8DotGeneralOp`) on top of routed int8 should stack additional lift when shared experts are enabled.
- Code change:
  - Added `quant_mode=int8_all` in `lib/levanter/scripts/bench/bench_moe_hillclimb.py`.
  - Behavior:
    - routed experts: same as `int8_routed` (per-token lhs scales + per-expert rhs scales, int32 ragged accumulate).
    - shared/dense path: int8 `dot_general` like existing `int8` mode.
- Command:
  - `ssh dev-tpu-codex-int8 'cd ~/marin; for q in none int8_routed int8_all; do for tk in 2 8; do ... --shared-expert-dim 5632 --shared-fused --quant-mode "$q" --topk "$tk" ...; done; done'`
- Config:
  - `tokens=32768 hidden=2048 mlp_dim=2048 experts=64 shared=5632 topk in {2,8}`
  - `backend=gmm impl=fused_w13 pass=forward_backward parallel_mode=ep`
- Result:
  - topk=2:
    - none `252.091`
    - int8_routed `266.848` (`+5.85%`)
    - int8_all `273.881` (`+8.64%` vs none, `+2.64%` vs int8_routed)
  - topk=8:
    - none `415.572`
    - int8_routed `471.612` (`+13.49%`)
    - int8_all `483.006` (`+16.23%` vs none, `+2.42%` vs int8_routed)
  - remote log: `.agents/logbooks/moe_ep_int8_run6_20260210_210439.log`
- Interpretation:
  - Dense/shared int8 contributes additional throughput on top of routed int8 for shared-heavy config.
- Next action:
  - Validate that `int8_all` does not regress shared-off regimes.

### 2026-02-10 13:09 - MOE-I8-013 Shared-off sanity for `int8_all` (confidence: exploratory)
- Hypothesis: with `shared_expert_dim=0`, `int8_all` should match `int8_routed` (or be very close).
- Command:
  - `ssh dev-tpu-codex-int8 'cd ~/marin; for q in none int8_routed int8_all; do for tk in 2 8; do ... --shared-expert-dim 0 --quant-mode "$q" --topk "$tk" ...; done; done'`
- Config:
  - `tokens=32768 hidden=2048 mlp_dim=2048 experts=64 shared=0`
  - `backend=gmm impl=fused_w13 pass=forward_backward parallel_mode=ep`
- Result:
  - topk=2:
    - none `180.651`
    - int8_routed `199.306` (`+10.33%`)
    - int8_all `199.696` (`+10.54%` vs none, `+0.20%` vs int8_routed)
  - topk=8:
    - none `443.042`
    - int8_routed `536.363` (`+21.06%`)
    - int8_all `534.349` (`+20.61%` vs none, `-0.38%` vs int8_routed)
  - remote log: `.agents/logbooks/moe_ep_int8_run7_20260210_210901.log`
- Interpretation:
  - `int8_all` is effectively neutral relative to `int8_routed` when shared path is disabled.
- Next action:
  - Confirm same neutrality at larger hidden/mlp shape.

### 2026-02-10 13:13 - MOE-I8-014 Larger-shape check for `int8_all` (confidence: exploratory)
- Hypothesis: at `3072/3072` with shared-off, `int8_all` should remain equivalent to `int8_routed`.
- Command:
  - `ssh dev-tpu-codex-int8 'cd ~/marin; for q in none int8_routed int8_all; do ... --hidden 3072 --mlp-dim 3072 --topk 8 --shared-expert-dim 0 --quant-mode "$q" ...; done'`
- Config:
  - `tokens=32768 hidden=3072 mlp_dim=3072 experts=64 topk=8 shared=0`
  - `backend=gmm impl=fused_w13 pass=forward_backward parallel_mode=ep`
- Result:
  - none `489.379`
  - int8_routed `668.039` (`+36.51%`)
  - int8_all `668.158` (`+36.53%` vs none, `+0.02%` vs int8_routed)
  - remote log: `.agents/logbooks/moe_ep_int8_run8_20260210_211310.log`
- Interpretation:
  - At large shared-off shape, `int8_all` and `int8_routed` are functionally identical.
- Next action:
  - Attribute `int8_all` incremental gain to shared path with stage timing.

### 2026-02-10 13:15 - MOE-I8-015 Stage attribution for `int8_all` delta (confidence: exploratory)
- Hypothesis: `int8_all` gain vs `int8_routed` comes from shared/dense path; routed stage timings should remain similar.
- Command:
  - `ssh dev-tpu-codex-int8 'cd ~/marin; for q in none int8_routed int8_all; do ... --bench-pass forward --parallel-mode none --stage-timing --topk 8 --shared-expert-dim 5632 --shared-fused --quant-mode "$q" ...; done'`
- Result:
  - forward TF/s:
    - none `266.438`
    - int8_routed `234.537`
    - int8_all `249.007`
  - stage seconds (pack/up/down/combine):
    - int8_routed: `0.004229 / 0.016332 / 0.008107 / 0.004785`
    - int8_all: `0.004216 / 0.016313 / 0.008123 / 0.004777`
  - remote log: `.agents/logbooks/moe_ep_int8_stage2_20260210_211520.log`
- Interpretation:
  - Routed-stage timings are effectively unchanged between `int8_routed` and `int8_all`.
  - Incremental gain comes from shared/dense work outside pack/up/down/combine decomposition.
- Next action:
  - Prefer `int8_all` as best-performing mode on shared-heavy configuration; keep `int8_routed` for shared-off parity.

### 2026-02-10 13:01 - MOE-I8-016 `/dev/vfio` contention handling attempt (confidence: operational)
- Hypothesis: persistent TPU busy errors can be cleared by identifying and killing file-handle owners.
- Command:
  - `sudo lsof /dev/vfio/*`
  - `sudo fuser -v /dev/vfio/{0,1,2,3,vfio}`
  - retry benchmark launches.
- Result:
  - Repeated failures: `TPU initialization failed ... /dev/vfio/* Device or resource busy`.
  - `lsof`/`fuser` showed no visible user-space holders at failure time.
- Interpretation:
  - Contention appears to be from non-observable or transient runtime ownership, not a stable user-space FD owner.
- Next action:
  - Re-allocate fresh TPU host and rerun (successful in MOE-I8-012+).

## Current Status (Updated 2026-02-10 13:16)
- `quant_mode=int8` (dense-only) gives modest positive lift on shared-heavy shape (`~1-3%`), but not the largest gain path.
- `quant_mode=int8_routed` gives strong lift across tested EP shapes:
  - `+5.9%` (topk2, shared=5632), `+13.5%` (topk8, shared=5632)
  - `+10.3%` (topk2, shared=0), `+21.1%` (topk8, shared=0)
  - `+36.5%` (3072/3072, shared=0, topk8)
- `quant_mode=int8_all` (routed + dense/shared int8) is the best mode on shared-heavy shape:
  - `+8.6%` (topk2, shared=5632), `+16.2%` (topk8, shared=5632)
  - effectively neutral vs `int8_routed` when `shared_expert_dim=0`.

### 2026-02-10 13:24 - MOE-I8-017 No-shared bottleneck diagnosis + required changes (confidence: actionable)
- Question: why does no-shared path not get *additional* lift beyond `int8_routed`/`int8_all` despite being HBM-bound?
- Key code facts:
  - routed int8 currently quantizes rhs weights *inside each expert matmul call*:
    - `rhs_q, rhs_scale = _symmetric_int8_quantize(rhs, axis=(1, 2))`
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `int8_all` adds dense/shared int8 only; when `shared_expert_dim=0`, this path is inactive.
- Implication:
  - For no-shared fused-w13 config (`experts=64, hidden=2048, mlp=2048`), expert weights are:
    - `w13`: ~1.0 GiB bf16
    - `w2`: ~0.5 GiB bf16
    - total bf16 read baseline: ~1.5 GiB
  - On-the-fly int8 quantization effectively touches (approx) read bf16 + write int8 + read int8:
    - ~3.0 GiB weight traffic equivalent per pass,
    - so it cannot realize the expected HBM savings from int8 weight residency.
- Why `int8_all` is neutral in no-shared:
  - It only changes shared/dense path, which is absent at `shared_expert_dim=0`.
  - No-shared throughput is dominated by routed expert path, so only routed-kernel/dataflow changes matter.
- What we need to do next (required for no-shared lift):
  1. **Persistent int8 expert weights**:
     - keep `w13_q/w2_q` (+ scales) resident and consumed directly by routed kernel,
     - avoid per-step on-the-fly rhs quantization in hot path.
  2. **Fused routed kernel dataflow**:
     - fuse quant/dequant/pack/combine boundaries so we do not materialize large intermediates (`out_repeat_sort`) in HBM between down-proj and combine.
  3. **Training-compatible backward**:
     - add custom VJP for fused routed int8 path so gradients do not backprop through round/clip graph and do not force expensive fallback.
  4. **Adopt existing quantized fused-MoE kernel patterns**:
     - `lib/levanter/src/levanter/kernels/pallas/moe/tpu_inference_v1/kernel.py` already supports sub-channel quantized expert weights via `subc_quant_w1_sz/subc_quant_w2_sz` + `w1_scale/w2_scale`.
     - Bridge this to train-like benchmark path (forward_backward) with custom VJP wrapper.
- Immediate implementation target:
  - add a routed mode that takes pre-quantized expert weights/scales (no rhs q in hot loop),
  - benchmark no-shared topk={2,8} against current `int8_routed` to validate the expected HBM benefit.

### 2026-02-10 13:33 - MOE-I8-018 XLA dump verification of int8 paths (confidence: replicated)
- Hypothesis: current HLO matches expected mode split:
  - `int8`: dense/shared AQT int8 only.
  - `int8_routed`: routed expert matmuls int8 + dense/shared fp32.
  - `int8_all`: both routed int8 and dense/shared AQT int8.
- Command:
  - `XLA_FLAGS='--xla_dump_to=/tmp/xla_moe_i8_20260210/<mode> --xla_dump_hlo_as_text ...' .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py ... --quant-mode <mode>`
  - modes: `none`, `int8`, `int8_routed`, `int8_all`
- Config:
  - `tokens=256 hidden=64 mlp_dim=64 experts=4 topk=2`
  - `shared_expert_dim=64 shared_fused backend=ragged_dot impl=fused_w13`
  - `bench_pass=forward parallel_mode=none dtype=float32`
- Result:
  - `none`:
    - dense/shared path appears as fp32 `dot_general` from bench source line 157.
  - `int8`:
    - dense/shared path lowers through AQT (`aqt_fwd` metadata) with `s8 -> s32 dot -> f32` scale/reduce chains.
    - routed path remains fp32.
  - `int8_routed`:
    - routed path contains `s8` tensors and `s32` accumulations sourced from bench routed matmul lines 116-122.
    - dense/shared path remains fp32 `dot_general` (bench source line 157).
  - `int8_all`:
    - includes both routed `s8/s32` chains and dense/shared `aqt_fwd` chains.
- Interpretation:
  - HLO lowering matches intended mode semantics; there is no evidence of accidental dense-int8 in `int8_routed` or missing routed-int8 in `int8_all`.
- Next action:
  - verify that a pre-quantized routed-weight mode removes rhs quantization ops from HLO.

### 2026-02-10 13:36 - MOE-I8-019 Add pre-quantized routed-weight mode (confidence: actionable)
- Hypothesis: a mode that feeds persistent `w*_q/w*_scale` into routed matmuls will eliminate on-the-fly rhs quantization in the hot path.
- Code change:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - new `quant_mode`: `int8_routed_prequant`
  - threaded optional `rhs_q/rhs_scale` through `_expert_matmul_int8_routed` / `_expert_matmul` / `_moe_from_dispatch`
  - pre-quantized expert weight bundle created once per benchmark setup via `_prequantize_expert_weights(...)`
  - mode guard: `int8_routed_prequant` currently supports `bench_pass=forward` only
- Local validation:
  - script runs for `parallel_mode=none` and `parallel_mode=ep` on CPU.
  - `forward_backward + int8_routed_prequant` intentionally raises clear `ValueError`.
- HLO check command:
  - dumped `int8_routed` vs `int8_routed_prequant` to `/tmp/xla_moe_i8_preq_20260210/*`
- HLO result:
  - `int8_routed` includes rhs-quant rounds for expert weight shapes (`round` on `[4,64,64]` and `[4,64,128]`).
  - `int8_routed_prequant` removes those rhs-weight round chains; only lhs activation quantization round chains remain.
- Interpretation:
  - implementation removes routed rhs quantization from the steady-state forward graph as intended.
- Next action:
  - benchmark no-shared EP throughput on v5p.

### 2026-02-10 13:39 - MOE-I8-020 v5p no-shared EP benchmark for prequant mode (confidence: exploratory)
- Hypothesis: if rhs quantization is removed from the hot loop, `int8_routed_prequant` should outperform both `none` and `int8_routed` on no-shared HBM-bound shapes.
- Command:
  - allocate:
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name codex-int8 allocate --tpu-type v5p-8`
  - benchmark:
    - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name codex-int8 execute -- bash -lc 'cd ~/marin && for q in none int8_routed int8_routed_prequant; do for tk in 2 8; do ... --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --shared-expert-dim 0 --backend gmm --impl fused_w13 --parallel-mode ep --bench-pass forward --quant-mode \"$q\" --topk \"$tk\" --iters 3 --warmup 1; done; done'`
- Config:
  - `tokens=32768 hidden=2048 mlp_dim=2048 experts=64 shared=0`
  - `topk in {2,8}`, `distribution=random`, `dtype=bfloat16`
  - `backend=gmm impl=fused_w13 parallel_mode=ep bench_pass=forward`
- Result:
  - topk=2:
    - none: `time_s=0.017877`, `tokens_per_s=1,832,938`
    - int8_routed: `time_s=0.017994`, `tokens_per_s=1,821,074`
    - int8_routed_prequant: `time_s=0.013326`, `tokens_per_s=2,458,984`
    - delta: `+35.0%` vs `int8_routed`, `+34.2%` vs `none` (time-based)
  - topk=8:
    - none: `time_s=0.024716`, `tokens_per_s=1,325,795`
    - int8_routed: `time_s=0.025323`, `tokens_per_s=1,294,021`
    - int8_routed_prequant: `time_s=0.020635`, `tokens_per_s=1,587,972`
    - delta: `+22.7%` vs `int8_routed`, `+19.8%` vs `none` (time-based)
- Interpretation:
  - no-shared path now gets the expected lift once rhs quantization is removed from steady-state routed compute.
  - previous no-shared plateau was primarily an artifact of on-the-fly weight quantization traffic.
- Next action:
  - extend this path with a training-compatible backward strategy (custom VJP/STE) to evaluate `forward_backward` apples-to-apples.

### 2026-02-10 13:40 - MOE-I8-021 Allocation actor termination note (confidence: operational)
- Observation:
  - after benchmark completion, the allocation actor later died with:
    - `ray.exceptions.ActorDiedError ... TPUAllocationActor ... node was terminated expectedly: received SIGTERM`
  - `dev-tpu-codex-int8` SSH alias was removed automatically by the tool.
- Interpretation:
  - TPU was valid for the benchmark window and results above were collected before actor termination.
- Next action:
  - re-allocate a fresh dev TPU for further sweeps/replication.

## Current Status (Updated 2026-02-10 13:40)
- The HLO/XLA path is now validated:
  - `int8` = dense/shared AQT int8 only,
  - `int8_routed` = routed int8 only,
  - `int8_all` = routed + dense/shared int8.
- `int8_routed_prequant` is implemented and removes routed rhs quantization from the forward hot path.
- On v5p no-shared EP forward (`32768 x 2048 x 2048`, experts=64):
  - topk=2: `+35.0%` vs `int8_routed`
  - topk=8: `+22.7%` vs `int8_routed`
- Remaining gap:
  - training-like `forward_backward` for prequant mode is not yet supported; requires a backward strategy (custom VJP/STE) for fair train-path comparison.

### 2026-02-10 14:02 - MOE-I8-022 First training-compatible prequant backward (confidence: exploratory)
- Hypothesis: custom VJP can unlock `forward_backward` for `int8_routed_prequant` while preserving forward no-shared lift.
- Code change:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - added `_expert_matmul_int8_routed_ste` custom VJP and enabled `bench_pass=forward_backward` for `int8_routed_prequant`.
- Command:
  - `... bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --shared-expert-dim 0 --backend gmm --impl fused_w13 --parallel-mode ep --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --quant-mode {none,int8_routed,int8_routed_prequant} --topk {2,8} --iters 3 --warmup 1`
- Result:
  - topk=2: none `0.027212`, int8_routed `0.024840`, prequant `0.028019`
  - topk=8: none `0.044622`, int8_routed `0.037024`, prequant `0.046594`
- Interpretation:
  - first custom VJP erased forward gains and regressed train-like runtime.
  - likely cause: backward pullback path still expensive and requant-heavy.
- Next action:
  - reduce backward overhead by reusing pre-quantized rhs in `grad_lhs`.

### 2026-02-10 14:14 - MOE-I8-023 Backward revision: prequant rhs reuse for `grad_lhs` (confidence: exploratory)
- Hypothesis: if backward `grad_lhs` consumes transposed pre-quantized rhs, runtime should recover versus MOE-I8-022.
- Code change:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - custom VJP residual now carries `rhs_q/rhs_scale`
  - `grad_lhs` computed via `_expert_matmul_int8_routed(grad_out, rhs^T, ..., rhs_q=rhs_q^T, rhs_scale=rhs_scale)`
  - `grad_rhs` remained via routed-int8 surrogate pullback.
- Command:
  - same v5p no-shared EP matrix as MOE-I8-022 on fresh allocation (`codex-int8b`).
- Result:
  - topk=2: none `0.027366`, int8_routed `0.024985`, prequant `0.026295`
  - topk=8: none `0.044690`, int8_routed `0.037017`, prequant `0.040893`
- Interpretation:
  - prequant became faster than `none`, but still behind `int8_routed`.
- Next action:
  - optimize `grad_rhs` pullback path (current surrogate still too expensive).

### 2026-02-10 14:29 - MOE-I8-024 Backward revision: `grad_rhs` surrogate via `gmm_sharded` (confidence: exploratory)
- Hypothesis: replacing `grad_rhs` routed-int8 surrogate pullback with `gmm_sharded` pullback will reduce train-like overhead.
- Code change:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - in `_expert_matmul_int8_routed_ste_bwd`, changed:
    - from routed-int8 surrogate VJP for rhs
    - to `jax.vjp(lambda rhs: gmm_sharded(lhs, rhs, group_sizes), rhs)` for rhs gradient
  - retained prequant-rhs reuse for `grad_lhs`.
- Command:
  - `ssh dev-tpu-codex-int8 '... --quant-mode int8_routed_prequant --topk-list 2,8 --bench-pass forward_backward --parallel-mode ep ...'`
- Result:
  - topk=2: prequant `0.018678`
  - topk=8: prequant `0.036637`
- Interpretation:
  - major throughput recovery; prequant now clearly ahead of `none` and competitive/better vs prior routed-int8 baseline values.
- Next action:
  - rerun full matrix (`none`, `int8_routed`, `int8_routed_prequant`) on same host for apples-to-apples confirmation.

### 2026-02-10 14:34 - MOE-I8-025 v5p replicated matrix with final backward rule (confidence: replicated)
- Hypothesis: with `grad_lhs` prequant-rhs reuse + `grad_rhs` gmm surrogate, prequant should beat both baselines in no-shared `forward_backward`.
- Command:
  - `ssh dev-tpu-codex-int8 '... --quant-mode none ... && ... --quant-mode int8_routed ... && ... --quant-mode int8_routed_prequant ...'`
  - fixed config:
    - `tokens=32768 hidden=2048 mlp_dim=2048 experts=64 shared=0`
    - `topk in {2,8}`, `distribution=random`, `dtype=bfloat16`
    - `backend=gmm impl=fused_w13 parallel_mode=ep bench_pass=forward_backward`
- Result:
  - topk=2:
    - none `0.027254`
    - int8_routed `0.024851`
    - int8_routed_prequant `0.018549`
    - prequant delta: `+46.9%` vs none, `+34.0%` vs int8_routed (time-based)
  - topk=8:
    - none `0.044957`
    - int8_routed `0.037052`
    - int8_routed_prequant `0.036706`
    - prequant delta: `+22.5%` vs none, `+0.9%` vs int8_routed (time-based)
- Interpretation:
  - final prequant train-like path is now faster than both comparison modes on no-shared v5p case.
  - especially strong win at topk=2; small but consistent win at topk=8.
- Next action:
  - evaluate scale granularity variants (per-token activation scale + single expert-weight scale baseline already in place; try finer sub-channel only if quality/accuracy requires).

### 2026-02-10 14:40 - MOE-I8-026 XLA dump sanity for final backward variant (confidence: exploratory)
- Hypothesis: final prequant backward should show `gmm`-based rhs-gradient path and avoid the earlier expensive routed-int8 rhs pullback behavior.
- Commands:
  - TPU attempt (killed by resource limits):
    - `XLA_FLAGS='--xla_dump_to=/tmp/xla_moe_i8_fb_cmp/<mode> --xla_dump_hlo_as_text' ... --bench-pass forward_backward ...`
  - fallback tiny CPU structural dump:
    - `XLA_FLAGS='--xla_dump_to=/tmp/xla_moe_i8_fb_cpu/<mode> --xla_dump_hlo_as_text' JAX_PLATFORMS=cpu ... --tokens 256 --hidden 128 --mlp-dim 128 --experts 8 --topk 2 --bench-pass forward_backward --quant-mode <mode>`
- Result:
  - TPU full-shape dump process was killed (`code 137`) before completion.
  - tiny CPU dump includes `tgmm` call-sites in prequant backward graph at source lines corresponding to the new rhs-gradient surrogate path.
- Interpretation:
  - structural evidence is consistent with the intended backward rewrite; full TPU dump still pending for high-fidelity kernel-level confirmation.
- Next action:
  - if needed, run targeted TPU dump with stricter dump filters/pass regex to avoid resource kill.

### 2026-02-10 14:34 - MOE-I8-027 Allocation teardown instability note (confidence: operational)
- Observation:
  - while terminating the long-lived `allocate` session after experiments, the `TPUAllocationActor` died with heartbeat-miss/node-healthcheck errors and removed SSH alias `dev-tpu-codex-int8`.
- Interpretation:
  - benchmark outputs above were already collected before actor failure.
  - this is an ops stability issue in the dev TPU allocation monitor path, not a benchmark correctness issue.
- Next action:
  - continue using short benchmark windows or reallocate fresh aliases when actor heartbeats become noisy.

## Current Status (Updated 2026-02-10 14:40)
- `int8_routed_prequant` now supports `forward_backward` with a custom VJP path.
- Final no-shared v5p EP benchmark (`32768 x 2048 x 2048`, experts=64, fused_w13, gmm, bfloat16):
  - topk=2: prequant `0.018549s` (faster than none `0.027254s` and int8_routed `0.024851s`)
  - topk=8: prequant `0.036706s` (faster than none `0.044957s` and int8_routed `0.037052s`)
- The no-shared train-like throughput objective is met for this benchmark shape.
