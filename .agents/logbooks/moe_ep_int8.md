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
