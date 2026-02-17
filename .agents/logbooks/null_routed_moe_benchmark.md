# Null-Routed MoE Benchmark: Research Logbook

## Scope
- Goal: measure performance impact of null routing in MoE blocks, where tokens can route to a "none" slot instead of a real expert.
- Primary metrics: `forward_backward` TF/s, tokens/s, and deltas vs `null_route_frac=0.0` baseline.
- Constraints: use the issue `#2704/#2710` hillclimb harness path and the null-expert routing approach from `origin/will/null-moe`.

## Links
- Experiment issue: https://github.com/marin-community/marin/issues/2844
- Prior issue (hillclimb harness): https://github.com/marin-community/marin/issues/2704
- Prior issue (EP benchmark): https://github.com/marin-community/marin/issues/2710
- Harness: `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
- Null routing block source: `experiments/speedrun/custom_mixtral.py`

## Baseline
- Date: 2026-02-17
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `experiments/speedrun/custom_mixtral.py`
- Baseline definition:
  - `null_route_frac=0.0`
  - compare against `null_route_frac in {0.1, 0.2, ..., 0.9}`

## Stop Criteria
- Null-routing sweep runs end-to-end for each shape family.
- We have per-shape tables for `null_route_frac=0.1..0.9` and a clear trend summary.
- We can state whether null routing improves throughput at fixed model shape, and how that changes with top-k / experts.

## Experiment Matrix (Initial)
- Axis A: null route fraction
  - `0.1, 0.2, ..., 0.9`
- Axis B: model size / shape
  - Shape S: `tokens=32768 hidden=2048 mlp_dim=1408 experts=60`
  - Shape M: `tokens=32768 hidden=2048 mlp_dim=2048 experts=64`
  - Shape L: `tokens=65536 hidden=3072 mlp_dim=3072 experts=128`
- Axis C: routing granularity (`topk`)
  - `2, 4, 8`
- Axis D: pass mode
  - `forward_backward` primary, `forward` optional sanity

## Command Templates
- Baseline (`null=0.0`):
```bash
uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py \
  --distribution random \
  --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk 4 \
  --backend gmm --impl fused_w13 --bench-pass forward_backward \
  --parallel-mode ep --queue-mode full --iters 3 --warmup 1 \
  --null-route-frac 0.0 --renormalize-real-after-null
```

- Null sweep (`0.1..0.9`):
```bash
uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py \
  --distribution random \
  --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk 4 \
  --backend gmm --impl fused_w13 --bench-pass forward_backward \
  --parallel-mode ep --queue-mode full --iters 3 --warmup 1 \
  --null-route-sweep --renormalize-real-after-null
```

- Multi-shape / multi-topk sweep shell:
```bash
set -euo pipefail
for shape in \
  "32768 2048 1408 60" \
  "32768 2048 2048 64" \
  "65536 3072 3072 128"; do
  read -r TOK H MLP E <<<"${shape}"
  for TK in 2 4 8; do
    uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py \
      --distribution random \
      --tokens "${TOK}" --hidden "${H}" --mlp-dim "${MLP}" --experts "${E}" --topk "${TK}" \
      --backend gmm --impl fused_w13 --bench-pass forward_backward \
      --parallel-mode ep --queue-mode full --iters 3 --warmup 1 \
      --null-route-sweep --renormalize-real-after-null
  done
done
```

## Experiment Log
### 2026-02-17 14:20 - Kickoff
- Hypothesis: increasing null-routing fraction reduces routed expert compute and increases throughput, with stronger gains at larger top-k.
- Command: N/A (setup and harness extension)
- Config: added null-routing controls to hillclimb harness and imported null-expert MoE block implementation.
- Result: harness supports `--null-route-frac`, `--null-route-frac-list`, and `--null-route-sweep` with machine-readable `RESULT` lines.
- Interpretation: setup complete; ready for TPU matrix runs.
- Next action: run shape/topk/null sweeps on dev TPU and append results.

### 2026-02-17 14:55 - TPU `gmm` sweep complete (`null=0.1..0.9`)
- Command:
```bash
RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py \
  --config infra/marin-us-central1.yaml \
  --tpu-name dlwh-null-moe-133632 execute --no-sync \
  -- /bin/bash -lc '
set -euo pipefail
OUT=.agents/logbooks/null_routed_moe_results_$(date +%Y%m%d_%H%M%S).log
mkdir -p .agents/logbooks
: > "$OUT"
for shape in "32768 2048 1408 60" "32768 2048 2048 64" "65536 3072 3072 128"; do
  read -r TOK H MLP E <<<"$shape"
  for TK in 2 4 8; do
    echo "RUN shape=$TOK,$H,$MLP,$E topk=$TK" | tee -a "$OUT"
    uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py \
      --distribution random \
      --tokens "$TOK" --hidden "$H" --mlp-dim "$MLP" --experts "$E" --topk "$TK" \
      --backend gmm --impl fused_w13 --bench-pass forward_backward \
      --parallel-mode ep --queue-mode full --iters 3 --warmup 1 \
      --null-route-sweep --renormalize-real-after-null | tee -a "$OUT"
  done
done'
```
- Artifacts:
  - Raw log: `.agents/logbooks/null_routed_moe_results_20260217_213956.log`
  - Parsed CSV (`shape,topk,null_target,null_realized,tflops,tokens_per_s`): `.agents/logbooks/null_routed_moe_results_20260217_213956.csv`
- Sweep coverage:
  - 3 shapes x 3 `topk` x 9 null fractions (`0.1..0.9`) = 81 `RESULT` rows (complete)
- Realized null accuracy:
  - `max |null_realized - null_target| = 0.003`
  - `mean |null_realized - null_target| = 0.000568`

#### End-to-end delta (`null=0.1` -> `0.9`)
| Shape | topk | tokens/s @0.1 | tokens/s @0.9 | delta tokens/s | TF/s @0.1 | TF/s @0.9 | delta TF/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `32768x2048x1408x60` | 2 | 1,572,435 | 2,001,485 | +27.29% | 146.826 | 20.886 | -85.77% |
| `32768x2048x1408x60` | 4 | 1,268,519 | 1,905,736 | +50.23% | 236.764 | 39.768 | -83.20% |
| `32768x2048x1408x60` | 8 | 920,847 | 1,833,742 | +99.14% | 343.774 | 76.137 | -77.85% |
| `32768x2048x2048x64` | 2 | 1,341,959 | 1,653,547 | +23.22% | 182.263 | 25.099 | -86.23% |
| `32768x2048x2048x64` | 4 | 1,110,265 | 1,588,905 | +43.11% | 301.420 | 48.228 | -84.00% |
| `32768x2048x2048x64` | 8 | 821,075 | 1,540,460 | +87.62% | 445.857 | 93.032 | -79.13% |
| `65536x3072x3072x128` | 2 | 665,998 | 869,324 | +30.53% | 203.410 | 29.685 | -85.41% |
| `65536x3072x3072x128` | 4 | 533,793 | 835,862 | +56.59% | 326.091 | 56.790 | -82.58% |
| `65536x3072x3072x128` | 8 | 383,765 | 771,626 | +101.07% | 469.213 | 104.977 | -77.63% |

- Interpretation:
  - Tokens/s increases monotonically with null fraction in all 9 shape/topk combinations.
  - Gain from null routing is much stronger at larger `topk` (largest wins at `topk=8`).
  - Reported TF/s decreases as null fraction rises, consistent with less real expert compute being executed.

### 2026-02-17 14:42 - Added full `null=0.0` baseline column (all shape/topk points)
- Command:
```bash
RAY_AUTH_MODE=token uv run --python 3.11 scripts/ray/dev_tpu.py \
  --config infra/marin-us-central1.yaml \
  --tpu-name dlwh-null-moe-133632 execute --no-sync \
  -e LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=50000' \
  -- /bin/bash -lc '
set -euo pipefail
OUT=.agents/logbooks/null_routed_moe_baseline0_$(date +%Y%m%d_%H%M%S).log
mkdir -p .agents/logbooks
: > "$OUT"
for shape in "32768 2048 1408 60" "32768 2048 2048 64" "65536 3072 3072 128"; do
  read -r TOK H MLP E <<<"$shape"
  for TK in 2 4 8; do
    uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py \
      --distribution random \
      --tokens "$TOK" --hidden "$H" --mlp-dim "$MLP" --experts "$E" --topk "$TK" \
      --backend gmm --impl fused_w13 --bench-pass forward_backward \
      --parallel-mode ep --queue-mode full --iters 3 --warmup 1 \
      --null-route-frac 0.0 --renormalize-real-after-null | tee -a "$OUT"
  done
done'
```
- Artifacts:
  - Baseline raw log: `.agents/logbooks/null_routed_moe_baseline0_20260217_223128.log`
  - Baseline parsed CSV: `.agents/logbooks/null_routed_moe_baseline0_20260217_223128.csv`
  - Combined CSV (`null=0.0..0.9`, 90 rows): `.agents/logbooks/null_routed_moe_results_with_null0_20260217_223128.csv`

#### Requested metric: `topk=8`, `null=0.1` vs `null=0.0`
| Shape | tokens/s @0.0 | tokens/s @0.1 | gain |
| --- | ---: | ---: | ---: |
| `32768x2048x1408x60` | 858,766 | 920,847 | +7.23% |
| `32768x2048x2048x64` | 765,115 | 821,075 | +7.31% |
| `65536x3072x3072x128` | 354,834 | 383,765 | +8.15% |

- Aggregate for `topk=8`:
  - Arithmetic mean gain across shapes: `+7.57%`
  - Geometric mean gain across shapes: `+7.56%`

### 2026-02-17 14:50 - Added variable experts-per-token mode (`take_until_null`)
- Goal:
  - Support per-token expert count in `[0, topk]` instead of fixed `topk` with independent null assignment.
  - Operational semantics: keep routed experts until the first null, then route remaining suffix slots to null.
- Code changes:
  - `experiments/speedrun/custom_mixtral.py`
    - Added `MixtralConfig.take_until_null: bool = False`.
    - In `MixtralSparseMoeBlock._route`, when null routing is active and `take_until_null=True`, real experts after the first selected null are converted to null slots and zero-weighted before renormalization.
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - Added `--null-routing-mode {independent,take_until_null}` (default `independent`).
    - Added `null_mode=...` to machine-readable `RESULT` line.
- Smoke validation (CPU, `gmm`, tiny shape):
  - `--null-routing-mode take_until_null --null-route-frac 0.5`:
    - `null_realized=0.495`, `null_assignments=507`, `real_assignments=517`
  - `--null-routing-mode independent --null-route-frac 0.5`:
    - `null_realized=0.501`, `null_assignments=513`, `real_assignments=511`
- Next action:
  - Run TPU matrix comparing `independent` vs `take_until_null` across the same shape/topk/null grid.
