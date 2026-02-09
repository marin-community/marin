# MoE Hillclimb Harness (2026-02-08)

## Goal
Build a simple inline MoE block benchmark that supports routing distributions and lets us hillclimb implementation choices quickly.

## Harness
File: `lib/levanter/scripts/bench/bench_moe_hillclimb.py`

Implemented:
- Routing distributions:
  - `random`
  - `runs` (stochastic run-length routing with load-aware balancing)
- Backends:
  - `gmm`
  - `ragged_dot`
- MoE impl variants:
  - `baseline`: separate `w1` + `w3`, unpermute with inverse-argsort
  - `fused_w13`: fused `w1/w3` projection via single `w13` grouped matmul, baseline unpermute
  - `preweight`: apply top-k weights before down projection (`w2`) and sum unweighted after unpermute
  - `fused_w13_preweight`: `fused_w13` + `preweight`
  - `scatter`: separate `w1/w3`, scatter-add unpermute
  - `fast`: fused `w13` + scatter-add unpermute
- Diagnostics:
  - top-1 run statistics (`mean/p95/max run len`)
  - expert load statistics (`min/max/std/cv`)

## TPU setup notes
- `dlwh-codex1` (central1) was reachable.
- `dlwh-codex2` SSH alias was not configured in this session.
- Command uses:
  - `-e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000"`

## Results

### Shape A (Qwen3-30B-A3B style MoE)
Params:
- `tokens=32768`
- `hidden=2048`
- `mlp_dim=768` (`moe_intermediate_size`)
- `experts=128`
- `topk=8`
- backend: `gmm`
- iters/warmup: `3/1`

Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute \
  -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --tokens 32768 --hidden 2048 --mlp-dim 768 --experts 128 --topk 8 \
    --backend gmm --distribution both --impl both --iters 3 --warmup 1
```

Expanded key numbers (`iters=2`, all impls):
- `random`
  - baseline: `time_s=0.023696`, `tflops=104.404`
  - fused_w13: `time_s=0.020451`, `tflops=120.968` (**best**, +15.9%)
  - preweight: `time_s=0.026206`, `tflops=94.402` (regression)
  - fused_w13_preweight: `time_s=0.022980`, `tflops=107.655` (below fused_w13)
  - scatter: `time_s=0.036677`, `tflops=67.451` (large regression)
  - fast: `time_s=0.033162`, `tflops=74.600` (regression)
- `runs`
  - baseline: `time_s=0.023784`, `tflops=104.013`
  - fused_w13: `time_s=0.020327`, `tflops=121.707` (**best**, +17.0%)
  - preweight: `time_s=0.026611`, `tflops=92.965` (regression)
  - fused_w13_preweight: `time_s=0.023425`, `tflops=105.608` (near baseline)
  - scatter: `time_s=0.037013`, `tflops=66.839` (large regression)
  - fast: `time_s=0.033620`, `tflops=73.584` (regression)

Interpretation:
- For this shape, **fusing `w1/w3` into one grouped matmul is a clear win**.
- `preweight` (moving top-k weighting before `w2`) did not help.
- Scatter-add unpermute is consistently worse than inverse-argsort unpermute.

### Shape B (Qwen1.5-MoE-A2.7B style MoE)
Params:
- `tokens=32768`
- `hidden=2048`
- `mlp_dim=1408` (`moe_intermediate_size`)
- `experts=60`
- `topk=4`
- backend: `gmm`
- iters/warmup: `3/1`

Commands:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute \
  -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk 4 \
    --backend gmm --distribution both --impl baseline --iters 3 --warmup 1
```

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute \
  -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk 4 \
    --backend gmm --distribution both --impl fused_w13 --iters 3 --warmup 1
```

Key numbers:
- `random`
  - baseline: `time_s=0.016991`, `tflops=133.465`
  - fused_w13: `time_s=0.017081`, `tflops=132.762` (~flat / slight regression)
- `runs`
  - baseline: `time_s=0.016982`, `tflops=133.541`
  - fused_w13: `time_s=0.017020`, `tflops=133.243` (~flat / slight regression)

Interpretation:
- For this shape with `gmm`, `fused_w13` is not beneficial.

### GMM vs ragged_dot comparison
Same shapes, same distributions.

Shape A (`H=2048, M=768, E=128, topk=8`):
- `gmm baseline`: ~`104.0-104.4` TF/s
- `gmm fused_w13`: ~`121.0-121.7` TF/s (**best**)
- `ragged_dot baseline`: ~`94.5-94.7` TF/s
- `ragged_dot fused_w13`: ~`110.4-110.7` TF/s
- Gap at best: `gmm fused_w13` is about **9-10% faster** than `ragged_dot fused_w13`.

Shape B (`H=2048, M=1408, E=60, topk=4`):
- `gmm baseline`: ~`133.5` TF/s
- `gmm fused_w13`: ~`132.8-133.2` TF/s
- `ragged_dot baseline`: ~`87.1-87.6` TF/s
- `ragged_dot fused_w13`: ~`111.7-112.0` TF/s
- Gap at best: `gmm` is about **19% faster** than `ragged_dot`.

### Shared expert (Shape B)
Params:
- `tokens=32768`
- `hidden=2048`
- `mlp_dim=1408`
- `experts=60`
- `topk=4`
- `shared_expert_dim=5632`
- backend: `gmm`
- impl: `fused_w13`
- iters/warmup: `2/1`

Command:
```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute \
  -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk 4 \
    --backend gmm --distribution both --impl fused_w13 --shared-expert-dim 5632 --iters 2 --warmup 1
```

Key numbers:
- `shared_fused=False`
  - random: `time_s=0.022201`, `tflops=204.294`
  - runs: `time_s=0.022139`, `tflops=204.859`
- `shared_fused=True`
  - random: `time_s=0.022275`, `tflops=203.614`
  - runs: `time_s=0.022426`, `tflops=202.243`

Interpretation:
- Shared expert adds substantial compute density for this shape.
- Fusing shared `w1/w3` was slightly worse here; keep `shared_fused=False`.

### Top-k falloff with shared expert (`topk=1,2,4,6,8`)
Shape:
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `experts=60`, `shared_expert_dim=5632`
- impl: `fused_w13`
- iters/warmup: `2/1`

`gmm` (random / runs):
- `topk=1`: `265.444 / 265.336` TF/s
- `topk=2`: `225.347 / 225.724` TF/s
- `topk=4`: `205.303 / 205.070` TF/s
- `topk=6`: `194.461 / 194.570` TF/s
- `topk=8`: `194.702 / 194.981` TF/s

`ragged_dot` (random / runs):
- `topk=1`: `238.954 / 238.378` TF/s
- `topk=2`: `200.757 / 200.350` TF/s
- `topk=4`: `178.162 / 178.501` TF/s
- `topk=6`: `167.591 / 167.635` TF/s
- `topk=8`: `165.857 / 166.013` TF/s

`gmm` over `ragged_dot` advantage (random):
- `topk=1`: ~`+11.1%`
- `topk=2`: ~`+12.2%`
- `topk=4`: ~`+15.2%`
- `topk=6`: ~`+16.0%`
- `topk=8`: ~`+17.4%`

Interpretation:
- Throughput declines sharply from `topk=1 -> 2 -> 4`, then flattens at `6/8`.
- With shared expert enabled, incremental routed-expert overhead at high `topk` is partially amortized by fixed dense shared compute.
- `gmm` stays ahead of `ragged_dot` across the full top-k sweep and the gap widens with larger `topk`.

## Ideas cribbed from references
Looked at:
- `jax-llm-examples/gpt_oss/gpt_oss_jax/model.py`
- `MaxText/layers/moe.py`

Tried from those patterns:
- **Fusing gate/up projections (`w1`/`w3`)** into one grouped matmul:
  - helps on shape A, not on shape B.
- **Pre-applying router weights** before `w2`:
  - regression in this harness.
- **Scatter-style combine path**:
  - regression in this harness.

Not implemented (yet):
- Custom VJP for sort/unsort (MaxText pattern): this benchmark is forward-only, so no expected runtime gain here.
- Capacity/padding policy and token-drop policy variants: would require extending benchmark semantics beyond current exact top-k combine.

## Current conclusion
- Keep `scatter` path disabled for this workload shape family.
- `fused_w13` is **shape-dependent**:
  - Strong win for `E=128, topk=8, M=768`.
  - Neutral/slightly negative for `E=60, topk=4, M=1408`.
- `gmm` is consistently faster than `ragged_dot` on tested shapes.
- For the shared-expert shape (`H=2048, M=1408, E=60, shared=5632`), `gmm + fused_w13 + shared_fused=False` is the strongest tested option.

## Backward-inclusive benchmark (item 6)
Goal: compare forward-only vs train-like (`forward+backward`) throughput on the same routed MoE block.

Notes:
- This run used fixed routing (`topk_idx/topk_weights` sampled once), so it measures MoE MLP train-step math cost (not router-backprop dynamics).
- Forward+backward FLOPs are accounted as `3x` forward FLOPs (standard approximation).
- Before running, TPU holders were stale `VLLM::EngineCore` processes in `ray_docker`; killing those cleared `/dev/vfio/*` lockups.

Shape:
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `experts=60`, `topk=4`
- `impl=fused_w13`, `shared_expert_dim=0`
- `iters=2`, `warmup=1`

### gmm
- forward:
  - random: `time_s=0.015738`, `tflops=144.092`
  - runs: `time_s=0.015758`, `tflops=143.914`
- forward_backward:
  - random: `time_s=0.054037`, `tflops=125.899`
  - runs: `time_s=0.054117`, `tflops=125.714`

### ragged_dot
- forward:
  - random: `time_s=0.019030`, `tflops=119.167`
  - runs: `time_s=0.019075`, `tflops=118.888`
- forward_backward:
  - random: `time_s=0.069607`, `tflops=97.738`
  - runs: `time_s=0.069599`, `tflops=97.749`

### Takeaways from backward-inclusive run
- `gmm` remains ahead in train-like mode:
  - forward: `144.1 / 119.2` => ~`+20.9%` (`gmm` over `ragged_dot`)
  - forward_backward: `125.9 / 97.7` => ~`+28.8%`
- Relative to forward-only, train-like mode drops effective TF/s:
  - `gmm`: `144.1 -> 125.9` (~`12.6%` drop)
  - `ragged_dot`: `119.2 -> 97.7` (~`18.0%` drop)
- The gap widening in backward indicates `gmm` gradient path is materially stronger than `ragged_dot` gradient path at this shape.

## Additional sweeps (forward_backward unless noted)

Shape unless noted:
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `topk=2`, `impl=fused_w13`
- distributions: `random` and `runs` were both run; values below report `random` (runs was effectively identical)

### Expert-count sweep (`topk=2`)

`gmm` vs `ragged_dot` TF/s:
- `E=32`: `119.405` vs `93.607`
- `E=64`: `105.745` vs `80.677`
- `E=128`: `86.808` vs `63.034`
- `E=256`: `63.486` vs `43.824`

Takeaway:
- Throughput falls with larger expert count at fixed token budget.
- `gmm` stays ahead and relative advantage widens as `E` increases.

### Expert-count sweep (`topk=2`, forward-only)

Shape:
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `impl=fused_w13`, distribution=`random`

`gmm` vs `ragged_dot` TF/s:
- `E=32`: `134.609` vs `112.353`
- `E=64`: `119.923` vs `98.992`
- `E=128`: `99.395` vs `80.326`
- `E=256`: `73.714` vs `57.832`

Takeaway:
- Same scaling trend as forward_backward: expert growth hurts throughput at fixed tokens.
- `gmm` remains consistently faster, by roughly `~24-27%` across this sweep.

### Token-count sweep (`E=64`, `topk=2`)

`gmm` vs `ragged_dot` TF/s:
- `tokens=8192`: `62.008` vs `43.794`
- `tokens=16384`: `85.652` vs `62.606`
- `tokens=32768`: `105.781` vs `80.501`
- `tokens=65536`: `120.863` vs `94.391`

Takeaway:
- Both backends scale up with more tokens.
- `gmm` lead remains consistent.

### Stage timing breakdown (`forward`, `E=64`, `topk=2`)

`gmm` (`time_s=0.009439`):
- `pack`: `0.001205s` (`11.9%`)
- `up`: `0.004407s` (`43.5%`)
- `down`: `0.002367s` (`23.4%`)
- `combine`: `0.002148s` (`21.2%`)

`ragged_dot` (`time_s=0.011474`):
- `pack`: `0.001213s` (`10.0%`)
- `up`: `0.005425s` (`44.8%`)
- `down`: `0.003346s` (`27.6%`)
- `combine`: `0.002132s` (`17.6%`)

Takeaway:
- `up` + `down` dominate total time.
- `combine` is material but secondary for this shape.

### Capacity policy sweep (`E=64`, `topk=2`, forward_backward)

Policy `drop`:
- `cf=0.50`: `gmm 103.947`, `ragged 79.886`, `drop_frac=0.50`
- `cf=0.75`: `gmm 103.814`, `ragged 79.631`, `drop_frac=0.25`
- `cf=1.00`: `gmm 103.865`, `ragged 79.827`, `drop_frac=0.0104`
- `cf=1.25`: `gmm 105.669`, `ragged 80.602`, `drop_frac=0`

Policy `pad`:
- `cf=0.50`: `gmm 88.964`, `ragged 71.618`, `drop_frac=0.50`
- `cf=0.75`: `gmm 93.047`, `ragged 72.912`, `drop_frac=0.25`
- `cf=1.00`: `gmm 111.462`, `ragged 89.700`, `drop_frac=0.0104`
- `cf=1.25`: `gmm 108.080`, `ragged 85.275`, `drop_frac=0`

Takeaway:
- `drop` mostly preserves runtime since tensor shapes remain dense.
- `pad` changes compute shape materially; best observed point was near `cf=1.0`.

## Practical next step
Add an auto-policy in this harness (or later in production path) that toggles `fused_w13` by shape bucket and verifies with one short warmup benchmark pass.

## Prequeue mode (pack outside timed region)

Implemented in harness:
- `--queue-mode full`: existing behavior (dispatch/pack + compute all timed together)
- `--queue-mode prequeue`: dispatch is precomputed once and excluded from timed region
- `--queue-mode both`: prints both in one run

Interpretation:
- `prequeue` is an upper bound for overlap quality (best case where packing/routing latency is hidden).
- It does **not** include the true end-to-end routing/pack cost in the timed number.

### Base shape (`E=64`, `topk=2`, no shared expert)
Shape:
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `impl=fused_w13`, distribution=`random`

`gmm`:
- `full`: `119.362` TF/s
- `prequeue`: `133.711` TF/s
- uplift: `+12.0%`

`ragged_dot`:
- `full`: `99.333` TF/s
- `prequeue`: `108.384` TF/s
- uplift: `+9.1%`

### Shared-expert top-k sweep (`E=60`, `shared_expert_dim=5632`)
Shape:
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `impl=fused_w13`, distribution=`random`

`gmm` full vs prequeue:
- `topk=1`: `273.569 -> 285.482` TF/s (`+4.4%`)
- `topk=2`: `235.765 -> 252.748` TF/s (`+7.2%`)
- `topk=4`: `216.254 -> 237.423` TF/s (`+9.8%`)
- `topk=6`: `201.221 -> 230.876` TF/s (`+14.7%`)
- `topk=8`: `208.011 -> 235.381` TF/s (`+13.2%`)

`ragged_dot` full vs prequeue:
- `topk=1`: `244.240 -> 254.504` TF/s (`+4.2%`)
- `topk=2`: `206.060 -> 221.795` TF/s (`+7.6%`)
- `topk=4`: `186.581 -> 203.754` TF/s (`+9.2%`)
- `topk=6`: `172.294 -> 194.758` TF/s (`+13.0%`)
- `topk=8`: `174.933 -> 194.028` TF/s (`+10.9%`)

Takeaway:
- Headroom from overlap grows with `topk` for both backends.
- At high `topk`, packing/dispatch overhead is substantial enough that perfect hiding can move throughput by ~`11-15%`.
- `gmm` remains faster than `ragged_dot` at all tested `topk` in both `full` and `prequeue`.

### Compile behavior note
- `--topk-list` + `--queue-mode both` triggers many compiled variants and large constant-folding work (notably on `argsort` over dispatch indices), so compile wall-time is high even when runtime numbers are good.

## Routing pack strategy comparison (`argsort` vs tuple `lax.sort`)

Added harness flag:
- `--routing-pack-strategy {argsort,tuple_sort}`

Compared two routing-pack implementations:
- `argsort`: `sort_idx = argsort(topk_idx_flat)` then `token_idx_sort = sort_idx // topk`
- `tuple_sort` (from user snippet): `lax.sort((topk_idx_flat, flat_pos, tok_id), dimension=0)`

Shape A:
- `tokens=32768, hidden=2048, mlp_dim=1408, experts=64, topk=2, backend=gmm, impl=fused_w13`
- `argsort`: `full 120.556`, `prequeue 135.192` TF/s
- `tuple_sort`: `full 116.993`, `prequeue 134.925` TF/s

Shape B:
- `tokens=32768, hidden=2048, mlp_dim=1408, experts=60, topk=8, shared_expert_dim=5632, backend=gmm, impl=fused_w13`
- `argsort`: `full 208.134`, `prequeue 235.990` TF/s
- `tuple_sort`: `full 201.158`, `prequeue 236.311` TF/s

Takeaway:
- For end-to-end (`full`), `argsort` is better on both tested shapes.
- For `prequeue`, both strategies are effectively tied.
- Conclusion: keep `argsort` as default for production-like timing runs.

## `mlp_dim` sweep (`gmm`, `argsort`, `queue_mode=both`)

Purpose:
- Check whether `mlp_dim=1408` is uniquely bad versus nearby dimensions.

Common settings:
- `tokens=32768`, `hidden=2048`, `backend=gmm`, `impl=fused_w13`, distribution=`random`

Case 1: `experts=64`, `topk=2`, no shared expert

`mlp_dim, full_tflops, prequeue_tflops, uplift_pct`
- `1024`: `129.667`, `153.487`, `18.37%`
- `1280`: `110.362`, `122.056`, `10.60%`
- `1408`: `120.037`, `133.961`, `11.60%`
- `1536`: `130.475`, `145.467`, `11.49%`
- `1792`: `151.461`, `171.643`, `13.32%`
- `2048`: `175.016`, `194.395`, `11.07%`

Case 2: `experts=60`, `topk=8`, `shared_expert_dim=5632`

`mlp_dim, full_tflops, prequeue_tflops, uplift_pct`
- `1024`: `238.545`, `284.156`, `19.12%`
- `1280`: `196.591`, `222.396`, `13.13%`
- `1408`: `208.236`, `236.072`, `13.37%`
- `1536`: `218.777`, `247.479`, `13.12%`
- `1792`: `246.514`, `278.726`, `13.07%`
- `2048`: `270.804`, `306.866`, `13.32%`

Takeaway:
- `1408` is not uniquely cursed in these runs; `1280` appears worse.
- Larger `mlp_dim` generally improves achieved TF/s at fixed tokens/expert config.
- Pack-overlap headroom remains material (`~11-19%`) across dims.

## Remat leverage (`forward_backward`, `gmm`, `argsort`, `queue_mode=full`)

Purpose:
- Test which checkpoint boundary has leverage for full train-step throughput.

Notes:
- In this harness, `prequeue` timing is only emitted in `forward` mode, not `forward_backward`.
- Therefore remat comparison is done on full end-to-end `forward_backward` only.

Settings:
- distribution=`random`
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`
- backend=`gmm`, impl=`fused_w13`, routing_pack_strategy=`argsort`
- `iters=3`, `warmup=1`

`case,remat_mode,full_tflops`
- `base_t2_e64,none,97.539`
- `base_t2_e64,expert_mlp,86.604`
- `base_t2_e64,combine,97.834`
- `shared_t8_e60,none,167.569`
- `shared_t8_e60,expert_mlp,151.972`
- `shared_t8_e60,combine,167.312`

Takeaway:
- `expert_mlp` remat is clearly negative (`~11%` down on both shapes).
- `combine` remat is effectively neutral (within noise) on these runs.
- Most-leverage conclusion for this harness: do **not** remat expert MLP if throughput is primary objective.

## Locality sensitivity (`random` vs `runs`)

Purpose:
- Validate whether stronger routed runs materially improve end-to-end throughput or overlap headroom.

Settings:
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`
- `experts=60`, `shared_expert_dim=5632`
- backend=`gmm`, impl=`fused_w13`, routing=`argsort`
- `bench_pass=forward`, `queue_mode=both`, `iters=3`, `warmup=1`

`random` distribution:
- `topk=1`: `272.446 -> 284.449` TF/s (`+4.4%`)
- `topk=2`: `229.911 -> 247.797` TF/s (`+7.8%`)
- `topk=4`: `213.154 -> 233.770` TF/s (`+9.7%`)
- `topk=6`: `199.248 -> 229.471` TF/s (`+15.2%`)
- `topk=8`: `205.854 -> 234.090` TF/s (`+13.7%`)

`runs` distribution:
- `topk=1`: `274.611 -> 287.328` TF/s (`+4.6%`)
- `topk=2`: `230.163 -> 246.683` TF/s (`+7.2%`)
- `topk=4`: `211.918 -> 234.356` TF/s (`+10.6%`)
- `topk=6`: `200.143 -> 229.900` TF/s (`+14.9%`)
- `topk=8`: `206.747 -> 233.486` TF/s (`+12.9%`)

Observed run statistics:
- `random`: run mean `~1.02`, p95 `1.00`, max `3`
- `runs`: run mean `~50.33`, p95 `138`, max `327`

Takeaway:
- Despite much longer runs in routing assignments, throughput and prequeue uplift are very close to `random`.
- This suggests current bottlenecks are dominated by pack/sort/scatter mechanics and total duplicated token volume (`topk`), not raw run-length locality in assignments.

## `topk` scaling in `forward_backward` (training-path)

Purpose:
- Measure train-step throughput sensitivity to `topk` on the shared-expert configuration.

Settings:
- distribution=`random`
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `experts=60`, `shared_expert_dim=5632`
- backend=`gmm`, impl=`fused_w13`, routing=`argsort`
- `queue_mode=full`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`

`topk, full_tflops`
- `1, 219.231`
- `2, 192.449`
- `4, 175.260`
- `6, 164.920`
- `8, 167.868`

Takeaway:
- Throughput declines as `topk` rises from `1` to `6`.
- `topk=8` shows a small rebound over `topk=6`, but remains far below `topk<=2`.
- Practical guidance remains: keep `topk` as low as model quality permits.

## Shared-expert fused path in `forward_backward`

Purpose:
- Check if fusing shared expert `w1/w3` projection helps on the training path.

Settings:
- distribution=`random`
- `tokens=32768`, `hidden=2048`, `mlp_dim=1408`, `experts=60`, `shared_expert_dim=5632`
- backend=`gmm`, impl=`fused_w13`, routing=`argsort`
- `queue_mode=full`, `bench_pass=forward_backward`, `iters=3`, `warmup=1`

`topk, shared_fused, full_tflops`
- `2, false, 191.905`
- `2, true, 196.058`
- `8, false, 167.546`
- `8, true, 168.198`

Takeaway:
- `shared_fused` is a modest positive knob (`~2.2%` at `topk=2`, `~0.4%` at `topk=8`).
- Keep it enabled by default when shared expert is present.

### Training-path locality spot-check (`forward_backward`, `topk=8`)

Settings:
- same as above, except `distribution in {random, runs}` and `topk=8`

`distribution, topk, full_tflops`
- `random, 8, 167.617`
- `runs, 8, 167.541`

Takeaway:
- Locality still does not move training-path throughput materially at this shape.
