# Grug MoE Muon Update Benchmark

`experiments/grug/moe/muon_update_bench.py` is the fast iteration harness for Grug Muon/MuonH optimizer work. It runs only the synthetic optimizer update/apply path: production-shaped MuonH parameter/update leaves, grouped routed expert leaves, Newton-Schulz step count, stack/group mode, and single-node mesh axes. It does not launch a full training job.

Use it before full May profiles when changing Muon representation, stack caps, padding/sharding strategy, or H1/H3/H5.

## Local Smoke

```bash
MUON_BENCH_PROFILE=fullprod-e8-h3 \
MUON_BENCH_HIDDEN_DIM=16 \
MUON_BENCH_INTERMEDIATE_DIM=8 \
MUON_BENCH_NUM_EXPERTS=8 \
MUON_BENCH_WARMUP=1 \
MUON_BENCH_ITERS=1 \
bash scratch/muon_update_bench_fast_loop.sh local
```

The wrapper sets `XLA_FLAGS=--xla_force_host_platform_device_count=<mesh size>` when `XLA_FLAGS` is unset. For GPU hosts, set `XLA_FLAGS` yourself if you do not want that CPU-device override.

## CoreWeave One-Node Runs

Outputs default to `s3://marin-na/tmp/ttl=7d`.

```bash
MUON_BENCH_PROFILE=fullprod-e8-l26-h3 \
bash scratch/muon_update_bench_fast_loop.sh iris
```

Useful profiles:

- `fullprod-e8`: routed-expert-only and full-production MuonH, H1/H3/H5, E8.
- `fullprod-e8-h3`: cheap two-layer H3 profile, E8.
- `fullprod-e8-l26-h3`: current H3 speed-clearing profile, E8, 26 layers, group size 4.
- `fullprod-r4e8-l26-h3`: 32-GPU Muon-only scaling gate, group axis `replica_dcn`, 26 layers.
- `grouped2d-decomp-r4e8-l26-h3`: 32-GPU grouped-2D decomposition sweep. Compares routed expert grouped MuonH, full production MuonH, full production grouped-2D MuonH, ordinary-2D-only MuonH apply, ordinary grouped-2D MuonH apply, grouped-2D stack+NS only, grouped-2D restore/split only, and full production apply-only plumbing.
- `fullprod-r16e8-l26-h3`: 128-GPU Muon-only scaling gate, group axis `replica_dcn`, 26 layers.
- `grouped-d2e4`: D2/E4 grouped MuonH and restore/split control.
- `padding-d2e4`: padding-vs-sharding probes for non-divisible group axes.

Common overrides:

```bash
MUON_BENCH_LAYERS=26
MUON_BENCH_NS4D_GROUP_SIZE=4
MUON_BENCH_SWEEP_BACKEND_STEPS=1,3,5
MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES=256,512
MUON_BENCH_DATA_AXIS=2
MUON_BENCH_EXPERT_AXIS=4
```

For scale probes, `MUON_BENCH_NS4D_GROUP_AXIS=replica_dcn,data` asks the grouped
4D path to shard the grouped layer axis over both inter-node replicas and local
data shards when the group size is divisible by their product. Use this for the
32/128 GPU Muon-only gates before attempting any train integration run.

Keep `MUON_BENCH_MODEL_AXIS=1` for this single-node harness.

## Output Fields

The JSON output contains raw per-variant events and a compact `summary_table`. The summary includes:

- wall time: `mean_seconds`, `median_seconds`, `min_seconds`
- compile time: `compile_seconds`
- work estimate: `estimated_ns_dot_flops`, `estimated_matrix_count`
- throughput: `mean_estimated_tflops`, `median_estimated_tflops`, `%` of nominal 8xH100 bf16 peak
- stack/chunking: `group_estimates`, `grouped_chunks`, `chunks`, `max_grouped_stack_size`
- grouped shape: `ns4d_group_size`, `ns4d_padded_group_size`, `grouped_expert_group_count`
- grouped 2D decomposition: `grouped_2d_estimates`, `grouped_2d_chunks`, and `grouped_2d_chunk_sizes`
- sharding: `ns4d_input_sharding_spec`, `ns4d_compute_sharding_spec`, `ns4d_result_sharding_spec`
- collectives: lowered and compiled all-gather/all-reduce/reduce-scatter/collective-permute counts

Grouped apply benchmarks assert that compiled all-gather/all-reduce/reduce-scatter counts stay zero at the grouped optimizer/apply boundary.

## Current Baseline

Issue: https://github.com/marin-community/marin/issues/6493

Current speed-clearing production compromise is the L26 H3 gate:

```bash
MUON_BENCH_PROFILE=fullprod-e8-l26-h3 \
bash scratch/muon_update_bench_fast_loop.sh iris
```

Parent `/dlwh/iris-run-job-20260618-153021`, output `s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/muon-update-bench/MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-G4-H3-N1-cw-20260618-153016-87deb1`:

| bench | H3 mean s / 26 layers | H3 peak % | compiled AG/AR/RS |
| --- | ---: | ---: | --- |
| routed expert grouped MuonH | 0.5236 | 58.63 | 0/0/0 |
| full production MuonH | 0.6171 | 50.56 | 0/0/0 |

Do not use an unchanged full train profile as the next Muon iteration step. First use this update-only harness to prove a new implementation axis or expose H3 cleanly in the production optimizer path.
