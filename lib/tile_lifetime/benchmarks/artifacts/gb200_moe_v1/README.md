# GB200 MoE snapshot

This directory preserves the Shuttle distributed BF16 MoE proof-of-life data.
`manifest.json` pins sources, toolchain, hardware policy, workload, correctness
identities, and every artifact digest. `candidate_space.json` records the legal,
measured, failed, and selected candidates. `benchmark_cache.json` maps candidate
fingerprints to raw run artifacts and timing phases.

`raw/schema1` contains the original 10-warmup, 50-iteration measurements. Those
runs used the cluster's default clock policy; exact clock and power telemetry was
not captured before the trays were released. The raw data is retained without
inventing missing metadata. `raw/schema2` contains the telemetry-enabled replay,
including per-rank timing distributions and deterministic output hashes.

The schema-2 replay measured 3.9830 ms for the selected concatenated/56-SM plan,
4.0649 ms without overlap, 4.4348 ms with deliberately coarse activation
materialization, 4.0690 ms with separate gate/up, and 3.5617 ms for the tuned
MoK oracle. Benchmark-boundary telemetry reported 1950 MHz SM and 3996 MHz
memory clocks on all four GPUs under the cluster-default unpinned policy.

The selected workload is four ranks with 2,048 tokens per rank, 384 global and
96 local experts, top-6 routing, hidden size 7,168, intermediate size 3,072,
BF16 inputs and weights, FP32 contraction accumulation, and deterministic
ascending-slot FP32 merge without atomics.

NPZ fixtures are stored as numbered base64 parts so each durable repository
artifact remains below the repository's file-size limit. Concatenate the parts
in lexical order, remove whitespace, and base64-decode to recover the byte-exact
NPZ whose SHA256 is recorded in the run or fixture identity.

Rebuild the indexes from the preserved inputs with:

```bash
uv run python lib/tile_lifetime/benchmarks/build_gb200_snapshot.py \
  --historical-root scratch/shuttle-generic-results \
  --replay-root scratch/shuttle-gb200-replay \
  --output-root lib/tile_lifetime/benchmarks/artifacts/gb200_moe_v1 \
  --shuttle-revision <commit> \
  --shuttle-tag shuttle-gb200-moe-v1
```
