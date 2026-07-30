# Grug MoE hero: one-rack EP64

This variant fixes the composed EP64 configuration at hidden dimension 5120
with 256 routed experts and top-8 routing for 16 workers with four GB200s each.
It is validated only at one rack. This branch has no accelerator verification.
The resolved settings and result provenance in this README are the durable
record; the [Grug archive](../../../docs/reports/grug-archive.md) indexes it.

## Resolved configuration

The launcher does not read `SCALE_*` configuration overrides. It resolves:

| Setting | Value |
|---|---|
| Hidden dimension / layers | 5120 / 48 |
| Attention heads / KV heads / head dimension | 40 / 10 / 128 |
| Routed experts / experts per token | 256 / 8 |
| Routed / shared intermediate dimension | 1280 / 5120 |
| Shared-expert count | 1 |
| Sequence length / sliding window | 4096 / 2048 |
| Initializer standard deviation | 0.006987712429686843 |
| MoE path | `ragged_all_to_all`, fixed capacity |
| Dispatch / adjoint | gather / custom |
| Capacity factor / spill attempts / chunks | 1.0625 / 3 / 1 |
| FSDP expert chunking | off (`expert_chunks=1`) |
| Fixed all-to-all optimization barrier | disabled |
| Expert / replica axis size | 64 / 1 |
| Topology | 16 workers × 4 GB200 = 64 GPUs |
| Global batch / steps | 1024 / 350 |
| Attention | `gpu_fa4_cute` |
| Rematerialization / layer execution | `recompute_all` / one homogeneous scan |
| Mixed precision | `params=float32,compute=bfloat16,output=bfloat16` |
| Router | QB routing |
| Optimizer | MuonH; padded non-expert Newton–Schulz; SYRK |
| Checkpoint / eval / parameter watches | disabled / disabled / disabled |

GatedNorm, the attention gate, and QB routing are structural in this copy of
`model.py`; they are not launcher flags. The capacity-factor library default
remains 1.0. This variant requests 1.0625, the value used for the measured drop
fraction. All-to-all chunking stays at 1 because chunks=2 measured worse.

MuonH uses a linear schedule with 1% warmup and a 5% minimum ratio. Its fixed
peak learning rates are 0.038956464533085024 for MuonH and
0.008989953353788853 for Adam, with `beta1=0.9062`,
`beta2=0.9684910757595268`, `epsilon=1.810213843721233e-16`, five
Newton–Schulz steps, and no gradient clipping. Optimizer state stays on the
device. Host offload is intentionally absent: the d5120 EP64 trial required a
135 GiB pinned-host arena and measured 19.694% MFU. The shared-expert split is
also absent: this template keeps the shared MLP as one expert because the
multi-way split used 89.49 GiB before step 0 at EP64.

## Result provenance

The source research build measured a three-placement-draw median of 22.398% MFU,
346,950 tokens/s, and 1.444% dropped assignments over steps 250–349 of
350-step one-rack runs. The MFU denominator was 2.5 PFLOP/s per GB200.

Those measurements came from research build `c24ccfcc2`, not this branch. They
establish the source recipe's result and do not establish that this template
reproduces it. This branch has not compiled or run on an accelerator.

All three measured draws replayed a build-specific manual PGLE profile. This
template ships without that profile and forces auto-PGLE off. That profile's
measured EP contribution was 0.427 percentage points, so the template as
shipped is expected to land about 0.427 points below the quoted 22.398% absent
a current-build profile.

The FSDP and EP headline numbers do not establish which strategy is faster.
They use different model shapes and measurement windows. In the available
same-shape comparison, the 0.942% margin is inside placement noise; repeated
draws on one placement agreed within 0.02%.

The JAX 0.11 baseline measured 1.217 percentage points below the 0.10.1-era
baseline. Do not use pre-0.11 results as controls for this template.

## Submit

The launcher hardcodes the model, optimizer, topology, batch, runtime MoE
choices, and 350-step schedule. Set only the run identity and the required GPU
runtime environment:

```bash
run_id="moe-hero-ep-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/iris job run \
  --no-wait \
  --max-retries 50 \
  --cpu 2 \
  --memory 3GB \
  --extra cpu \
  --job-name "$run_id" \
  -e RUN_ID "$run_id" \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4 --xla_gpu_enable_latency_hiding_scheduler=true" \
  -- python -m experiments.grug.moe_hero_ep.launch \
    --version dev \
    --run
```

`--max-retries 50` is the standing rack-submission protocol.

`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is required. The default BFC allocator
has caused silent deadlocks and fragmentation OOMs at 64×GB200; clique-init
stalls on this platform have been traced to BFC fragmentation.

All three XLA flags in the command are required:

- `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` is
  mandatory on JAX 0.11. Without it, a 64-process run compiles and then
  segfaults in `ncclDevCommCreate` before step 0.
- `--xla_gpu_experimental_parallel_collective_overlap_limit=4` clears the four
  SYNCs seen at limit 1. The census found 12 MoE all-to-alls at limits 1, 2, 4,
  and 8, with SYNC counts `4,0,0,0`. Performance is not monotone: limit 2
  measured worse than limit 1.
- `--xla_gpu_enable_latency_hiding_scheduler=true` enables collective
  scheduling with the chosen overlap limit.

Auto-PGLE is forced off by the variant because it crashes multi-host runs. No
manual PGLE profile ships here because a profile is tied to one compiled
executable. To restore the measured configuration, use JAX's manual FDO flow
for the exact build and flags in use: collect five profiling runs, combine the
profiles into one shared `.pb`, place that file at the same path on every
worker, and add
`--xla_gpu_pgle_profile_file_or_directory_path=<current-build-profile.pb>` to
`XLA_FLAGS` for the replay run. Regenerate the profile after any code,
dependency, shape, or XLA-flag change.

The replayed profile matched 225 of 533 scheduled instructions, but match rate has
been retired as a profile-quality signal. A separate ECHO-line manual profile
measured 0.235 percentage points below AutoPGLE; that result does not remove the
source profile from the provenance of the quoted number.

Do not add `xla_gpu_enable_custom_fusions` or
`xla_gpu_enable_address_computation_fusion`. Both killed the measured build
before distributed initialization.

## Multi-rack status

No weak-scaling factor has been measured for this configuration. Do not
extrapolate one from the one-rack result or reuse the historical approximately
19% FSDP one-to-two-rack penalty.

Two blockers prevent a multi-rack claim:

1. Multi-rack placement requires the Fray gang-topology fix tracked in
   [#7753](https://github.com/marin-community/marin/issues/7753). That fix is
   not on this branch. Before the fix, a requested 16+16 placement silently
   admitted 14+18 workers across two NVLink domains.
2. The two-rack EP64 attempt stalled in an asynchronously dispatched device
   collective that never completed. Both racks used uniform CUDA 13 NCCL and
   the same `libnccl.so.2`. The public evidence is tracked in
   [#7344](https://github.com/marin-community/marin/issues/7344). The evidence
   excludes placement, host-side drop reporting, and a CUDA-major mismatch;
   the stall remains unresolved.

The collective-hang watchdog is armed for every GPU run by
`resolve_training_env` with
`--xla_gpu_nccl_termination_timeout_seconds=600`. It is diagnostic coverage,
not recovery for the two-rack failure: the attempt remained wedged for 879
seconds without a `TimeoutError`, `ncclCommAbort`, or NCCL output.

The stalled attempt reported a 7.99% drop fraction at step 90. The one-rack
1.44% result covers steps 250–349, so these values are not comparable. The
step-90 value is also not evidence of two-rack fidelity.
