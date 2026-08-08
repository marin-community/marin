# GB200 MoE transport/merge boundary checkpoint

This artifact isolates the distributed MoE return path from DeepEP's semantic
`combine`. DeepEP performs forward payload dispatch. Torch
`all_to_all_single` performs reverse payload permutation without reduction. A
generated CUDA kernel then folds owner-rank contributions in ascending rank
order with explicit FP32 multiply/add and adds the shared-expert output. The
merge uses no atomics.

The grouped expert contractions use the standalone MoK-derived SM100 grouped
GEMM primitive. Grouped GEMM is an allowed physical skeleton under Shuttle's
synthesis boundary. The measured path is therefore a synthesized distributed
schedule at a supplied-route boundary: Shuttle owns relation legalization,
packing, generated SwiGLU, deterministic return/merge, overlap, and kernel
composition. Router/top-k execution and index-plan construction are excluded
from the timed region, so this is not yet a complete ordinary-JAX-to-runtime
measurement.

## Result

The workload uses four GB200 ranks, 2,048 tokens per rank, 384 global experts,
96 local experts, top-6 routing, hidden size 7,168, intermediate size 3,072,
BF16 inputs and weights, concatenated W13, and 56 DeepEP communication SMs.
Each run used 10 warmups and retained 30 rank-maximum and four per-rank timing
samples for every phase.

| Phase | First median | Confirmation median |
|---|---:|---:|
| Routed compute after dispatch | 3.555584 ms | 3.536016 ms |
| Generated shared expert | 0.242464 ms | 0.240784 ms |
| Payload return plus generated merge | 0.365168 ms | 0.368576 ms |
| Clean sequential region | 4.229424 ms | 4.175808 ms |
| Clean overlapped region | 4.082608 ms | 4.142576 ms |
| Historical DeepEP-combine control | 4.085024 ms | 4.044240 ms |
| DeepEP combine plus shared bias component | 0.271072 ms | 0.274544 ms |

The first clean run is 1.1463 times the frozen 3.561696-ms MoK replay. The
confirmation is 1.1631 times the replay. Their median of medians is 4.112592 ms,
or 1.1547 times the replay. Both runs meet the 1.2-times target.

All four generated outputs are bitwise equal to the corresponding
DeepEP-combine controls. Generated repeats are bitwise stable. The saved
semantic fixture SHA256 values are identical across the two runs for every
rank.

## Reproduce

The benchmark ran from `/tmp/tile_lifetime_runtime` with this environment:

```bash
export PYTHONPATH=/tmp/deepep-torch-intranode-build/lib:/tmp/DeepEP-torch-intranode:/tmp/tile_lifetime_runtime/lib/tile_lifetime/src:/tmp/tile_lifetime_runtime/lib/tile_lifetime/benchmarks
export OMP_NUM_THREADS=1
export CUDA_HOME=/tmp/mok-route-env/lib/python3.12/site-packages/nvidia/cu13
export CUDA_CCCL_INCLUDE=/tmp/mok-route-env/lib/python3.12/site-packages/nvidia/cu13/include/cccl
export TORCH_CUDA_ARCH_LIST=10.0a
export DEEPEP_BUILD_INTRANODE_ONLY=1
export DEEPEP_DISABLE_NVSHMEM=1
export MAX_JOBS=8
export NCCL_SOCKET_IFNAME='^ibs,ibp,lo,docker,veth,cilium,lxc'

/tmp/mok-route-env/bin/torchrun --standalone --nproc-per-node=4 \
  lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_distributed.py \
  --route-fixture /tmp/mok_routes_t2048_e384_k6_seed1234_torch2.10-reserialized.npz \
  --probe-extension /tmp/mok-gmm-probe-clean-merge-cu130/_mok_gmm_probe.cpython-312-aarch64-linux-gnu.so \
  --mok-root /tmp/mixture-of-kittens-fixture-source \
  --deepep-root /tmp/DeepEP \
  --shuttle-revision 4fba36752bdbfd28ad9a0ea8dee121bb382b21c9+clean-merge-dirty \
  --clock-policy cluster_default_unpinned \
  --deepep-sms 56 \
  --gate-up-layout concatenated_e_2i_k \
  --warmup 10 \
  --iterations 30 \
  --json-output /tmp/shuttle-clean-merge-results/deepep-mok-clean-merge-concat-sms56.json \
  --semantic-fixture-output /tmp/shuttle-clean-merge-results/semantic-fixture-clean.npz
```

The confirmation changes only the two output filenames by adding `-repeat` and
`-repeat` before `.npz`.

`source/` contains the executed benchmark and extension sources rather than
the later working-tree versions. Repository formatting normalized the Python
snapshot after capture; `manifest.json` records the normalized source hash.
`raw/` contains both complete JSON distributions, stdout logs, build logs,
package pins, and GPU telemetry. `fixtures/` contains the route fixture and
both sets of per-rank semantic fixtures. The route NPZ is stored as gzip to
meet the repository file-size limit; decompressing it reproduces the
`c143b12f...` content hash in
`fixtures/mok-route-fixture-content-identity.json`. Validate every stored file
with `sha256sum -c SHA256SUMS`.
