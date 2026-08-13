# MoK EP64 Research

## Framing

Marin's current Mixture-of-Kittens adapter has sealed EP4 correctness and stability through a
two-rack, 128-GPU data-parallel run, but every native expert group remains four GPUs inside one
JAX process. The next target is one NVL72 rack used as a single EP64 group: 128 routed experts,
two experts per GB200, and no data-parallel replication.

## In-repository findings

- The model and mesh already support a cross-process expert axis. The existing EP hero launcher
  uses 16 tasks by four GB200s, expert axis 64, batch 1024, E128, and two experts per device
  (`experiments/grug/moe_hero_ep/launch.py`, `experiments/grug/moe_hero_ep/heuristic.py`).
- Iris already supports four supervised processes per four-GPU task. It assigns each child a
  global process rank, global process count, and one local device
  (`lib/iris/src/iris/hooks/multigpu_main.py`). JAX initialization consumes this contract
  (`lib/iris/src/iris/runtime/jax_init.py`).
- The current adapter is process-local by construction:
  - `api.py` requires an expert axis of four and four visible local GPUs.
  - `runtime.py` validates that each expert group is contained in one process.
  - `mok_forward_ffi.cu` fixes `kNumDevices = 4`, allocates all four devices with `cudaMalloc`,
    exchanges raw pointers through one host mutex, and represents arrival with eight-bit masks.
  - generated source uses `1U << peer_rank`, which is invalid for ranks 32-63.
- Current promoted forward zero-copy uses XLA pointers registered inside one process. Those
  allocations are not remotely mapped. EP64 must begin with runtime-staged forward and backward.
- The JAX schedule builder is rank-count generic, but its capacity is
  `local_assignments * schedule_capacity_factor`. Factor 4 is strict for EP4; strict all-to-one
  EP64 requires factor 64.
- The terminal audit currently multiplies handler counts by four local devices. One-GPU JAX
  processes require counts derived from `jax.local_device_count()`.

## Upstream Mixture-of-Kittens

Pinned source: [cursor/mixture-of-kittens@6438bf48](https://github.com/cursor/mixture-of-kittens/tree/6438bf48f88094d305972fbe0fa6deba0f7d4d1a).

- Upstream explicitly accepts EP sizes 4, 8, 16, 32, and 64 and instantiates BF16/MXFP8
  forward and backward kernels for each size.
- Its process model is one process/current CUDA device per EP rank, launched with `torchrun`.
- It delegates all remote mapping to PyTorch 2.11 symmetric memory. Each scratch allocation is
  created with `symm_mem.empty`, rendezvoused over a process group, and exposed to the kernel as
  one pointer per peer.
- Multicast is used only for upstream's route all-gather and barrier. Marin can retain its JAX
  route all-gather and its own generation/cancellation protocol; the fused kernel's scratch
  accesses use ordinary peer-pointer arrays.
- Upstream has no bounded peer-failure closure. Marin's v15 cancellation and typed failure fence
  remain necessary.
- The source implements and claims EP64/NVL72 support, but this checkout contains no durable
  EP64 result. Marin must reproduce it.

## Prior Marin EP64 work

- [Echo wiki 94](https://echo.oa.dev/wiki/94) records a rank-32 failure in JAX/NCCL's direct
  one-sided EP64 path on GB200. A metadata-only kernel reproduced the boundary. Standard
  all-gather/psum-scatter transport passed EP64 parity.
- The MoonEP investigation on `origin/rav/moonep-jax` tried one process per GPU, multiple XLA/NCCL
  builds, smaller transfers, serialized launches, and direct LSA/GIN variants. Direct JAX/NCCL
  device transport still failed above rank 31. A two-slice collective fallback was correct but
  several times below its performance gate.
- Those failures do not exercise PyTorch symmetric memory or the upstream MoK pointer kernel.
  They rule out treating XLA's current one-sided collective memory as the EP64 transport.

## Transport alternatives

1. **PyTorch symmetric-memory workspace, 64 one-GPU JAX processes.** This matches upstream's
   allocation model and uses an exact dependency already present in Marin's GPU environment.
   Torch owns only a flat workspace allocation and mapping; JAX owns compute. Risk: private alpha
   API and collective teardown. PyTorch 2.11 only needs a process-group Store for metadata, so the
   initial implementation uses a dedicated CPU/Gloo group instead of loading a second NCCL stack.
2. **Native CUDA fabric-VMM arena, 16 four-GPU JAX processes.** One exportable arena per global
   rank is mapped into every process using `CU_MEM_HANDLE_TYPE_FABRIC`. This preserves the sealed
   process layout but requires new handle exchange, IMEX validation, mapping, access control, and
   teardown code. It is the fallback if Torch symmetric memory cannot expose all 64 peers.
3. **JAX/NCCL direct collective memory.** Rejected for the first implementation because prior
   rack experiments consistently fail at rank 32 on this stack.
4. **Standard collectives around the fused local compute.** Correct, but prior EP64 evidence shows
   transport-dominated performance. Retain only as a correctness oracle/fallback.

## Decision

Start with alternative 1 because it is the smallest path to exercising upstream's actual EP64
contract. Keep it isolated behind a workspace-owner interface so alternative 2 can replace it
without changing the FFI or model API. The first experiment is a standalone symmetric-memory
probe in the same 64-process JAX/Iris topology. The fused kernel is not modified until the probe
proves remote reads/writes, system-wide barriers, repeated generations, and clean teardown.

Marin's production GPU extra already pins PyTorch 2.11. Its CUDA symmetric-memory backend selects
fabric VMM handles when fabric access is available on GB200. The local tensor, rendezvous handle,
and Gloo process group remain strongly owned until a collective reverse-order shutdown.

## Questions resolved from the conversation

- Hardware scope: one or two racks are authorized, but EP64 targets one 64-GPU NVLink domain.
- Goal: working EP64, not merely a design or capability assessment.
- Initial transport may stage XLA inputs; cross-process XLA zero-copy is not required.
- The existing issue #8108 remains the coordinating issue.

## Remaining unknowns

- Whether the installed PyTorch CUDA backend selects fabric mappings for all 64 GB200 ranks and
  whether every pod has the required IMEX channel access.
- Whether JAX and a dedicated Torch Gloo metadata group coexist without allocator or shutdown
  interference.
- Whether the EP64 schedule's compilation and temporary memory fit the production shape.
- Whether strict factor-64 capacity is practical; initial correctness also needs a deliberately
  capacity-limited contract with exact drop accounting.
