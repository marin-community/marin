# Debugging log for DeepEP internode JAX integration

Goal: drive Grug MoE expert-parallel communication overhead down by getting a
cross-node DeepEP-style transport working in Marin/Levanter, or by replacing it
with a JAX-native implementation of the same packed variable-size dispatch and
combine idea.

## Initial status

Upstream DeepEP normal-mode internode is cross-node and has a working CoreWeave
control script at `scratch/launch_deepep_upstream_internode_test_n2.sh`.

The Marin/JAX path differs from upstream:

- upstream DeepEP uses 8 GPU-owning processes per H100x8 node;
- Marin/JAX currently uses one Python process per H100x8 node and controls 8
  local GPUs through JAX `pmap`;
- the current C++ bridge emulates 8 DeepEP local device runtimes inside that
  one process and initializes NVSHMEM once per node.

The latest 2-node JAX FFI smoke,
`/dlwh/deepep-internode-jax-ffi-smoke-20260621-114957`, initialized local
runtimes, NVSHMEM, and RDMA successfully. All 16 pmap ranks returned from
`internode::notify_dispatch`, then every rank timed out waiting for counters:

- `num_recv_tokens=-1`
- `num_rdma_recv_tokens=-1`
- `expert_counters=[-1]`

This means the current failure is after notify returns: receive counters are not
updated or not visible to the one-process-per-node runtime.

## Hypothesis 1

The host-mapped counter allocation itself may be incompatible with the
one-process-per-node runtime. If a trivial device kernel cannot write the mapped
counter pointers and make those writes visible on the host, the bug is in
counter ownership/mapping rather than in DeepEP's notify algorithm.

## Changes to make

- Add `levanter_deepep_run_internode_mapped_counter_smoke` to the internode
  transport library.
- The smoke launches one tiny CUDA kernel per local `DeviceRuntime`.
- Each kernel writes sentinel values to:
  - `moe_recv_counter_mapped`
  - `moe_recv_rdma_counter_mapped`
  - `moe_recv_expert_counter_mapped[0]`
- The host synchronizes each runtime stream and verifies the corresponding host
  counters observed the sentinel values.
- Expose the smoke as `run_internode_mapped_counter_smoke()` in
  `transport_ffi.py`.
- Run it in `scratch/launch_deepep_internode_jax_ffi_smoke_n2.sh` immediately
  after `ensure_internode_runtime()` and before `deepep_dispatch_internode()`.

## Results

CoreWeave run:

- Parent Iris job:
  `/dlwh/deepep-internode-jax-ffi-smoke-20260621-120255`
- State file:
  `scratch/20260621-120255_deepep_internode_jax_ffi_smoke_state.json`
- Runtime shape:
  2 H100x8 tasks, one Python/JAX process per node, 8 local pmap ranks per
  process.

Observed result:

- Both tasks initialized the internode runtime:
  `InternodeRuntimeStatus(initialized=True, process_count=2,
  num_local_ranks=8, num_global_ranks=16, ...)`.
- The mapped-counter smoke passed on both tasks:

```text
DEEPEP_JAX_FFI_COUNTER_SMOKE {"internode_mapped_counter_smoke_status_code": 0, "last_error": "", "num_checked": 8}
```

- After that, the remote-route dispatch reproduced the previous failure shape:
  all 16 ranks reached `internode_jax_after_notify_dispatch`, then
  `internode_jax_before_wait_recv_counts`, with counters still unset at
  `num_recv_tokens=-1`.
- The job then reached terminal failure. Task 1 emitted timeout diagnostics for
  ranks `8..15`; rank 15 had `num_rdma_recv_tokens=1` while
  `num_recv_tokens=-1` and `expert_counters=[-1]`. Iris reported task 0 as
  failed with exit code 1 and task 1 as coscheduled-failed after its sibling.

Interpretation:

- CUDA host-mapped counter allocation/visibility is not the primary bug for the
  one-process-per-node runtime. A trivial kernel on every local device can write
  the mapped counter pointers and the host observes the sentinels.
- The remaining failure is inside DeepEP's internode notify/counter handshake
  under the Marin/JAX process model: `internode::notify_dispatch` returns, but
  it does not produce the full set of recv-counter writes visible to the
  `DeviceRuntime` counters used by the JAX FFI bridge. The rank-15 RDMA counter
  increment shows this is not necessarily "no device wrote anything"; it may be
  partial progress to a different normal-mode counter contract.
- This shifts the next hypothesis away from generic counter mapping and toward
  DeepEP normal-mode assumptions: rank grouping, peer pointer tables, IPC/NVL
  buffer ownership, RDMA buffer ownership, or the fact that upstream uses one
  GPU-owning process per rank while Marin is emulating 8 device runtimes inside
  one process.

## Future work after Hypothesis 1

- [x] Rule out generic CUDA host-mapped counter allocation/visibility as the
      primary failure.
- [x] Test whether DeepEP notify fails without JAX FFI array inputs.
- [ ] If DeepEP remains blocked, continue with the JAX-native packed
      variable-size assigned-token bridge and profile it against EP16 ring.

## Hypothesis 2

The failure is not in JAX FFI array plumbing. It is in the one-process-per-node
emulation of DeepEP normal-mode local ranks. A host-only internode dispatch
round that bypasses jitted JAX arrays should fail the same way if the emulated
`DeviceRuntime` rank/buffer tables do not satisfy upstream DeepEP's assumptions.

## Changes to make

Run the existing `run_host_internode_dispatch_round(...)` probe on two H100x8
nodes using the current internode runtime and CoreWeave NVSHMEM settings.

## Results

CoreWeave run:

- Parent Iris job:
  `/dlwh/deepep-host-internode-dispatch-smoke-20260621-120924`
- State file:
  `scratch/20260621-120924_deepep_host_internode_dispatch_smoke_state.json`

Observed result:

- Both tasks initialized the internode runtime.
- Host-only local-rank threads reached `internode_after_notify_dispatch` and
  `internode_before_wait_recv_counts`.
- One rank failed early with:

```text
HOST_INTERNODE_DISPATCH_ERROR {"local_rank":3,"error":"Failed: CUDA error /tmp/DeepEP/csrc/kernels/internode.cu:346 'an illegal memory access was encountered'"}
```

- The other local ranks timed out with:

```text
DEEPEP_INTERNODE_COUNTER_TIMEOUT {"num_recv_tokens":-1,"num_rdma_recv_tokens":-1,"num_local_experts":1,"expert_counters":[-1]}
```

Interpretation:

- The same DeepEP normal-mode boundary fails without JAX FFI array inputs.
- The blocker is therefore lower than the jitted dispatch wrapper: the current
  one-process-per-node runtime is not a valid substitute for upstream DeepEP's
  one-process-per-GPU normal-mode `Buffer` setup.
- Continuing with this exact emulation is unlikely to produce a training win
  without reworking the rank/buffer ownership model. The pragmatic paths are:
  one-process-per-GPU execution if it can fit Grug/JAX, or a JAX-native/lower
  level packed variable-size bridge that borrows DeepEP's transport ideas but
  does not rely on upstream normal-mode process ownership.

## Hypothesis 3

Iris/JAX one-process-per-GPU is the right rank topology for upstream DeepEP
normal mode, but the Marin FFI must make that topology explicit and refuse to
launch until the CUDA runtime can exchange same-node peer handles.

For EP16 on two H100x8 nodes the intended topology is:

- `process_count=16`
- `ranks_per_node=8`
- `node_count=2`
- `node_rank=process_index // 8`
- `local_rank=process_index % 8`

The current C++ `InternodeRuntimeManager` still only supports the older
same-process local-rank emulation. It takes `process_rank`, `process_count`,
and `num_local_ranks`, then creates `num_local_ranks` `DeviceRuntime`s inside
the process and fills local NVLink peer pointer tables directly. That is not
valid when Iris gives each JAX process one visible GPU.

## Changes to make

Add an explicit topology helper and preflight gate:

- `InternodeProcessTopology`
- `current_internode_process_topology(...)`
- `preflight_internode_process_topology(...)`
- `DEEPEP_RANKS_PER_NODE`, surfaced by `run_cw_may_d2560.sh` as
  `--deepep-ranks-per-node`

The preflight deliberately rejects the target one-process-per-GPU topology with
a clear error until the C++ runtime grows CUDA IPC/local-peer exchange.

## Results

Focused local checks passed:

```text
uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q -k 'internode_process_topology or local_internode_bootstrap_metadata or ensure_internode_runtime_calls_internode_init or internode_bootstrap_metadata_round_trips_bytes'
bash -n experiments/grug/moe/run_cw_may_d2560.sh
uv run python -m py_compile lib/levanter/src/levanter/kernels/deepep/transport_ffi.py lib/levanter/src/levanter/kernels/deepep/preflight.py
./infra/pre-commit.py --files lib/levanter/src/levanter/kernels/deepep/transport_ffi.py lib/levanter/src/levanter/kernels/deepep/preflight.py lib/levanter/src/levanter/kernels/deepep/__init__.py lib/levanter/tests/kernels/test_deepep_availability.py experiments/grug/moe/run_cw_may_d2560.sh
```

Next implementation boundary:

1. Extend the C++ init signature from
   `(process_rank, process_count, num_local_ranks, ...)` to explicit
   `(global_rank, global_rank_count, node_rank, node_count, local_rank,
   ranks_per_node, ...)`.
2. Exchange CUDA IPC handles and same-node NVL buffer/barrier pointer metadata
   across JAX processes with the existing JAX distributed key-value client.
3. Initialize exactly one `DeviceRuntime` per process, with global rank
   `process_index`, local rank `process_index % ranks_per_node`, and peer
   tables populated from IPC-opened same-node handles.
4. Keep RDMA/NVSHMEM init at the global-rank level, not the old emulated
   node-rank level.
5. Only then run an EP16 DeepEP smoke with `DEEPEP_RANKS_PER_NODE=8`.

## Hypothesis 4

The CoreWeave/Iris scheduler can launch the rank topology DeepEP wants: one
JAX process per GPU, packed as eight ranks per H100x8 node. If that shape works
for ordinary JAX distributed collectives, then the remaining DeepEP blocker is
runtime initialization and peer-handle exchange, not Iris scheduling or JAX
distributed bringup.

## Changes to make

Add a narrow topology probe launcher:

- `experiments/grug/moe/launch_cw_deepep_topology_probe.py`
- `scratch/launch_deepep_topology_probe_gpu1_n2.sh`

The launcher requests `ResourceConfig.with_gpu("H100", count=1, replicas=16)`
and sets `DEEPEP_RANKS_PER_NODE=8`. The probe prints JAX process/device state,
the derived `InternodeProcessTopology`, and runs one 16-way JAX `psum`.

## Results

CoreWeave run:

- Parent Iris job:
  `/dlwh/iris-run-job-20260621-135045`
- Child Iris job:
  `/dlwh/iris-run-job-20260621-135045/grug-train-DEEPEP-TOPOLOGY-PROBE-GPU1-RPN8-N2-cw-20260621-1350`
- State file:
  `scratch/20260621-065054_deepep_topology_probe_gpu1_n2_state.json`

Observed result:

- Parent and child reached `JOB_STATE_SUCCEEDED`.
- The child ran 16 tasks and all 16 tasks succeeded.
- Every task reported `local_devices=1`, `global_devices=16`, and
  `process_count=16`.
- Derived topology matched the intended EP16/N2 process-per-GPU shape:
  - ranks `0..7`: `node_rank=0`, `local_rank=0..7`
  - ranks `8..15`: `node_rank=1`, `local_rank=0..7`

## Hypothesis 5

The one-GPU-pod topology is not stable enough for DeepEP CUDA IPC assumptions
under CoreWeave Kueue placement. A more upstream-like launch shape should work:
launch two Iris tasks with `H100x8`, then have each task spawn eight local
single-GPU Python/JAX worker processes. That makes local ranks physically share
one host and avoids relying on Kueue to pack 16 independent one-GPU pods as
exactly `8 + 8`.

This also tests whether the process-per-GPU C++ runtime and IPC exchange added
for Hypothesis 3 is actually sufficient once placement is correct.

## Changes to make

Add a supervised DeepEP runtime smoke:

- `experiments/grug/moe/launch_cw_deepep_runtime_smoke.py`
  - supports the original `16 x H100x1` child shape;
  - adds `DEEPEP_RUNTIME_PROCESSES_PER_TASK`;
  - when `processes_per_task > 1`, each `H100x8` Iris task preps RDMA headers,
    checks out DeepEP source, prebuilds the transport FFI once, then spawns
    eight subprocesses;
  - each subprocess masks `CUDA_VISIBLE_DEVICES` to one GPU and explicitly calls
    `jax.distributed.initialize(coordinator_address, num_processes=16,
    process_id=global_rank, local_device_ids=0)`;
  - after JAX init, each subprocess calls `ensure_internode_runtime(...)` and
    `run_internode_mapped_counter_smoke()`.
- `scratch/launch_deepep_runtime_smoke_h100x8_n2.sh`
  - launches the supervised shape: 2 Iris tasks, each `H100x8`, each spawning 8
    one-GPU JAX workers.

Two launch bugs were fixed while testing:

- worker subprocesses initially raced on `apt-get update`/`libibverbs-dev`
  install; the supervisor now does this once before spawning workers;
- worker subprocesses initially raced while compiling/loading the shared DeepEP
  FFI cache, causing `file too short` and CUDA registration symbol load errors;
  the supervisor now prebuilds/loads the FFI once per node before spawning
  workers.

Local validation:

```text
uv run python -m py_compile experiments/grug/moe/launch_cw_deepep_runtime_smoke.py lib/levanter/src/levanter/kernels/deepep/transport_ffi.py
bash -n scratch/launch_deepep_runtime_smoke_h100x8_n2.sh scratch/launch_deepep_runtime_smoke_gpu1_n2.sh
uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q -k 'internode_process_topology or process_per_gpu or internode_bootstrap_metadata or root_nvshmem or ensure_internode_runtime_calls'
```

## Results

Final successful CoreWeave run:

- Parent Iris job:
  `/dlwh/iris-run-job-20260621-144426`
- Child Iris job:
  `/dlwh/iris-run-job-20260621-144426/grug-train-DEEPEP-RUNTIME-SMOKE-H100X8-RPN8-N2-cw-20260621-1444`
- State file:
  `scratch/20260621-144424_deepep_runtime_smoke_h100x8_n2_state.json`

Observed result:

- Parent and child reached `JOB_STATE_SUCCEEDED`.
- The child ran 2 `H100x8` Iris tasks:
  - task 0 host: `g83cc82`
  - task 1 host: `g83d3ee`
- Each task spawned 8 single-GPU subprocesses.
- All 16 subprocesses initialized JAX:
  - `process_count=16`
  - `local_devices=1`
  - `global_devices=16`
- All 16 subprocesses derived the intended process-per-GPU topology:
  - ranks `0..7`: `node_rank=0`, `local_rank=0..7`
  - ranks `8..15`: `node_rank=1`, `local_rank=0..7`
- DeepEP process-per-GPU runtime initialized successfully on every rank:

```text
InternodeRuntimeStatus(initialized=True, process_count=16,
num_local_ranks=8, num_global_ranks=16, num_nvl_bytes=8388608,
num_rdma_bytes=8388608)
```

- The mapped-counter smoke passed on every rank:

```text
{"internode_mapped_counter_smoke_status_code": 0, "last_error": "", "num_checked": 1}
```

Interpretation:

- DeepEP's cross-node runtime can initialize on CoreWeave from Marin when the
  launch shape matches upstream assumptions: one GPU-owning JAX process per
  GPU, eight local ranks per H100x8 node.
- The earlier one-GPU-pod runtime smoke failure was a placement/topology issue:
  Kueue/Fray coscheduled by `leafgroup`, but placed 16 one-GPU pods as
  `4 + 4 + 4 + 4` across four hosts while the code assumed `8 + 8`.
- The one-process-per-node emulation remains a dead end for upstream DeepEP
  normal-mode dispatch, but the supervised H100x8 process-per-GPU shape is a
  viable path for integrating DeepEP-style EP16 transport into Grug MoE.

Next implementation boundary:

1. Turn the supervised process-per-GPU smoke into a reusable CoreWeave launch
   mode for Grug MoE experiments, or integrate equivalent per-GPU workers into
   the training launcher.
2. Run a minimal DeepEP dispatch/combine correctness and timing smoke under the
   successful supervised shape, not just runtime init/counter smoke.
3. Wire the Grug MoE DeepEP path to this process topology for EP16 and compare
   communication overhead against ring/all-to-all.
  - all ranks: `process_model='process_per_gpu'`,
    `ranks_per_node=8`, `node_count=2`, `visible_local_gpus=1`
- The 16-way JAX collective returned the expected `psum` value on every rank:
  `local_shards=[[120.0]]`, the sum of ranks `0..15`.

The logs emitted `WatchTasksAsync failed` / `Connection refused` warnings after
the `probe: done ...` lines. Iris still reported successful termination for all
tasks, so these are shutdown noise from fast task exit rather than the DeepEP
counter/illegal-memory failure seen in the previous smokes.

Interpretation:

- One-process-per-GPU execution on CoreWeave is viable for JAX distributed
  collectives and gives the topology DeepEP normal mode expects.
- The old one-process-per-node emulation should not be the primary path.
- The next DeepEP implementation boundary is the C++ runtime init:
  it must initialize exactly one `DeviceRuntime` per process, open same-node
  CUDA IPC peer handles, and treat RDMA ranks as nodes rather than all GPU
  ranks.

## Hypothesis 4: DeepEP internode dispatch/combine works once the launch shape is fixed

The runtime-only smoke proved that the CoreWeave `2 x H100x8` process-per-GPU
shape can initialize DeepEP's internode runtime. The next question was whether
the real JAX FFI transport entry points, not just runtime init and the mapped
counter smoke, work under that shape.

## Changes to make

Add a direct GPU Iris launcher for dispatch/combine smoke:

- `scratch/launch_deepep_dispatch_smoke_h100x8_n2.sh`
  - launches 2 Iris tasks, each with `H100x8`;
  - each task spawns 8 one-GPU JAX workers via
    `DEEPEP_RUNTIME_DIRECT_CHILD=1` and
    `DEEPEP_RUNTIME_PROCESSES_PER_TASK=8`;
  - defaults to `--cpu 32` because `--cpu 64` caused Kueue/TAS to reject the
    2-node H100x8 placement on CoreWeave;
  - writes under `s3://marin-na/tmp/ttl=7d`;
  - runs dispatch by default and can run combine with
    `DEEPEP_RUNTIME_RUN_COMBINE_SMOKE=1`.
- `experiments/grug/moe/launch_cw_deepep_runtime_smoke.py`
  - added an optional internode dispatch smoke using
    `deepep_dispatch_internode`;
  - added an optional combine smoke using `deepep_combine_internode`;
  - added a precheck for DeepEP's combine invariant:
    `hidden_int4 = hidden * sizeof(dtype) / 16` must be divisible by 32.

Local validation:

```text
uv run python -m py_compile experiments/grug/moe/launch_cw_deepep_runtime_smoke.py
bash -n scratch/launch_deepep_dispatch_smoke_h100x8_n2.sh
```

## Results

Dispatch-only success:

- Parent Iris job:
  `/dlwh/DEEPEP-DISPATCH-SMOKE-H100X8-RPN8-N2-cw-20260621-1507`
- State file:
  `scratch/20260621-150719_deepep_dispatch_smoke_h100x8_n2_state.json`
- Placement:
  - 2 H100x8 tasks;
  - 16 JAX processes;
  - 8 local ranks per node.
- Evidence:
  - all 16 workers initialized JAX with `process_count=16`;
  - all 16 workers passed the mapped-counter smoke;
  - all 16 workers printed `dispatch_smoke_done`;
  - every rank received the expected remote-node token payload:
    `num_recv_tokens=[8]`, `num_recv_rdma_tokens=[8]`,
    `local_expert_counts=[8]`.

Combine-enabled hidden-size failure:

- Parent Iris job:
  `/dlwh/DEEPEP-DISPATCH-SMOKE-H100X8-RPN8-N2-cw-20260621-1511`
- Result:
  dispatch completed, then combine failed in upstream DeepEP with
  `internode.cu:1250, condition: hidden_int4 % 32 == 0`.
- Interpretation:
  the smoke had `hidden=128`, so bf16 `hidden_int4` was 16. This is a valid
  DeepEP precondition, not a CoreWeave transport failure.

Dispatch+combine success:

- Parent Iris job:
  `/dlwh/DEEPEP-DISPATCH-SMOKE-H100X8-RPN8-N2-cw-20260621-1516`
- State file:
  `scratch/20260621-151616_deepep_dispatch_smoke_h100x8_n2_state.json`
- Configuration:
  - `DEEPEP_RUNTIME_RUN_COMBINE_SMOKE=1`;
  - `DEEPEP_RUNTIME_DISPATCH_HIDDEN=256`.
- Evidence:
  - all 16 workers passed the mapped-counter smoke;
  - all 16 workers printed `dispatch_smoke_done`;
  - all 16 workers printed `combine_smoke_done`;
  - combine outputs had shape `[8, 256]` for activations and `[8, 1]` for
    weights, with `combined_weights_sum=8.0` on every rank.

The fast-exit JAX shutdown logs still include occasional
`WatchTasksAsync failed` / `Connection refused` messages after the success
lines. Iris reports the jobs as succeeded, so these are shutdown noise rather
than DeepEP dispatch/combine failures.

Interpretation:

- DeepEP internode normal-mode dispatch and combine work on CoreWeave from the
  Marin JAX FFI under the process-per-GPU `2 x H100x8` launch shape.
- The remaining integration boundary is Grug MoE, not DeepEP internode support
  itself: the current Grug `deepep` backend still needs to call the internode
  dispatch/combine path and run under the process-per-GPU topology.
- EP16 is now a plausible target for the next Grug MoE experiment, because it
  matches the 16 global DeepEP ranks in the proven two-node smoke.

## Hypothesis 5: Grug can route through the internode DeepEP FFI

The smoke proved the raw JAX FFI transport. The next code boundary is the Grug
MoE implementation dispatcher: the existing `deepep` and `deepep_composed`
backend names intentionally call DeepEP's intranode transport wrappers.

## Changes to make

Add a separate Grug implementation string instead of changing existing
intranode behavior:

- `deepep_internode` is now a valid `MoeImplementation`;
- `grouped_moe_mlp` and `moe_mlp` route it to a new
  `_moe_mlp_ep_deepep_internode_local` backend;
- the backend computes normal DeepEP dispatch layout metadata, derives
  `num_tokens_per_rdma_rank` by grouping global ranks by
  `DEEPEP_RANKS_PER_NODE`, calls `deepep_dispatch_internode`, reuses the
  existing local assignment pack/collapse FFIs around the expert MLP, and calls
  `deepep_combine_internode`;
- DeepEP internode handle tensors are included in `DEEPEP_REMAT_SAVE_NAMES`;
- the May launcher accepts `--moe-implementation deepep_internode`, forwards
  `DEEPEP_RANKS_PER_NODE`, requests the `deepep` Iris extra, and documents that
  this path needs the process-per-GPU topology.

Validation:

```text
uv run python -m py_compile \
  lib/levanter/src/levanter/grug/_moe/common.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py \
  lib/levanter/src/levanter/grug/grug_moe.py \
  experiments/grug/moe/train.py \
  experiments/grug/moe/model.py

uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q \
  -k 'deepep_internode or tokens_per_rdma_rank'

uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'internode_process_topology or process_per_gpu or internode_bootstrap_metadata or internode_runtime_status or dispatch_internode_exposes_static_jax_contract or combine_internode_exposes_static_jax_contract'

bash -n experiments/grug/moe/run_cw_may_d2560.sh

./infra/pre-commit.py \
  lib/levanter/src/levanter/grug/_moe/common.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py \
  lib/levanter/src/levanter/grug/grug_moe.py \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  experiments/grug/moe/train.py \
  experiments/grug/moe/model.py \
  experiments/grug/moe/run_cw_may_d2560.sh \
  lib/levanter/tests/grug/test_grugformer_moe.py \
  docs/debug-log-deepep-internode-jax.md
```

## Results

The first Grug routing patch passes CPU-safe checks and static FFI contract
tests, but it is not yet a complete train-step backend.

Remaining blockers before launching a real Grug training run:

1. Training currently launches one JAX process per H100x8 Iris task in the
   ordinary May path. The successful DeepEP normal-mode smoke used one JAX
   process per GPU, i.e. 8 worker processes inside each H100x8 task. The Grug
   training launcher still needs an equivalent process-per-GPU mode.
2. The internode transport wrappers are forward-callable, but unlike the
   intranode wrappers they do not yet expose custom VJPs. The forward Grug
   backend can call internode dispatch/combine, but a train step still needs the
   reverse transport:
   - dispatch backward is a combine-like operation;
   - combine backward needs a cached/reverse dispatch-like operation using the
     dispatch handles.

## Hypothesis 6: Internode dispatch backward can reuse internode combine

The intranode transport wrappers already have custom VJPs. For internode,
dispatch backward is the easier half: cotangents on received activations and
top-k weights can be routed back to the original token layout with the same
handle set consumed by `deepep_combine_internode`.

## Changes to make

- Refactor `deepep_dispatch_internode` into a raw implementation plus a
  custom-VJP wrapper.
- Save the dispatch handles needed by combine:
  `is_token_in_rank`, `recv_src_meta`, RDMA/global prefix matrices,
  `send_rdma_head`, `send_nvl_head`, and receive counts.
- In dispatch backward, materialize cotangents for `recv_x` and
  `recv_topk_weights`, call the raw internode combine implementation, and
  return gradients for `x` and `topk_weights`.
- Add a static autodiff test that monkeypatches `jax.ffi.ffi_call` and verifies
  a gradient through `deepep_dispatch_internode` calls both
  `levanter_deepep_dispatch_internode` and `levanter_deepep_combine_internode`.
- Fix the first Grug internode capacity estimate to use RDMA node count, not
  global EP rank count:
  `max_recv_tokens = local_tokens * topk * num_rdma_ranks`.

Validation:

```text
uv run python -m py_compile \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py

uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'dispatch_internode or combine_internode'

uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q \
  -k 'deepep_internode or tokens_per_rdma_rank'
```

## Results

The dispatch VJP now passes the static contract test. This removes one of the
two autodiff blockers.

Remaining blocker:

- `deepep_combine_internode` is still forward-only. Its backward path needs a
  cached/reverse internode dispatch binding, analogous to
  `_dispatch_intranode_cached_impl`. The underlying C++ already calls
  `deep_ep::internode::cached_notify(...)` and `deep_ep::internode::dispatch`
  for forward combine/dispatch, but the Marin FFI does not yet register a
  cached internode dispatch target that takes the combine handles and returns
  `grad_recv_x`.

The next implementation step is therefore the lower-level C++/FFI binding for
cached internode dispatch, then a custom VJP for `deepep_combine_internode`.
After that, run a tiny process-per-GPU Grug compile/train-step smoke with
`moe_implementation="deepep_internode"` and `expert_axis=16`.

## Hypothesis 7: Internode combine backward can reuse cached internode dispatch

Upstream DeepEP normal-mode internode dispatch supports a cached mode:

- cached dispatch takes an already-computed dispatch handle instead of layout
  counts;
- it calls `internode::cached_notify(..., is_cached_dispatch=true)` only to
  synchronize and clean buffers;
- it calls `internode::dispatch(..., is_cached_dispatch=true)` with the
  original RDMA/global prefix matrices and rank prefix sums;
- it does not need `recv_src_meta` or top-k metadata when used as the reverse of
  combine for activation gradients.

That makes the reverse of `deepep_combine_internode` a cached internode
dispatch from `grad_combined_x` back to the dispatched receive layout.

## Changes made

- Added `levanter_deepep_dispatch_internode_cached` in
  `deepep_transport_ffi.cu`.
- Added `_dispatch_internode_cached_impl` in `transport_ffi.py`.
- Wrapped `deepep_combine_internode` in a custom VJP that calls the cached
  internode dispatch binding for `grad_recv_x` and returns zero gradients for
  `recv_topk_weights`.
- Threaded `recv_gbl_rank_prefix_sum` through the public combine wrapper,
  Grug `deepep_internode` backend, remat checkpoint names, and runtime smoke
  script because cached internode dispatch needs that handle.
- Bumped the DeepEP transport build cache schema to rebuild the shared library.
- Added a static JAX autodiff contract test showing that a gradient through
  `deepep_combine_internode` calls
  `levanter_deepep_dispatch_internode_cached`.

Validation:

```text
uv run python -m py_compile \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py \
  experiments/grug/moe/launch_cw_deepep_runtime_smoke.py

uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'internode and (dispatch or combine)'

uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q \
  -k 'deepep_internode or tokens_per_rdma_rank'

./infra/pre-commit.py \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu \
  lib/levanter/src/levanter/grug/_moe/common.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py \
  lib/levanter/tests/kernels/test_deepep_availability.py \
  experiments/grug/moe/launch_cw_deepep_runtime_smoke.py \
  docs/debug-log-deepep-internode-jax.md
```

## Results

The Python/JAX autodiff contract is now complete for normal-mode internode
dispatch and combine: dispatch backward routes through internode combine, and
combine backward routes through cached internode dispatch.

Remaining work before claiming a train-step win:

1. Compile/load the rebuilt DeepEP internode transport shared library on
   CoreWeave and run the process-per-GPU runtime smoke with combine enabled.
2. Run a tiny process-per-GPU Grug train-step smoke with
   `moe_implementation="deepep_internode"`.
3. If the smoke passes, launch the real forward/backward profile target, with
   EP16 as a viable first comparison against ring/all-to-all.

## Hypothesis 8: Combine backward needs both halves of the internode dispatch handle

The first CoreWeave backward smoke after adding cached internode dispatch
failed with:

```text
jax.errors.JaxRuntimeError: NOT_FOUND: No FFI handler registered for
levanter_deepep_dispatch_internode_cached on a platform CUDA
```

That was a Python registration bug; adding
`levanter_deepep_dispatch_internode_cached` to `_register_internode_targets`
fixed it.

The next backward smoke reached the cached dispatch FFI target but failed with:

```text
jax.errors.JaxRuntimeError: INTERNAL: CUDA error: CUDA_ERROR_LAUNCH_FAILED
```

The wrapper was using the receive-side prefix matrices for cached dispatch.
Upstream DeepEP's internode handle uses both sides:

- forward combine uses `recv_rdma_channel_prefix_matrix` and
  `recv_gbl_channel_prefix_matrix`;
- cached dispatch, including combine backward, uses the original
  `rdma_channel_prefix_matrix` and `gbl_channel_prefix_matrix` plus the
  receive rank prefix sums.

## Changes made

- Expanded `deepep_combine_internode` to carry original RDMA/global channel
  prefix matrices for its custom VJP while still passing receive-side prefix
  matrices to the forward combine FFI.
- Checkpointed the original internode prefix matrices in the Grug
  `deepep_internode` path so remat can preserve the cached-dispatch handle for
  backward.
- Changed cached internode dispatch notify to pass the original top-k width to
  `internode::cached_notify`, matching upstream's cached-mode contract.
- Bumped the DeepEP transport build-cache schema to `transport_ffi_raw_dlink_v40`.

Validation:

```text
uv run python -m py_compile \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py \
  experiments/grug/moe/launch_cw_deepep_runtime_smoke.py

uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'internode and (dispatch or combine)'

uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q \
  -k 'deepep_internode or tokens_per_rdma_rank'

./infra/pre-commit.py \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu \
  lib/levanter/src/levanter/grug/_moe/common.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py \
  lib/levanter/tests/kernels/test_deepep_availability.py \
  experiments/grug/moe/launch_cw_deepep_runtime_smoke.py
```

Runtime validation:

```text
RUN_ID="DEEPEP-COMBINE-BWD-SMOKE-H100X8-RPN8-N2-cw-20260621-1625" \
  DEEPEP_RUNTIME_RUN_COMBINE_SMOKE=0 \
  DEEPEP_RUNTIME_RUN_BACKWARD_SMOKE=1 \
  bash scratch/launch_deepep_dispatch_smoke_h100x8_n2.sh
```

Result:

- Iris job:
  `/dlwh/DEEPEP-COMBINE-BWD-SMOKE-H100X8-RPN8-N2-cw-20260621-1625`
- Terminal state: `JOB_STATE_SUCCEEDED`.
- All 16 process-per-GPU workers printed `dispatch_smoke_done` and
  `backward_smoke_done`.
- Representative backward summary:
  `grad_recv_x_shape=[32, 256]`, `grad_recv_x_sum=2048.0`.

The next proof point is no longer transport-only: launch a tiny Grug
`moe_implementation="deepep_internode"` train-step smoke, preferably including
an EP16 variant because DeepEP may scale better there than ring/all-to-all.

## Hypothesis 9: Grug must use process-per-GPU topology for internode DeepEP

The standalone CoreWeave DeepEP internode smoke proved that DeepEP itself can
run cross-node on the target cluster, but the first Grug attempts exposed that
the normal Iris task layout is wrong for DeepEP internode:

- Normal Grug runs start one JAX process per 8-GPU node.
- DeepEP internode expects one rank per GPU, with node-local ranks grouped into
  RDMA nodes.

The Grug runner now has a process-per-GPU supervisor for
`moe_implementation="deepep_internode"`:

- each 2-node Iris child task starts one supervisor process;
- each supervisor prebuilds the DeepEP layout, intranode transport, and
  internode transport FFI libraries;
- each supervisor then spawns eight one-GPU JAX worker processes;
- worker process IDs are assigned as `task_index * processes_per_task + local_rank`.

Validation sequence:

- May246: process-per-GPU supervisor exposed concurrent internode transport FFI
  build/load races.
- May247: after prebuilding internode transport, exposed the same race for
  intranode transport.
- May248: after prebuilding both transport libraries, exposed the same race for
  `deepep_layout_ffi`.
- May249: after prebuilding layout plus both transports, all 16 workers started,
  DeepEP internode runtime initialized on ranks 0-15, and all ranks dispatched
  Grug train step 0.

May249 then failed at `jax.block_until_ready(metrics["train/loss"])` with:

```text
jax.errors.JaxRuntimeError: INTERNAL: DeepEP intranode JAX runtime is not initialized
```

This is not an internode DeepEP failure. The Grug internode path uses internode
dispatch/combine for transport, but it still uses the intranode-local helper
FFIs for assignment packing/collapse:

- `deepep_pack_local_assignments_from_counts`
- `deepep_collapse_local_assignments`

Those helpers call the intranode `RuntimeManager` even though they are local
post-processing helpers in this topology. The current patch initializes a
rank-1 intranode helper runtime in each one-GPU worker after the internode
runtime is initialized.

Validation:

```text
uv run python -m py_compile experiments/grug/moe/train.py
./infra/pre-commit.py experiments/grug/moe/train.py
```

Runtime validation in progress:

```text
/dlwh/iris-run-job-20260621-174628
GM2560-MAY-250-DEEPEPINTERNODE-EP16-PPG-SMOKE-N2-cw-20260621-1746
```

## Hypothesis 10: Keep local assignment pack/collapse out of the intranode FFI for internode smoke

May250 proved the rank-1 intranode helper runtime initialization was not the
right boundary. It got further than May249:

- all 16 process-per-GPU workers initialized DeepEP internode runtime;
- all 16 workers initialized the local assignment helper runtime;
- all 16 workers started and finished Grug `train_step` dispatch for step 0.

It then failed during loss materialization with:

```text
jax.errors.JaxRuntimeError: INTERNAL: PrefixLocalAssignmentCursorsKernel(counts): invalid resource handle
```

The failing kernel is not the internode dispatch/combine transport. It is the
local assignment pack helper used after internode dispatch returns received
tokens and local expert counts. That helper was still registered through the
intranode transport FFI and therefore used the intranode `RuntimeManager` inside
a process-per-GPU internode worker.

The next patch avoids that mixed-runtime boundary for the smoke test:

- `deepep_internode` uses pure-JAX local assignment pack/collapse;
- the existing intranode DeepEP path continues to use the CUDA helper FFI;
- process-per-GPU workers no longer initialize a dummy rank-1 intranode
  runtime.

Validation:

```text
uv run python -m py_compile experiments/grug/moe/train.py lib/levanter/src/levanter/grug/_moe/ep_deepep.py
./infra/pre-commit.py experiments/grug/moe/train.py lib/levanter/src/levanter/grug/_moe/ep_deepep.py
```

Runtime validation:

```text
/dlwh/iris-run-job-20260621-175625
GM2560-MAY-251-DEEPEPINTERNODE-EP16-PPG-JAXPACK-SMOKE-N2-cw-20260621-1756
scratch/20260621-175623_may251_deepep_internode_ep16_ppg_jaxpack_smoke_n2_state.json
```


May251 result:

- all 16 process-per-GPU workers initialized DeepEP internode runtime;
- all 16 workers started and finished Grug `train_step` dispatch for step 0;
- pure-JAX local assignment pack/collapse removed the previous `PrefixLocalAssignmentCursorsKernel(counts): invalid resource handle` failure.

May251 then failed during loss materialization with a DeepEP internode config assertion:

```text
jax.errors.JaxRuntimeError: INTERNAL: Failed: Assertion error /tmp/marin-deepep/DeepEP/csrc/kernels/internode.cu:1863 'num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder'
```

The next retry should keep the JAX local assignment patch and restore `DEEPEP_INTERNODE_DISPATCH_MAX_RDMA_SEND_TOKENS=16` while leaving the other reduced smoke settings in place.

## Hypothesis 11: Grug Top-K may violate the currently validated internode dispatch envelope

May252 restored `DEEPEP_INTERNODE_DISPATCH_MAX_RDMA_SEND_TOKENS=16` and kept
the pure-JAX local assignment helpers from Hypothesis 10.

Runtime validation:

```text
/dlwh/iris-run-job-20260621-180407
GM2560-MAY-252-DEEPEPINTERNODE-EP16-PPG-JAXPACK-SMOKE-N2-cw-20260621-1804
```

May252 result:

- both Iris tasks prebuilt DeepEP layout, intranode transport, and internode
  transport libraries before spawning process-per-GPU workers;
- all 16 process-per-GPU workers initialized DeepEP internode runtime with
  `process_count=16`, `num_local_ranks=8`, `num_global_ranks=16`,
  `num_nvl_bytes=134217728`, and `num_rdma_bytes=134217728`;
- all 16 workers started Grug `train_step` dispatch for step 0;
- all 16 workers finished `train_step` dispatch for step 0 and blocked on
  materializing `train/loss`.

The previous May251 DeepEP assertion is gone. May252 failed instead with:

```text
jax.errors.JaxRuntimeError: INTERNAL: DeepEP internode JAX dispatch timed out waiting for recv counters
```

This is the same transport-level symptom seen in earlier Grug attempts, but now
after the process-per-GPU topology, FFI prebuild races, local assignment FFI
boundary, and undersized RDMA-send-token assertion have all been cleared.

One clear mismatch remains between the passing standalone process-per-GPU
DeepEP smoke and the Grug run:

- the passing standalone smoke routes `topk=1`;
- the May Grug launcher defaults to `MAY_TOP_K=4`;
- the current host-only internode dispatch smoke has an explicit guard that its
  route generator expects `num_topk <= number of RDMA ranks`;
- the May252 two-node EP16 topology has only `num_rdma_ranks=2`.

The next diagnostic is therefore a two-node EP16 Grug smoke with `MAY_TOP_K=2`
and host dispatch debug enabled. This does not prove production viability, but
it should quickly distinguish "general Grug/JAX integration still broken" from
"DeepEP normal-mode path is only validated for top-k within the RDMA-rank
envelope."

Supporting wrapper change:

- `experiments/grug/moe/run_cw_may_d2560.sh` now forwards `MAY_TOP_K`,
  `LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG`, and
  `LEVANTER_DEEPEP_INTERNODE_DEBUG` to worker tasks.

Runtime validation in progress:

```text
/dlwh/iris-run-job-20260621-181429
GM2560-MAY-253-DEEPEPINTERNODE-EP16-TOPK2-PPG-SMOKE-N2-cw-20260621-1814
```

May253 result:

- all 16 process-per-GPU workers initialized DeepEP internode runtime;
- host dispatch debug confirmed the intended Grug diagnostic shape:
  `num_tokens=512`, `hidden=2560`, `num_experts=256`, `num_topk=2`;
- a subset of ranks successfully reached `internode_jax_after_wait_recv_counts`
  on the first dispatch, for example rank 0 received 1083 tokens and rank 8
  received 924 tokens;
- those ranks progressed to `internode_jax_before_dispatch_launch` and
  `internode_jax_before_combine_launch`;
- the run still failed during `train/loss` materialization with:

```text
jax.errors.JaxRuntimeError: INTERNAL: DeepEP internode JAX dispatch timed out waiting for recv counters
```

The top-k=2 diagnostic rules out the simple "Grug top-k=4 exceeds two RDMA
ranks" hypothesis. The current failure is now narrower: some ranks are making
it through the internode recv-count wait and dispatch/combine stages, while a
later dispatch in the same train step has ranks whose recv counters remain
unset (`num_recv_tokens=-1`, `num_rdma_recv_tokens=-1`, all expert counters
`-1`). The next useful probe is to log dispatch call sequence numbers and route
layout checksums per rank so we can tell whether the failing call is the second
layer, backward, or a mismatched cached dispatch/combine handle.

## Hypothesis 12: Internode normal-mode state is not reusable across Grug layers

May254 added host-only sequence ids and small routing-buffer summaries to the
DeepEP internode JAX FFI path.

Runtime validation:

```text
/dlwh/iris-run-job-20260621-182521
GM2560-MAY-254-DEEPEPINTERNODE-EP16-SEQDBG-TOPK2-N2-cw-20260621-1825
```

May254 result:

- the CUDA FFI instrumentation compiled and emitted debug records;
- call sequence 1 was the first uncached internode dispatch;
- call sequence 2 was the matching cached combine;
- call sequence 3 was the next uncached internode dispatch;
- the explicit timeout lines were all on `call_sequence=3`, e.g. ranks 5, 6,
  8, 9, 10, 12, 13, 14, and 15 reported `num_recv_tokens=-1`,
  `num_rdma_recv_tokens=-1`, and all local expert counters `-1`;
- sequence 3 route summaries were present before notify on both nodes, with
  `num_tokens_per_expert` sums of 1024 and `num_tokens_per_rank`/RDMA-rank
  sums around the expected local top-k assignment count.

This means the first layer's dispatch/combine sequence is not the failing
boundary. The first repeated use of the internode normal-mode dispatch state in
the same train step is the failing boundary. The current likely causes are:

- the DeepEP normal-mode internode runtime/handle path needs an explicit
  synchronization or cleanup between combine and the next dispatch;
- the Grug JAX graph is allowing a second dispatch to reset or reuse the shared
  host counters before prior internode work has fully retired;
- the cached combine path is not producing a reusable state for the next
  uncached dispatch in the same process-per-GPU JAX program.

The debug buffer summaries use device-to-host copies and, after the first
transport timeout poisons the CUDA stream, later workers can report
`cudaMemcpyAsync(debug int buffer summary): unspecified launch failure`. Those
are follow-on errors, not the first failure.

## Hypothesis 13: The single-layer failure is at cached dispatch/backward

May255 reduced the diagnostic to one Grug layer while keeping EP16, two nodes,
top-k 2, process-per-GPU, and host dispatch debug.

Runtime validation:

```text
/dlwh/iris-run-job-20260621-183409
GM2560-MAY-255-DEEPEPINTERNODE-EP16-SEQDBG-TOPK2-L1-N2-cw-20260621-1834
```

May255 result:

- all 16 process-per-GPU workers initialized DeepEP internode runtime;
- call sequence 1, the one-layer forward uncached dispatch, reached
  `internode_jax_before_dispatch_launch` on both nodes;
- call sequence 2, the matching cached combine, reached
  `internode_jax_before_combine_launch`;
- no normal dispatch `call_sequence=3` timeout was observed in the filtered
  logs before failure;
- the failure during `train/loss` materialization was:

```text
jax.errors.JaxRuntimeError: INTERNAL: cudaMemcpyAsync(read internode cached dispatch num_recv_tokens): unspecified launch failure
```

This suggests the two-layer May254 `call_sequence=3` timeout is not just a raw
"second forward layer" failure. In the one-layer graph, the next boundary after
forward dispatch/combine is the cached-dispatch/backward path, and the first
visible synchronous error is reading the cached dispatch count buffer. Because
DeepEP launches are asynchronous, this may still be surfacing a previous
combine failure. The next diagnostic adds debug-only stream synchronizes after
internode notify/dispatch/combine launches so the first CUDA failure is
attributed to the actual DeepEP call site instead of the next scalar read.

## Hypothesis 14: The first CUDA failure is the forward combine launch

May256 added debug-only `cudaStreamSynchronize` calls after each DeepEP
internode JAX launch site while keeping the one-layer EP16/top-k-2 diagnostic.

Runtime validation:

```text
/dlwh/iris-run-job-20260621-184220
GM2560-MAY-256-DEEPEPINTERNODE-EP16-SYNCDBG-L1-N2-cw-20260621-1842
```

May256 result:

- all 16 process-per-GPU workers initialized the DeepEP internode runtime;
- call sequence 1 reached the forward uncached dispatch boundary;
- call sequence 2 reached the matching cached combine boundary;
- the first attributed CUDA failure was:

```text
jax.errors.JaxRuntimeError: INTERNAL: cudaStreamSynchronize(DeepEP internode combine): unspecified launch failure
```

This means the previous
`cudaMemcpyAsync(read internode cached dispatch num_recv_tokens)` and later
dispatch timeout symptoms were downstream effects of the forward combine launch
poisoning the stream.

Follow-up source comparison must use Marin's pinned DeepEP revision
`7febc6e25660af0f54d95dd781ecdcd62265ecca`, not upstream `main`. Upstream
`main` has since changed the internode `notify_dispatch`/`dispatch` ABI to add a
`num_worst_tokens` argument, but the pinned revision used by Iris does not have
that argument. A May257 compile probe confirmed this by failing when the wrapper
was temporarily patched for the newer ABI. The argument patch was reverted.

The remaining target is therefore not an ABI slot shift against the pinned
DeepEP revision. The first real failing boundary remains the pinned-revision
normal-mode `internode::combine` launch, likely due to the handle metadata,
prefix/head tensors, or cached notify/combine usage produced by the Grug JAX FFI
wrapper.

May257 compile probe:

```text
/dlwh/iris-run-job-20260621-185652
/dlwh/iris-run-job-20260621-185652/grug-train-cw-may-d2560-profile-20260621-185649
```

Result: terminal FFI compile failure after adding the newer `num_worst_tokens`
argument. This proves the validated DeepEP revision's ABI is the older one and
rules out that particular mismatch as the May256 combine failure.

## Hypothesis 15: The internode runtime buffers are undersized for top-k=2 d2560

The direct process-per-GPU DeepEP top-k=2 dispatch/combine smoke failed before
combine:

```text
/dlwh/DEEPEP-COMBINE-TOPK2-SMOKE-H100X8-RPN8-N2-cw-20260621-190036
```

Failure:

```text
jax.errors.JaxRuntimeError: INTERNAL: Failed: Assertion error
/tmp/marin-deepep/DeepEP/csrc/kernels/internode.cu:340
'(nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes'
```

This is a pinned DeepEP `notify_dispatch` clean-buffer bound. For hidden=2560
bf16, top-k=2, `num_max_nvl_chunked_recv_tokens=512`, 8 NVL peers, and
`num_sms=20` (`num_channels=10`), the NVL clean range is roughly 201 MiB before
small metadata terms:

```text
align(2560 * 2 + sizeof(SourceMeta) + 2 * sizeof(int) + 2 * sizeof(float), 16)
  * 512 * 8 * 10 ~= 201 MiB
```

The previous runtime default was 64 MiB for both NVL and RDMA, so this smoke was
invalid for the May d2560 top-k=2 shape and could also explain some Grug
internode failures before the combine-specific diagnosis. The defaults were
raised to 256 MiB in Grug train startup and the DeepEP runtime smoke launchers.
The next check is the same top-k=2 standalone smoke with 256 MiB buffers.

## Hypothesis 16: d2560/top-k=2 still fails in forward combine after fixing the clean-buffer floor

The first 256 MiB-buffer retry proved that the clean-buffer assertion was a
real bug, but it ran the backward combine probe before the plain forward
combine and did not forward the DeepEP debug env into worker subprocesses:

```text
/dlwh/DEEPEP-COMBINE-TOPK2-SMOKE-H100X8-RPN8-N2-BUF256-cw-20260621-190552
```

Observed result:

- all workers initialized the process-per-GPU internode runtime;
- dispatch completed;
- the job then got stuck/fell over in the combine path before giving a clean
  plain-forward attribution.

The second retry reordered the smoke so plain forward combine runs first and
forwarded the debug env into all worker subprocesses:

```text
/dlwh/DEEPEP-COMBINE-FWDFIRST-TOPK2-SMOKE-H100X8-RPN8-N2-BUF256-cw-20260621-191133
```

Observed result:

- Iris terminal state: `JOB_STATE_FAILED`;
- all workers initialized the process-per-GPU internode runtime;
- the d2560/top-k=2 dispatch completed with the 256 MiB buffers;
- the first failing boundary was the plain forward `internode::combine` path;
- logs emitted repeated pinned-DeepEP sender timeouts such as:

```text
DeepEP combine NVL sender timeout, channel: 11, RDMA: 1, dst NVL: 7,
head: 33554432, tail: 0, start: 512, end: 512
DeepEP timeout check failed: rank = ..., thread = ..., value = 2048
```

- the synchronized JAX failure was:

```text
jax.errors.JaxRuntimeError: INTERNAL:
cudaStreamSynchronize(DeepEP internode combine): unspecified launch failure
```

Interpretation:

- The 64 MiB runtime buffers were insufficient for d2560/top-k=2 dispatch clean
  regions, but raising them to 256 MiB does not fix the remaining failure.
- The remaining bug is not Grug-specific. It reproduces in the standalone
  process-per-GPU JAX FFI dispatch/combine smoke.
- The failure is now localized to pinned DeepEP normal-mode internode
  `combine` for the d2560/top-k=2 shape.
- The timeout values suggest a metadata/capacity-shape mismatch in the combine
  handle path. One concrete next probe is to run the same smoke with
  `max_recv_tokens` and `max_rdma_recv_tokens` set to the actual active receive
  count so `send_nvl_head` is not padded to a larger capacity. If that succeeds,
  the wrapper is likely passing a padded `send_nvl_head` shape where pinned
  DeepEP's combine path expects the actual `num_rdma_recv_tokens`.

## Hypothesis 17: `combine` must use active receive counts, not padded JAX capacities

The d2560/top-k=2 standalone process-per-GPU smoke succeeded after two wrapper
fixes:

```text
/dlwh/DEEPEP-COMBINE-FWDFIRST-TOPK2-ACTUALCOUNT-H100X8-RPN8-N2-BUF256-cw-20260621-192201
```

Observed result:

- Iris terminal state: `JOB_STATE_SUCCEEDED`;
- all 16 one-GPU JAX processes initialized the DeepEP internode runtime;
- each rank routed 512 tokens to two distinct remote-node destination ranks;
- dispatch completed;
- the first plain forward combine completed;
- all worker processes printed `deepep_runtime_smoke_worker: done`.

The important wrapper changes were:

- `CombineInternode` now reads `num_recv_tokens` and `num_recv_rdma_tokens` from
  the device scalar buffers and passes the active receive count into pinned
  DeepEP's `internode::combine`, instead of passing the padded static JAX
  `recv_x` capacity.
- The standalone top-k smoke now gives each top-k slot a distinct destination
  rank and counts all top-k entries in `num_tokens_per_rank`,
  `num_tokens_per_rdma_rank`, `num_tokens_per_expert`, and `token_in_rank`.

Interpretation:

- DeepEP's cross-node normal-mode path is working on the CoreWeave H100x8
  process-per-GPU topology for this shape.
- The previous `start: 512, end: 512` / `value = 2048` combine timeout was a
  Marin/JAX wrapper contract bug around padded static capacities and incorrect
  top-k smoke metadata, not evidence that DeepEP cannot do internode routing.
- The next validation boundary is the real one-layer Grug `deepep_internode`
  path, which has the same padded-buffer issue but includes the full Grug MoE
  call graph and sharding.

## Hypothesis 18: The first Grug retry was over-requesting CPU, not failing DeepEP

The first real one-layer Grug `deepep_internode` retry after the standalone
transport success was launched as:

```text
/dlwh/iris-run-job-20260621-192839
GM2560-MAY-259S4096-W2048-B16-R1-E16M1-PALLASCE-DEEPEPINTERNODE-L1-SGD-ACTUALCOUNT-N2-cw-20260621-192836
```

It did not reach containers or DeepEP. Kueue kept both H100x8 child pods in
`SchedulingGated` with:

```text
couldn't assign flavors to pod set ... topology "infiniband" doesn't allow to
fit any of 2 pod(s). Total nodes: 32; excluded: resource "cpu": 32
```

Interpretation:

- This run requested `worker_cpu=64`, which excluded every candidate H100 node.
- The result is not evidence about Grug or DeepEP correctness.
- The pending run was stopped and should be relaunched with the same Grug shape
  but a lower CPU request, matching the standalone smoke that Kueue admitted.

## Hypothesis 19: Grug reaches DeepEP internode autograd; cached dispatch has a payload-size mismatch

The lower-CPU one-layer Grug retry was launched as:

```text
/dlwh/iris-run-job-20260621-193511
GM2560-MAY-260S4096-W2048-B16-R1-E16M1-PALLASCE-DEEPEPINTERNODE-L1-SGD-ACTUALCOUNT-N2-cw-20260621-193508
```

It reached the two-task process-per-GPU runtime:

- Kueue admitted both H100x8 tasks.
- All 16 one-GPU JAX workers initialized DeepEP internode runtime.
- The compact mesh was `replica_dcn=1,data=1,expert=16,model=1`.
- The run entered the first Grug train step and blocked on `train/loss`.

The first concrete failure was not startup, Kueue, or the forward DeepEP
internode transport. It was the backward path for DeepEP combine, implemented
as internode cached dispatch:

```text
jax.errors.JaxRuntimeError: INTERNAL:
cudaStreamSynchronize(DeepEP internode cached dispatch): unspecified launch failure

Assertion failed: /tmp/marin-deepep/DeepEP/csrc/kernels/internode.cu:717,
condition: num_tokens_to_recv_from_rdma >= 0

Assertion failed: /tmp/marin-deepep/DeepEP/csrc/kernels/internode.cu:706,
condition: start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum
```

Interpretation:

- This is the next missing validation boundary after the standalone forward
  dispatch/combine success. The earlier standalone run did not exercise
  `deepep_combine_internode`'s custom VJP unless
  `DEEPEP_RUNTIME_RUN_BACKWARD_SMOKE=1` was set.
- The cached dispatch wrapper had an internal inconsistency: `cached_notify`
  for cached dispatch used `num_topk_idx=num_topk` and
  `num_topk_weights=num_topk`, which makes DeepEP clean/stride the RDMA/NVL
  buffers as if top-k payload is present. The actual cached dispatch launch
  passes null top-k buffers and `num_topk=0`, because combine-backward only
  needs to route `grad_combined_x`.
- Pinned DeepEP's dispatch kernel computes its per-token payload stride from
  `num_topk`. Therefore notify and dispatch must agree on the top-k payload
  size. The wrapper should use zero top-k payload sizes in cached-dispatch
  notify.

The stale May260 retry was stopped after this was isolated. The next focused
probe is:

```text
/dlwh/DEEPEP-COMBINE-BWD-TOPK2-ACTUALCOUNT-H100X8-RPN8-N2-BUF256-cw-20260621-194439
```

That run exercises dispatch, forward combine, and combine backward at the same
d2560/top-k=2 two-node process-per-GPU shape, after patching cached-dispatch
notify to use zero top-k payload sizes.

## Hypothesis 20: Cached-dispatch notify must use the same payload shape as cached dispatch

The focused backward smoke succeeded:

```text
/dlwh/DEEPEP-COMBINE-BWD-TOPK2-ACTUALCOUNT-H100X8-RPN8-N2-BUF256-cw-20260621-194439
```

Observed result:

- Iris terminal state: `JOB_STATE_SUCCEEDED`;
- 2 H100x8 tasks, 16 one-GPU JAX workers;
- d2560, top-k=2, 512 local tokens, static receive capacity 2048;
- forward dispatch completed;
- forward combine completed;
- `deepep_combine_internode` custom-VJP backward completed on all workers;
- representative backward log lines showed:

```text
deepep_runtime_smoke_worker: backward_smoke_done
{"grad_recv_x_shape": [2048, 2560], "grad_recv_x_sum": 2621440.0, ...}
```

The patch changed internode cached-dispatch notify from cleaning/striding the
RDMA/NVL buffers as if top-k idx/weight payload was present to cleaning/striding
as if no top-k payload is present:

```text
cached_notify(..., num_topk_idx=0, num_topk_weights=0, is_cached_dispatch=true)
```

Interpretation:

- DeepEP normal-mode internode dispatch/combine and combine-backward now work in
  the standalone process-per-GPU JAX FFI smoke for the d2560/top-k=2 shape.
- May260's failure was caused by the wrapper using inconsistent cached-dispatch
  payload sizes, not by a fundamental inability to route cross-node.
- The next validation boundary is the real Grug one-layer path with
  `deepep_internode`, starting with top-k=2 to match the proven transport shape.

## Hypothesis 21: Real Grug top-k=2 should now pass the combine-backward boundary

Launched the real one-layer Grug validation after Hypothesis 20:

```text
/dlwh/iris-run-job-20260621-195051/grug-train-GM2560-MAY-261S4096-W2048-B16-R1-E16M1-PALLASCE-DEEPEPINTERNODE-L1-SGD-TOPK2-CACHEDBWD-N2-cw-20260621-195048
```

Config intent:

- 2 H100 nodes;
- process-per-GPU DeepEP internode runtime;
- `expert_axis=16`, `model_axis=1`, `replica_axis=1`;
- one Grug layer, batch 16, sequence length 4096, sliding window 2048;
- `moe=deepep_internode`, `top_k=2`;
- SGD optimizer, Pallas CE, FA4 CuTe attention;
- ttl=7d output prefix.

This is specifically testing whether the cached-dispatch notify fix that passed
the standalone combine-backward smoke also works inside Grug's real
forward/backward/autograd path.

May261 failed before it reached the model path:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'nvcc'
```

The standalone DeepEP runtime smoke used the CUDA devel task image
`ghcr.io/marin-community/iris-task-cuda-devel:969c0e3`; the Grug launch used
the default task image, which did not have `nvcc` on `PATH` for the supervisor's
DeepEP FFI prebuild. The May261 parent and child were stopped because retries
would hit the same deterministic prebuild failure.

Relaunched the same top-k=2 Grug validation on the CUDA devel image:

```text
/dlwh/iris-run-job-20260621-195318/grug-train-GM2560-MAY-262S4096-W2048-B16-R1-E16M1-PALLASCE-DEEPEPINTERNODE-L1-SGD-TOPK2-CACHEDBWD-N2-cw-20260621-195316
```

May262 reached the real Grug/JAX path:

- DeepEP FFI prebuild succeeded on the CUDA devel image;
- all 16 process-per-GPU workers initialized the internode runtime;
- compact mesh was `replica_dcn=1,data=1,expert=16,model=1`;
- W&B run was created;
- all workers started and finished dispatching `train_step` step 0.

It then failed while blocking `metrics["train/loss"]`:

```text
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclGroupEnd() failed:
unhandled cuda error ... Last NCCL warning(error) log entry (may be unrelated)
'Cuda failure 2 'out of memory''.
```

Interpretation:

- This is no longer the cached-dispatch DeepEP assertion from May260.
- The model/autograd path now reaches a compiled step with DeepEP internode
  initialized.
- The next hypothesis is that JAX's default memory fraction leaves too little
  HBM headroom for NCCL plus the DeepEP NVL/RDMA buffers in the process-per-GPU
  runtime. Retry the same shape with a lower `XLA_PYTHON_CLIENT_MEM_FRACTION`
  before shrinking batch/sequence.

## Hypothesis 22: DeepEP internode Grug works when JAX leaves enough HBM for NCCL/runtime buffers

Relaunched the same top-k=2 one-layer Grug validation with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`:

```text
/dlwh/iris-run-job-20260621-200035/grug-train-GM2560-MAY-263S4096-W2048-B16-R1-E16M1-PALLASCE-DEEPEPINTERNODE-L1-SGD-TOPK2-XLA80-N2-cw-20260621-200032
```

Observed:

- DeepEP FFI prebuild succeeded;
- all 16 process-per-GPU workers initialized the internode runtime;
- compact mesh was `replica_dcn=1,data=1,expert=16,model=1`;
- step 0, 1, and 2 all completed `train/loss` blocking;
- no cached-dispatch assertion and no NCCL OOM recurrence.

Representative step-2 direct metrics:

```text
loss=11.791821479797363
duration ~= 0.0481-0.0487s
tokens_per_second ~= 1.34M-1.36M
mfu ~= 20.7-21.0
```

Interpretation:

- DeepEP internode now works inside the real Grug model/autograd path for the
  EP16, two-node, one-layer, batch-16, top-k=2 validation shape.
- May262's NCCL OOM was caused by insufficient HBM headroom for NCCL/DeepEP
  alongside JAX's default memory reservation; `0.80` was enough for this
  validation.
- Next validation should move back to the production top-k setting (`top_k=4`)
  while keeping `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`.

## Hypothesis 23: Top-k=4 DeepEP internode works with the same HBM headroom

Launched the production-router top-k validation:

```text
/dlwh/iris-run-job-20260621-201109/grug-train-GM2560-MAY-264S4096-W2048-B16-R1-E16M1-PALLASCE-DEEPEPINTERNODE-L1-SGD-TOPK4-XLA80-N2-cw-20260621-201106
```

Config intent:

- 2 H100 nodes;
- process-per-GPU DeepEP internode runtime;
- `expert_axis=16`, `model_axis=1`, `replica_axis=1`;
- one Grug layer, batch 16, sequence length 4096, sliding window 2048;
- `moe=deepep_internode`, `top_k=4`;
- SGD optimizer, Pallas CE, FA4 CuTe attention;
- CUDA devel task image `ghcr.io/marin-community/iris-task-cuda-devel:969c0e3`;
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`;
- DeepEP NVL/RDMA runtime buffers 256 MiB each;
- ttl=7d output prefix.

This is specifically testing whether the May263 successful top-k=2 real Grug
path extends to the production top-k setting without a new DeepEP payload,
buffer-size, or NCCL HBM issue.

Observed:

- the child job was created and both H100-node tasks ran;
- DeepEP FFI prebuild succeeded on both tasks;
- process-per-GPU supervisors started for both tasks with `process_count=16`
  and `processes_per_task=8`;
- step 1 and step 2 both completed `train/loss` blocking across all ranks;
- DeepEP internode shutdown reached `internode_shutdown_barrier_done` and
  `internode_shutdown_skip_nvshmem_finalize` on observed ranks;
- Iris terminal state was `JOB_STATE_SUCCEEDED` for both parent and child;
- no cached-dispatch assertion, buffer-size error, or NCCL OOM was observed.

Representative step-2 direct metrics:

```text
loss=11.791836738586426
duration ~= 0.0716-0.0729s
tokens_per_second ~= 899k-915k
mfu ~= 14.50-14.76
```

Interpretation:

- DeepEP internode works inside the real Grug model/autograd path for the EP16,
  two-node, one-layer, batch-16, top-k=4 validation shape.
- The top-k=4 shape is slower than the May263 top-k=2 validation
  (`~0.072s` vs `~0.048s` representative step time). That is expected to some
  degree because routing volume roughly doubles, but it is now a concrete
  performance target rather than a correctness blocker.
- The next useful experiment is an apples-to-apples EP16 top-k=4 comparison
  against ring/all-to-all or a short DeepEP profile at the same shape, with
  host-dispatch debug disabled to avoid log volume.

Caveat:

- W&B finish still emitted a background tracker timeout/`HandleAbandonedError`
  after uploading history steps 0-2. Iris still marked the job succeeded. Treat
  this as a W&B cleanup issue, not a DeepEP training failure.

## Hypothesis 24: DeepEP internode should beat the existing top-k=4 all-to-all baseline

Launched an apples-to-apples baseline using the same EP16, two-node,
one-layer, batch-16, top-k=4 shape as May264, but with the existing ragged
all-to-all MoE implementation instead of DeepEP:

```text
/dlwh/iris-run-job-20260621-202144/grug-train-GM2560-MAY-265S4096-W2048-B16-R1-E16M1-PALLASCE-RAGGEDA2A-L1-SGD-TOPK4-XLA80-N2-cw-20260621-202141
```

Config intent:

- 2 H100 nodes;
- compact mesh `replica_dcn=1,data=1,expert=16,model=1`;
- one Grug layer, batch 16, sequence length 4096, sliding window 2048;
- `moe=ragged_all_to_all`, `top_k=4`;
- SGD optimizer, Pallas CE, FA4 CuTe attention;
- CUDA devel task image `ghcr.io/marin-community/iris-task-cuda-devel:969c0e3`;
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`;
- ttl=7d output prefix.

Observed:

- the child job was created and both H100-node tasks ran;
- compact mesh was confirmed as
  `{'replica_dcn': 1, 'data': 1, 'expert': 16, 'model': 1}`;
- step 0, 1, and 2 completed `train/loss` blocking on both tasks;
- no OOM, NCCL, rendezvous, or traceback signal was observed in the bounded log
  slice through the measured steps;
- after step 2 the job entered the same W&B finish path that can remain running
  after measurements have already been uploaded;
- because the wrapper remained running in W&B cleanup after the metrics were
  uploaded, the job was manually stopped to free the two H100 nodes. Treat the
  May265 terminal state as cleanup-killed after measurement, not as a training
  failure.

Representative step-2 direct metrics:

```text
loss=11.791834831237793
duration ~= 0.0568-0.0576s
tokens_per_second ~= 1.137M-1.154M
mfu ~= 18.34-18.61
```

Comparison to May264 DeepEP internode top-k=4:

```text
May264 DeepEP internode: duration ~= 0.0716-0.0729s, MFU ~= 14.50-14.76
May265 ragged all-to-all: duration ~= 0.0568-0.0576s, MFU ~= 18.34-18.61
```

Interpretation:

- DeepEP internode is now a working cross-node EP16 implementation, but it is
  not yet faster than the existing ragged all-to-all path on this small
  one-layer, top-k=4 validation shape.
- For this probe, ragged all-to-all is roughly 20-25% faster by step time.
- The next useful checks are either a ring baseline at the same shape or a
  profile/larger-layer DeepEP run to find whether DeepEP wins only when MoE
  communication is a larger fraction of the step.

## Hypothesis 25: Ring is slower than the all-to-all/DeepEP EP16 baselines

Launched a third apples-to-apples baseline using the same EP16, two-node,
one-layer, batch-16, top-k=4 shape, but with the ring MoE implementation:

```text
/dlwh/iris-run-job-20260621-203122/grug-train-GM2560-MAY-266S4096-W2048-B16-R1-E16M1-PALLASCE-RING-L1-SGD-TOPK4-XLA80-N2-cw-20260621-203119
```

Config intent:

- 2 H100 nodes;
- compact mesh `replica_dcn=1,data=1,expert=16,model=1`;
- one Grug layer, batch 16, sequence length 4096, sliding window 2048;
- `moe=ring`, `top_k=4`;
- SGD optimizer, Pallas CE, FA4 CuTe attention;
- CUDA devel task image `ghcr.io/marin-community/iris-task-cuda-devel:969c0e3`;
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`;
- ttl=7d output prefix.

Observed:

- the child job was created and both H100-node tasks ran;
- compact mesh was confirmed as
  `{'replica_dcn': 1, 'data': 1, 'expert': 16, 'model': 1}`;
- step 0, 1, and 2 completed `train/loss` blocking on both tasks;
- the child and parent Iris jobs both reached `JOB_STATE_SUCCEEDED`;
- no OOM, NCCL, rendezvous, or training-path traceback signal was observed in
  the measured bounded log slice.

Representative step-2 direct metrics:

```text
loss=11.79183292388916
duration ~= 0.0464-0.0470s
tokens_per_second ~= 1.396M-1.411M
mfu ~= 22.50-22.76
```

Comparison at the same EP16, two-node, one-layer, batch-16, top-k=4 shape:

```text
May264 DeepEP internode: duration ~= 0.0716-0.0729s, MFU ~= 14.50-14.76
May265 ragged all-to-all: duration ~= 0.0568-0.0576s, MFU ~= 18.34-18.61
May266 ring: duration ~= 0.0464-0.0470s, MFU ~= 22.50-22.76
```

Interpretation:

- On this small EP16 validation, ring is currently the best MoE transport path,
  followed by ragged all-to-all, then DeepEP internode.
- This contradicts the expectation that DeepEP should automatically win once
  the expert axis crosses nodes. The likely remaining explanations are
  DeepEP/runtime overhead at this small per-layer payload, a layout/copy cost in
  the JAX wrapper, or a shape where ring's simpler compiled path is just better.
- The next useful DeepEP work is not more correctness smoke testing; it is a
  profile or larger-layer run that separates DeepEP dispatch/combine time from
  wrapper/layout overhead and checks whether the ranking changes when the MoE
  path is a larger fraction of total step time.

## Hypothesis 26: DeepEP internode needs a readable profile at the EP16 top-k=4 shape

May268 attempted a readable DeepEP profile for the same EP16, two-node,
one-layer, batch-16, top-k=4 shape, but it failed during profiler finalization:

```text
gzip.BadGzipFile
```

The process-per-GPU DeepEP supervisor launches eight local JAX processes inside
each Iris task. Those processes shared the same local profile directory, so
`jax.profiler.stop_trace()` could race while compressing trace files. The fix is
to write local worker traces under per-process subdirectories:

```text
<trainer.log_dir>/<run_id>/profiler/process_<jax.process_index()>
```

May269 reran after that change and succeeded. It also confirmed the same
performance level as May264:

```text
run_id=MAY269...
global_step=5
train/loss=11.791577339172363
throughput/duration=0.06965329893864691
throughput/tokens_per_second=940888.6728786016
throughput/mfu=15.17063602931975
```

The per-process profile layout fix worked: task logs showed
`profiler/process_0..7` directories with XPlane and trace JSON outputs. However,
the W&B artifact upload did not preserve the profiler. The run finished with
only source/config/requirements artifacts attached.

May271 added a direct profiler tarball upload path, intended to write:

```text
s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-profiles/<run_id>-profiler.tgz
```

May271 also succeeded and reproduced May269 performance:

```text
run_id=MAY271-DEEPEP-EP16-TK4-PROFREM-N2-20260621-211512
parent=/dlwh/iris-run-job-20260621-211515
global_step=5
train/loss=11.791577339172363
throughput/duration=0.06965248915366828
throughput/tokens_per_second=940899.6117197416
throughput/mfu=15.170812404253683
```

But May271 logs did not contain `Creating profiler tarball`, `Uploading
profiler tarball`, or `Uploaded profiler tarball`. The root cause was an
environment propagation gap: Grug child jobs only forwarded selected runtime
prefixes from the launcher. `MAY_UPLOAD_PROFILER_REMOTE_PREFIX` was set on the
dispatcher job but was not reliably forwarded into the Fray training child and
process-per-GPU local workers.

Changes made after May271:

- `experiments/grug/dispatch.py` now forwards `MAY_UPLOAD_PROFILER_*` to Grug
  training children.
- `experiments/grug/moe/run_cw_may_d2560.sh` now forwards and prints
  `MAY_UPLOAD_PROFILER_REMOTE_PREFIX`.
- `experiments/grug/moe/train.py` creates a profiler tarball and uploads it via
  `fsspec` when `MAY_UPLOAD_PROFILER_REMOTE_PREFIX` is present.

Validation:

```text
uv run python -m py_compile experiments/grug/dispatch.py experiments/grug/moe/train.py
bash -n experiments/grug/moe/run_cw_may_d2560.sh
```

May272 is the current validation run for the profile-upload path:

```text
parent=/dlwh/iris-run-job-20260621-212546
run_id=MAY272-DEEPEP-EP16-TK4-PROFREM-N2-20260621-212543
remote_prefix=s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-profiles
```

The next evidence needed is the exact uploaded tar path, followed by an
`lib/marin/tools/profile_summary.py` summary that separates DeepEP
dispatch/combine time from surrounding layout/copy/wrapper overhead.

May272 completed successfully and validated the remote profile upload path:

```text
parent=/dlwh/iris-run-job-20260621-212546
child=/dlwh/iris-run-job-20260621-212546/grug-train-MAY272-DEEPEP-EP16-TK4-PROFREM-N2-20260621-212543
run_id=MAY272-DEEPEP-EP16-TK4-PROFREM-N2-20260621-212543
remote_profile=s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-profiles/MAY272-DEEPEP-EP16-TK4-PROFREM-N2-20260621-212543-profiler.tgz
```

Representative W&B summary:

```text
global_step=5
train/loss=11.791579246520996
throughput/duration=0.07006779499351978
throughput/tokens_per_second=935322.7114691007
throughput/mfu=15.080891963809387
throughput/mean_mfu=9.673914083277738
```

The profile was downloaded to:

```text
scratch/profiles/may272_deepep_ep16_tk4
scratch/profile_reports/issue4312_may272_deepep_ep16
```

Important caveat: the tarball was created by `jax.process_index()==0`, so it
captured task 0's local `process_0..7` traces only. That is enough to identify
local rank staging costs, but not enough to prove full two-node symmetry.

The May272 profile makes the current bottleneck more concrete. On every local
rank in the captured task, the largest repeated costs were:

```text
input_reduce_fusion_1:   ~23 ms per local process across profiled steps
input_scatter_fusion_8:  ~21-22 ms per local process across profiled steps
NCCL all-reduce:         ~21-23 ms per local process across profiled steps
```

The large `input_reduce_fusion_1` occurs immediately after
`moe_ep_deepep_internode/dispatch/deepep_dispatch_internode_transport/ffi_call`
and after `deepep_collapse_local_assignments/broadcast_in_dim`. The large
`input_scatter_fusion_8` occurs immediately before the first Pallas
`w13_ragged_dot` expert GEMM. This points at local assignment
pack/collapse/materialization around the DeepEP transport, not at the DeepEP
internode kernels alone.

Visible DeepEP kernels were smaller than the surrounding staging costs. In the
same profile, representative per-process totals were roughly:

```text
deep_ep::internode::notify_dispatch:  ~4-22 ms, depending on local rank
deep_ep::internode::dispatch:         ~2.2 ms
deep_ep::internode::combine:          ~2.7 ms
deep_ep::internode::cached_notify:    ~1.1-2.9 ms
```

Interpretation:

- DeepEP internode is crossing nodes and producing correct training metrics.
- The current performance loss is likely not "DeepEP cannot do cross-node"; it
  is the JAX/Grug staging path around DeepEP's received token assignment.
- For this shape, DeepEP removes the ring/ragged transport structure but pays a
  large local scatter/reduce tax before the expert matmuls, plus the normal
  non-MoE all-reduce sync.

## Hypothesis 27: Use existing FFI pack/collapse for the internode assignment boundary

The internode path was still using JAX implementations for both sides of the
local assignment boundary:

```text
_pack_local_assignments_from_counts_jax(...)
_collapse_local_assignments_jax(...)
```

The intranode and composed paths already have FFI helpers for these operations:

```text
deepep_pack_local_assignments_from_counts(...)
deepep_collapse_local_assignments(...)
```

Change made:

- `lib/levanter/src/levanter/grug/_moe/ep_deepep.py` now uses
  `deepep_pack_local_assignments_from_counts` and
  `deepep_collapse_local_assignments` in `_moe_mlp_ep_deepep_internode_local`.
- The DeepEP internode transport itself is unchanged.

Local validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/grug/_moe/ep_deepep.py
```

May273 is the validation run for this change:

```text
parent=/dlwh/iris-run-job-20260621-214104
run_id=MAY273-DEEPEP-EP16-TK4-FFIPACK-PROFREM-N2-20260621-2140...
```

May273 failed before child creation because the wrapper launch omitted the
diagnostic cross-node expert-axis toggle:

```text
ValueError: MAY_EXPERT_AXIS * MAY_MODEL_AXIS must divide the 8 GPUs on each worker so expert/model groups stay local; got 16 * 1 = 16.
```

This is not a DeepEP runtime or training failure.

May274 is the corrected relaunch with the same FFI pack/collapse code change
and `--allow-cross-node-expert-axis true`:

```text
parent=/dlwh/iris-run-job-20260621-214310
run_id=MAY274-DEEPEP-EP16-TK4-FFIPACK-PROFREM-N2-20260621-2143...
```

May274 reached the child job but failed before training while prebuilding the
newly used DeepEP layout FFI:

```text
RuntimeError: DEEPEP_SRC_ROOT must point at a DeepEP checkout before using the DeepEP JAX FFI layout kernel.
```

This is also not a DeepEP runtime/training result. The launch did not set
`MAY_DEEPEP_BOOTSTRAP_SOURCE=true`, so the worker could not clone the validated
DeepEP source checkout needed by the FFI pack/collapse libraries.

May275 is the corrected relaunch with both cross-node expert-axis and DeepEP
source bootstrap enabled:

```text
parent=/dlwh/iris-run-job-20260621-214611
run_id=MAY275-DEEPEP-EP16-TK4-FFIPACK-PROFREM-N2-20260621-2146...
```

Expected result if the hypothesis is right:

- the large JAX `input_reduce_fusion_1` and `input_scatter_fusion_8` staging
  kernels should disappear or shrink materially;
- DeepEP FFI pack/collapse kernels may appear instead;
- step time should move toward, or at least closer to, the ragged/ring
  baselines if local staging was the dominant incremental cost.

If May273 fails or does not improve, the next likely explanations are:

- the existing FFI pack/collapse primitive is not shape-compatible with the
  internode receive metadata;
- its backward path still materializes similar scatter/reduce work;
- the remaining performance gap is mainly non-MoE all-reduce or DeepEP
  notify/dispatch/combine overhead rather than local assignment staging.

### 2026-06-21 14:58 PT: May275 failure and runtime-free local assignment FFI fix

May275 reached the child and initialized the DeepEP internode process-per-GPU
runtime on all 16 ranks:

```text
InternodeRuntimeStatus(initialized=True, process_count=16, num_local_ranks=8, num_global_ranks=16, ...)
```

It then failed before first metrics while blocking on step-0 `train/loss`:

```text
jax.errors.JaxRuntimeError: INTERNAL: DeepEP intranode JAX runtime is not initialized
```

Root cause: the reused local assignment FFI helpers
`deepep_pack_local_assignments_from_counts` and
`deepep_collapse_local_assignments` were written beside the intranode DeepEP
transport and unnecessarily queried `RuntimeManager::RuntimeForCurrentDevice()`.
The internode Grug path initializes the internode runtime, not the separate
intranode runtime. The count-seeded local assignment kernels themselves are
local CUDA kernels and do not need the intranode runtime.

Patch:

- `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
  - removed `RuntimeManager::RuntimeForCurrentDevice()` from
    `PackLocalAssignmentsFromCounts`;
  - changed `CollapseLocalAssignments` to read `num_recv_tokens` directly with
    `ReadDeviceScalarInt` instead of `ReadRecvCount(runtime, ...)`.

Local validation:

```text
uv run python -m py_compile \
  experiments/grug/dispatch.py \
  experiments/grug/moe/train.py \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py
bash -n scratch/launch_may276_deepep_ep16_ffipack_runtimefree_n2.sh
```

Relaunch:

```text
parent=/dlwh/iris-run-job-20260621-215535
run_id=MAY276-DEEPEP-EP16-TK4-FFIPACK2-PROFREM-N2-20260621-2155
state=scratch/20260621-1455_may276_deepep_ffipack_runtimefree_state.json
```

At first check, the child existed and was still prebuilding DeepEP FFI/CUDA
libraries. No W&B run or first metrics had appeared yet. The next check should
verify whether May276 passes the May275 runtime error, then compare profile
staging kernels against May272.

### 2026-06-21 15:05 PT: May276 passed runtime init, failed CUDA stream/device launch

May276 reached W&B but failed before first metrics:

```text
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY276-DEEPEP-EP16-TK4-FFIPACK2-PROFREM-N2-20260621-2155
state=failed
```

It did get past May275's intranode-runtime failure. The logs showed all ranks
initialized the DeepEP internode runtime, entered train step 0, and selected
`pallas_gpu` CE. The new failure was:

```text
jax.errors.JaxRuntimeError: INTERNAL: PrefixLocalAssignmentCursorsKernel(counts): invalid resource handle
```

Interpretation: the count-seeded local assignment helper is now correctly
runtime-free, but the standalone FFI helper can launch its CUDA kernels while
the process current CUDA device is stale relative to XLA's platform stream. The
intranode transport path usually touches a DeepEP runtime before launching
kernels; this internode local staging helper no longer does, so it needs its
own stream/device hygiene.

Patch:

- added `CudaDeviceGuard`, which derives the target CUDA device from an FFI
  buffer pointer with `cudaPointerGetAttributes`, sets the current CUDA device
  before stream work, and restores the original device afterward;
- wrapped count-seeded local assignment pack, collapse, and assignment-gradient
  helpers;
- removed the intranode runtime dependency from `AssignmentGradients` as well,
  since that backward helper would otherwise fail after forward.

Local validation:

```text
uv run python -m py_compile \
  experiments/grug/dispatch.py \
  experiments/grug/moe/train.py \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py
```

### 2026-06-21 19:00 PT: Current head - May295 result and May297 profile target

May295 (`FFICOLLAPSE`, B16) succeeded and showed that the explicit-collapse FFI
plus normal DeepEP combine recovers the May293/May294 fused-combine regression:

```text
run_id=MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142
profile_artifact=MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142-profiler:v0
late_steps_3_5_avg=13.8586 MFU, 128159 tok/s, 0.51145s
final_step_5=13.7576 MFU, 127224.7 tok/s, 0.51512s
```

Interpretation: May295 is within about 0.7% of May292 baseline and recovers
essentially the whole fused-combine regression. Continue with explicit collapse
FFI plus normal combine; do not pursue the current fused-combine path unless its
internals are redesigned.

May297 is the current throughput-shaped profile target:

```text
script=scratch/launch_may297_deepep_ep16_fusedbwd_ffiexplicit_l26_b256_n2.sh
parent=/dlwh/iris-run-job-20260622-015354
child=/dlwh/iris-run-job-20260622-015354/grug-train-MAY297-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-B256-N2-20260622-0153
run_id=MAY297-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-B256-N2-20260622-0153
shape=2 nodes, 16 H100s, EP16, global batch 256, 16 sequences/device
profile=command buffers disabled, HLO proto enabled, profiler steps 3-4
```

Current May297 evidence as of 19:00 PT:

- Iris parent and child are running, with both child tasks running;
- W&B exists and is running, but no history rows yet;
- DeepEP internode runtime initialized on all 16 ranks;
- compact mesh is `{'replica_dcn': 1, 'data': 1, 'expert': 16, 'model': 1}`,
  with `batch_shards=16`, matching the intended B256 / 16 seq-device shape;
- no OOM, traceback, rendezvous failure, or terminal failure seen yet.

Fallback if May297 terminally fails due memory before a useful profile:

```text
script=scratch/launch_may296_deepep_ep16_fusedbwd_ffiexplicit_l26_b128_n2.sh
shape=2 nodes, 16 H100s, EP16, global batch 128, 8 sequences/device
```

### 2026-06-21 18:59 PT: May295 explicit-collapse recovers fused-combine regression

The B16 explicit-collapse run succeeded end to end:

```text
parent=/dlwh/iris-run-job-20260622-014258
run_id=MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142
profile_artifact=MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142-profiler:v0
```

Performance:

```text
late steady steps 3-5 avg: 13.8586 MFU, 128159 tok/s, 0.51145s
final step 5: 13.7576 MFU, 127224.7 tok/s, 0.51512s
best late step 4: 14.1085 MFU, 130469.3 tok/s, 0.50231s
```

Interpretation:

- May295 is within about 0.7% of the May292 baseline
  (`13.9573 MFU`, `129071.8 tok/s`, `0.50777s`).
- May295 recovers essentially the regression introduced by May293/May294
  fused-combine (`~10.7 MFU`, `~99k tok/s`, `~0.66s`).
- The current best direction is therefore explicit local collapse FFI followed
  by normal DeepEP internode combine, not the fused-combine FFI.
- B16 remains a latency/debug shape. Throughput/profile work should use B128 or
  B256.

### 2026-06-21 19:00 PT: May297 B256 throughput-shaped profile in flight

Launched the 16 sequences/device profile:

```text
script=scratch/launch_may297_deepep_ep16_fusedbwd_ffiexplicit_l26_b256_n2.sh
parent=/dlwh/iris-run-job-20260622-015354
child=/dlwh/iris-run-job-20260622-015354/grug-train-MAY297-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-B256-N2-20260622-0153
run_id=MAY297-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-B256-N2-20260622-0153
shape=2 nodes, 16 H100s, EP16, global batch 256, 16 sequences/device
profile=command buffers disabled, HLO proto enabled, profiler steps 3-4
```

Current evidence:

- parent and child are running, with both child tasks running;
- W&B exists and is running, but no history rows yet;
- DeepEP internode runtime initialized on all 16 ranks;
- compact mesh is `{'replica_dcn': 1, 'data': 1, 'expert': 16, 'model': 1}`,
  with `batch_shards=16`, matching the intended B256 / 16 seq-device shape;
- no OOM, traceback, rendezvous failure, or terminal failure seen yet.

Fallback if May297 terminally fails due memory before a useful profile:

```text
script=scratch/launch_may296_deepep_ep16_fusedbwd_ffiexplicit_l26_b128_n2.sh
shape=2 nodes, 16 H100s, EP16, global batch 128, 8 sequences/device
```

### 2026-06-21 18:57 PT: explicit-collapse FFI throughput profile at B256

After the B16 explicit-collapse diagnostic run showed that the path is alive
but far too latency dominated for throughput analysis, launched a 16
sequences/device profile:

```text
script=scratch/launch_may297_deepep_ep16_fusedbwd_ffiexplicit_l26_b256_n2.sh
parent=/dlwh/iris-run-job-20260622-015354
child=/dlwh/iris-run-job-20260622-015354/grug-train-MAY297-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-B256-N2-20260622-0153
run_id_prefix=MAY297-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-B256-N2-20260622-0153
shape=2 nodes, 16 H100s, EP16, global batch 256, 16 sequences/device
profile=command buffers disabled, HLO proto enabled, profiler steps 3-4
```

First checks:

- parent and child are `JOB_STATE_RUNNING`, with both child tasks running;
- no W&B run had appeared yet;
- recent logs were still DeepEP/JAX FFI build warnings, with no OOM,
  traceback, rendezvous failure, or first train-step dispatch yet.

Fallback if May297 terminally fails due memory before a useful profile:

```text
script=scratch/launch_may296_deepep_ep16_fusedbwd_ffiexplicit_l26_b128_n2.sh
shape=2 nodes, 16 H100s, EP16, global batch 128, 8 sequences/device
```

### 2026-06-21 18:35 PT: May292 vs May293 fused-combine comparison

May292 is the current L26 reference for the fused backward-assignment-gradient
path without fused combine:

```text
parent=/dlwh/iris-run-job-20260622-004616
run_id=MAY292-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-L26-N2-20260622-0046
final_mfu=13.930986
final_tokens_per_second=128828
final_duration=0.508709s
```

Process-0 profile summary:

```text
summary=scratch/profile_summaries/may292_deepep_fusedbwd_l26_process0.json
report=scratch/profile_reports/may292_deepep_fusedbwd_l26_process0.md
compute=639.75 ms
comm=90.064 ms
target_residual=input_scatter_fusion_52, 19.793 ms / 52 calls
target_path=.../combine/deepep_collapse_local_assignments_jax/scatter-add
```

May293 enabled `LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE=fused_combine` to
move local combine-side collapse into the DeepEP FFI:

```text
parent=/dlwh/iris-run-job-20260622-011108
run_id=MAY293-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FUSEDCOMBINE-L26-N2-20260622-0111
final_mfu=10.7871
final_tokens_per_second=99754.6
final_duration=0.65697s
```

The targeted JAX scatter disappeared:

```text
input_scatter_fusion_52: 19.793 ms -> 0
```

but the total step regressed by roughly 30% wall time. Process-0 comparison:

```text
May292 steady avg: 13.9573 MFU, 129071.8 tok/s, 0.50777s
May293 steady avg: 10.7069 MFU, 99012.7 tok/s, 0.66223s
comparison=scratch/profile_reports/may292_vs_may293_deepep_fusedcombine_process0_compare.txt
May293 other/idle bucket=643.931 ms, up from 326.227 ms
May293 moe bucket=416.193 ms, up from 400.976 ms
May293 loss_xent bucket=49.443 ms, roughly unchanged
```

Interpretation: fused combine is not a win as implemented. It removes the
visible JAX scatter, but the internal `cudaMallocAsync`/`cudaMemsetAsync`
temporary collapse plus cached-notify/combine path appears heavier and creates
more idle time. The next useful experiment is either:

- repair fused-combine overhead directly; or
- use an explicit collapse FFI followed by the normal DeepEP combine, which
removes XLA scatter without coupling the collapse to DeepEP's combine runtime
allocation/notify path.

### 2026-06-21 18:40 PT: May294 fused-combine backward-collapse FFI launch

Added a new backward FFI target for the fused-combine local collapse:

```text
target=levanter_deepep_collapse_local_assignments_internode_bwd
python=lib/levanter/src/levanter/kernels/deepep/transport_ffi.py
cuda=lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu
test=lib/levanter/tests/kernels/test_deepep_availability.py
```

Local validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/kernels/deepep/transport_ffi.py
git diff --check -- \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu \
  lib/levanter/tests/kernels/test_deepep_availability.py
uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'combine_internode_with_local_collapse or assignment_gradient'
uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'combine_internode_with_local_collapse_backward_contract'
```

Launched:

```text
parent=/dlwh/iris-run-job-20260622-013012
child=/dlwh/iris-run-job-20260622-013012/grug-train-MAY294-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FUSEDCOMBINE-BWDFUSED-L26-N2-20260622-0130
run_id=MAY294-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FUSEDCOMBINE-BWDFUSED-L26-N2-20260622-0130
```

At first check the child had both tasks running, W&B existed, and DeepEP
internode runtime initialized for 16 ranks. No train metrics were present yet.
This run should answer whether the backward JAX-collapse fallback was a major
part of May293's regression. If May294 is still materially worse than May292,
do not keep tuning `fused_combine` blindly; pivot to an explicit collapse-FFI
plus normal-combine experiment or eliminate the internal fused-combine temp
allocation/zeroing path.

May294 steady rows arrived while the run was still active:

```text
global_step=2 duration=0.685378s tokens/s=95620.2 MFU=10.3400
global_step=3 duration=0.667625s tokens/s=98162.9 MFU=10.6150
global_step=4 duration=0.669628s tokens/s=97869.3 MFU=10.5832
```

This is essentially May293 performance, not a recovery toward May292. The
fused-combine regression is therefore not primarily the backward JAX-collapse
fallback.

Follow-up patch prepared while May294 runs:

```text
mode=LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE=ffi
script=scratch/launch_may295_deepep_ep16_fusedbwd_ffiexplicit_l26_n2.sh
```

This mode uses:

1. standalone local collapse FFI registered against the internode transport
   library (`levanter_deepep_collapse_local_assignments_internode`);
2. the new CUDA backward-collapse FFI
   (`levanter_deepep_collapse_local_assignments_internode_bwd`);
3. the normal DeepEP internode combine path.

The point is to isolate "remove the JAX scatter" from "change DeepEP combine
internals." If May294 confirms fused-combine is still slow, launch May295
before doing deeper fused-combine runtime surgery.

May295 was launched:

```text
parent=/dlwh/iris-run-job-20260622-014258
child=/dlwh/iris-run-job-20260622-014258/grug-train-MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142
run_id=MAY295-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-FFICOLLAPSE-L26-N2-20260622-0142
```

At first check the child existed and was assigned/pending.

### 2026-06-21 17:35 PT: Prepare fused dispatch-backward assignment-gradient/combine path

May289 showed that the standalone assignment-gradient FFI is not a valid drop-in
replacement for the May285 JAX `scatter-add`:

```text
run_id=MAY289-DEEPEP-EP16-TK4-FUSEDPACK-ASSIGNGRADFFI-N2-20260622-0012
error=jax.errors.JaxRuntimeError: INTERNAL: AssignmentGradRecvXKernel: invalid resource handle
```

Hypothesis: the assignment-gradient work should be fused into the internode
dispatch-backward transport boundary instead of materializing a separate
receive-gradient tree through XLA. The new opt-in mode
`LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE=fused` does this by calling
one FFI target, `levanter_deepep_dispatch_internode_bwd_fused`, which:

1. copies the base `recv_x` and `recv_topk_weights` cotangents into temporary
   device buffers;
2. adds assignment-gradient contributions into those temporary buffers; and
3. calls the same cached-notify plus DeepEP internode combine path used by the
   normal dispatch backward.

The working May285 default remains `jax`; `ffi` remains available only as the
known-negative May289 diagnostic mode.

Local validation:

```text
uv run python -m py_compile lib/levanter/src/levanter/kernels/deepep/transport_ffi.py
bash -n experiments/grug/moe/run_cw_may_d2560.sh
uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'assignment_gradient or dispatch_internode_backward or combine_internode_with_local_collapse'
```

Result: `5 passed`. The new static contract test verifies that fused mode calls
`levanter_deepep_dispatch_internode` in the forward pass and
`levanter_deepep_dispatch_internode_bwd_fused` in the reverse pass, without
emitting the standalone assignment-gradient or combine targets.

Next check: launch the May290 wrapper:

```text
scratch/launch_may290_deepep_ep16_fusedbwd_n2.sh
```

Expected outcome: first prove the remote CUDA/XLA FFI build succeeds. If it
runs, compare against May285 (`18.671 MFU`, `1.158M tokens/s`, `56.6 ms`) and
inspect the profile for disappearance of the dispatch-backward `scatter-add`.

May290 failed during remote CUDA compilation before W&B/training:

```text
parent=/dlwh/iris-run-job-20260622-002724
child=/dlwh/iris-run-job-20260622-002724/grug-train-MAY290-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-N2-20260622-0027
error=/app/lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu(4319):
  argument of type "void *" is incompatible with parameter of type "const float *"
```

This was a narrow C++ type error in the fused handler's call to
`deep_ep::internode::combine`. Fixed by casting the temporary top-k gradient
buffer to `const float*`. Relaunch as May291 with the same fused-mode config.

### 2026-06-21 17:45 PT: May291 fused dispatch-backward succeeds and removes the target scatter

May291 validated the opt-in fused dispatch-backward mode on 2-node EP16:

```text
parent=/dlwh/iris-run-job-20260622-003250
child=/dlwh/iris-run-job-20260622-003250/grug-train-MAY291-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-N2-20260622-0032
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY291-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-N2-20260622-0032
profile_artifact=marin-community/marin_moe/MAY291-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-N2-20260622-0032-profiler:v0
```

Final W&B summary:

```text
global_step=5
train/loss=11.791556358337402
throughput/mfu=20.580021032712047
throughput/mean_mfu=13.94774042414188
throughput/tokens_per_second=1276380.8082837777
throughput/duration=0.051345178158953786
```

Compared with May285:

```text
May285 fused-pack/JAX-grad: mfu=18.67137165787524, tokens/s=1158005.6410323821, duration=0.05659385211765766
May291 fused-bwd:           mfu=20.580021032712047, tokens/s=1276380.8082837777, duration=0.051345178158953786
delta: +1.91 absolute MFU, +10.2% tokens/s, -9.3% duration
```

Profile outputs:

```text
summary=scratch/profile_summaries/may291_deepep_fusedbwd.json
process0_summary=scratch/profile_summaries/may291_deepep_fusedbwd_process0.json
report=scratch/profile_reports/may291_deepep_fusedbwd.md
process0_report=scratch/profile_reports/may291_deepep_fusedbwd_process0.md
compare=scratch/profile_reports/may285_vs_may291_deepep_fusedbwd_process0_compare.txt
```

Process-0 profile comparison confirms the intended May285 target moved:

```text
May285 input_scatter_fusion_4: 20.893 ms total over 2 profiled occurrences (~10.447 ms each)
May291 input_scatter_fusion_4: absent from hot ops / compare delta -20.893 ms
May291 remaining input_scatter_fusion_2: 0.761 ms total over 2 profiled occurrences
```

The fused path also reduced DeepEP notify time:

```text
May285 notify_dispatch: 5.627 ms total over 2 occurrences
May291 notify_dispatch: 2.818 ms total over 2 occurrences
May285 cached_notify: 1.976 ms total over 6 occurrences
May291 cached_notify: 1.721 ms total over 6 occurrences
May285 combine: 2.661 ms total over 4 occurrences
May291 combine: 2.701 ms total over 4 occurrences
```

Important caveat: the remaining top communication in May291 is no longer the
DeepEP dispatch-backward materialization target. It is mostly all-reduce:

```text
May291 process-0 communication_ops:
  all-reduce: 18.494 ms total over 16 occurrences
  send-recv:   0.573 ms total over 2 occurrences
May291 hot all-reduce rows:
  xent psum:      8.119 ms total over 2 occurrences
  model scatter-add: 6.682 ms total over 2 occurrences
  other bf16 AR:  2.019 ms total over 6 occurrences
  xent f32 psum:  1.323 ms total over 2 occurrences
```

Interpretation: DeepEP multihost is functionally working at 2 nodes / EP16, and
the fused dispatch-backward boundary removed the specific May285
`input_scatter_fusion_4` bottleneck. The next MoE-transport work should target
the remaining DeepEP kernel cost (`notify_dispatch`, `dispatch`, `combine`) and
whether the `scatter-add`-named all-reduce is avoidable/overlappable, but the
largest May291 process-0 family is now xent rather than MoE transport.

### 2026-06-21 18:08 PT: May292 L26 2-node EP16 fused-bwd run succeeds

May292 validated the fused dispatch-backward mode on a model-like L26 2-node
EP16 run. This run used SGD, Pallas GPU xent, FA4 CuTe attention, and
`moe=deepep_internode`.

```text
parent=/dlwh/iris-run-job-20260622-004616
child=/dlwh/iris-run-job-20260622-004616/grug-train-MAY292-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-L26-N2-20260622-0046
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY292-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-L26-N2-20260622-0046
profile_artifact=marin-community/marin_moe/MAY292-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-L26-N2-20260622-0046-profiler:v0
s3_profile=s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-profiles/MAY292-DEEPEP-EP16-TK4-FUSEDPACK-FUSEDBWD-L26-N2-20260622-0046-profiler.tgz
```

Final W&B summary:

```text
global_step=5
train/loss=11.791276931762695
throughput/mfu=13.930986447798679
throughput/mean_mfu=9.314407836823335
throughput/tokens_per_second=128828.07387980678
throughput/duration=0.5087089950684458
```

Steady-state steps 2-5 averaged:

```text
mfu=13.957340323889223
tokens/s=129071.7837641523
duration=0.5077709685428999
```

Profile outputs:

```text
summary=scratch/profile_summaries/may292_deepep_fusedbwd_l26.json
process0_summary=scratch/profile_summaries/may292_deepep_fusedbwd_l26_process0.json
report=scratch/profile_reports/may292_deepep_fusedbwd_l26.md
process0_report=scratch/profile_reports/may292_deepep_fusedbwd_l26_process0.md
compare=scratch/profile_reports/may291_vs_may292_deepep_fusedbwd_process0_compare.txt
```

Process-0 xprof kernel table breakdown:

```text
compute:       639.750 ms (87.66%)
communication: 90.064 ms (12.34%)

semantic families:
  moe:             400.976 ms (38.53%)
  attention_flash:  78.404 ms ( 7.53%)
  loss_xent:        50.209 ms ( 4.82%)
  attention_dense:  50.114 ms ( 4.82%)
  collective:       49.721 ms ( 4.78%)
  norm_gating:      35.056 ms ( 3.37%)
  optimizer_apply:  25.309 ms ( 2.43%)
  dense_mlp:        24.631 ms ( 2.37%)

communication_ops:
  all-reduce: 75.137 ms total over 166 occurrences
  send-recv:  14.926 ms total over 52 occurrences
```

Top process-0 kernels relevant to MoE/DeepEP:

```text
deepep combine, dispatch-bwd side: 35.086 ms over 52 occurrences
deepep combine, forward side:      34.595 ms over 52 occurrences
deepep dispatch, forward side:     28.766 ms over 52 occurrences
deepep dispatch, bwd side:         28.557 ms over 52 occurrences
deepep cached_notify, bwd side:    23.826 ms over 52 occurrences
deepep cached_notify, other side:  19.965 ms over 52 occurrences
notify_dispatch:                   15.113 ms over 52 occurrences
residual JAX local collapse:
  input_scatter_fusion_52:         19.793 ms over 52 occurrences
  tf_op_path=.../moe_ep_deepep_internode/combine/deepep_collapse_local_assignments_jax/scatter-add
```

The one-layer May291 result established that the fused dispatch-backward path
removed the earlier May285 `input_scatter_fusion_4` bottleneck. May292 shows the
same fused path holds at L26, but the model-like run is still only about 14 MFU.
The remaining MoE-specific work is no longer one giant JAX scatter; it is the
sum of per-layer DeepEP internode dispatch/combine/notify plus the smaller
residual JAX local-collapse scatter in the combine path. The next concrete
target should be to remove or fuse
`deepep_collapse_local_assignments_jax/scatter-add` into the combine boundary,
then reprofile to see whether DeepEP kernels or all-reduce become the dominant
remaining MoE-side cost.

### 2026-06-21 17:12 PT: May289 dispatch-VJP assignment-gradient FFI probe

The latest May288 fused-combine profile showed that fusing local-collapse into
the combine call was not the useful boundary: the remaining large XLA op was a
dispatch-backward `scatter-add` under:

```text
jit(train_step)/forward_backward/transpose(forward_backward)/jvp(Transformer)/Block/Block._mlp_update/MoEMLP/moe_experts/MoEExpertMlp/moe_mlp/shard_map/moe_ep_deepep_internode/dispatch/deepep_dispatch_internode_transport/scatter-add
```

This points at `_dispatch_internode_with_vjp_bwd`, which still used
`_assignment_gradients_jax(...)` for the cotangents of `x_dispatch` and
`assignment_weights`. The intranode assignment-aware path already uses the FFI
assignment-gradient helper, so this experiment adds an opt-in switch:

```text
LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE=jax  # default
LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE=ffi  # diagnostic
```

Local validation before launch:

```text
uv run python -m py_compile \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py
bash -n experiments/grug/moe/run_cw_may_d2560.sh
uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q \
  -k 'assignment_gradient or dispatch_internode_backward or combine_internode_with_local_collapse'
```

Result: `4 passed`. The new contract test differentiates through
`dispatch.x_dispatch` and `dispatch.assignment_weights` and verifies that the
opt-in mode calls `levanter_deepep_assignment_gradients` before the cached
combine backward.

Launched May289:

```text
parent=/dlwh/iris-run-job-20260622-001249
child=/dlwh/iris-run-job-20260622-001249/grug-train-MAY289-DEEPEP-EP16-TK4-FUSEDPACK-ASSIGNGRADFFI-N2-20260622-0012
run_id=MAY289-DEEPEP-EP16-TK4-FUSEDPACK-ASSIGNGRADFFI-N2-20260622-0012
script=scratch/launch_may289_deepep_ep16_assignmentgradffi_n2.sh
baseline=MAY285-DEEPEP-EP16-TK4-FUSEDPACK-JAXGRAD-N2-20260621-2309
only_intended_delta=LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE=ffi
```

Initial bounded Iris check showed parent and child `JOB_STATE_RUNNING`, with two
child tasks running. Logs showed DeepEP source clone and FFI compilation on
both tasks, with no early `Traceback`, OOM, invalid-resource, or rendezvous
failure in the sampled tail.

May289 then failed before first train metrics:

```text
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY289-DEEPEP-EP16-TK4-FUSEDPACK-ASSIGNGRADFFI-N2-20260622-0012
wandb_state=running at check time, but no train history rows and no profile artifact
error=jax.errors.JaxRuntimeError: INTERNAL: AssignmentGradRecvXKernel: invalid resource handle
failure_point=after "Finished Grug train_step dispatch for step 0; waiting for train/loss"
```

Interpretation: the existing standalone assignment-gradient FFI has the same
invalid-resource shape as the previous local assignment pack/collapse helpers.
It is not a usable drop-in fix for the May288 dispatch-backward scatter-add.
The next implementation target should avoid this standalone helper shape and
instead fuse the dispatch-backward assignment-gradient work with the cached
combine boundary, or otherwise implement the backward transport as one
internode-aware FFI operation.

### 2026-06-21 17:05 PT: May288 fused local-collapse combine is functional but negative

May288 tested an opt-in fused local-collapse combine path:

```text
parent=/dlwh/iris-run-job-20260621-235717
run_id=MAY288-DEEPEP-EP16-TK4-FUSEDLOCALCOMBINE-N2-20260621-2357
env=LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE=fused_combine
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY288-DEEPEP-EP16-TK4-FUSEDLOCALCOMBINE-N2-20260621-2357
```

The run succeeded. This proves the new
`levanter_deepep_combine_internode_with_local_collapse` FFI target builds and
can execute in the two-node EP16 path. It did not improve throughput:

```text
May285 baseline: mfu=18.67137165787524, tokens/s=1158005.6410323821, duration=0.05659385211765766
May287 gather:   mfu=16.466202101620965, tokens/s=1021240.1782497771, duration=0.06417295499704778
May288 fused:    mfu=17.65475417053485, tokens/s=1094954.634031737, duration=0.059852708014659584
```

Profile outputs:

```text
summary=scratch/profile_summaries/may288_deepep_fusedlocalcombine.json
report=scratch/profile_reports/may288_deepep_fusedlocalcombine.md
artifact=marin-community/marin_moe/MAY288-DEEPEP-EP16-TK4-FUSEDLOCALCOMBINE-N2-20260621-2357-profiler:v0
remote=s3://marin-na/tmp/ttl=7d/experiments/grug-moe-cw/deepep-profiles/MAY288-DEEPEP-EP16-TK4-FUSEDLOCALCOMBINE-N2-20260621-2357-profiler.tgz
```

Process-0 profile comparison:

```text
May285 input_scatter_fusion_4: 20.893 ms total over 2 profiled occurrences (~10.447 ms each)
May287 input_scatter_fusion_1: 21.875 ms total over 2 profiled occurrences (~10.938 ms each)
May287 input_scatter_fusion_5: 21.743 ms total over 2 profiled occurrences (~10.872 ms each)
May288 input_scatter_fusion_3: 21.778 ms total over 2 profiled occurrences (~10.889 ms each)
May288 CollapseLocalAssignmentsKernel: 0.535 ms total over 2 profiled occurrences
May288 notify_dispatch: 24.185 ms total over 2 profiled occurrences
```

Interpretation: the fused-combine path did execute and the local collapse kernel
itself is small, but the large XLA `input_scatter_fusion_*` remains at the same
per-step scale as May285. The new path also made DeepEP `notify_dispatch`
substantially slower than May285 (~24.2 ms total vs ~5.6 ms total in the same
profile window). This is not a useful default. Keep it as an opt-in diagnostic
only, unless future work explains and removes the remaining scatter and notify
regression.

Next likely target: the remaining large `input_scatter_fusion_*` is probably not
the local-collapse combine boundary anymore. It is more likely in dispatch-side
assignment/materialization or the routing/packing path before the expert GEMM.
The next experiment should locate which HLO/fusion owns `input_scatter_fusion_3`
using the HLO proto/xprof tables or XLA dump, then move/fuse that specific
assignment/materialization boundary rather than adding more work to combine.

### 2026-06-21 16:47 PT: May287 gather-collapse is correct but slower

Added an opt-in JAX gather-collapse mode for the internode local assignment
collapse:

```text
env=LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE=gather
parent=/dlwh/iris-run-job-20260621-233750
child=/dlwh/iris-run-job-20260621-233750/grug-train-MAY287-DEEPEP-EP16-TK4-FUSEDPACK-GATHERCOLLAPSE-N2-20260621-2337
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY287-DEEPEP-EP16-TK4-FUSEDPACK-GATHERCOLLAPSE-N2-20260621-2337
profile=scratch/profile_summaries/may287_deepep_gathercollapse_process0.json
```

Local validation before launch:

```text
uv run python -m py_compile lib/levanter/src/levanter/grug/_moe/ep_deepep.py
bash -n experiments/grug/moe/run_cw_may_d2560.sh
uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'collapse_local_assignments or deepep_internode or tokens_per_rdma_rank'
```

The focused tests passed, including primal and gradient equivalence between the
scatter and gather collapse helpers.

May287 succeeded but regressed throughput:

```text
May285 baseline step 5: MFU=18.67137165787524, tokens/s=1158005.6410323821, duration=0.05659385211765766
May287 gather step 5:  MFU=16.466202101620965, tokens/s=1021240.1782497771, duration=0.06417295499704778
```

Process-0 profile comparison shows why this is not the right direction:

```text
May285: input_scatter_fusion_4 = 20.893 ms total, 2 calls
May287: input_scatter_fusion_1 = 21.875 ms total, 2 calls
May287: input_scatter_fusion_5 = 21.743 ms total, 2 calls
```

The gather formulation avoided the exact May285 scatter name but introduced two
scatter-like fusions, likely one in the forward/collapse path and one in the
reverse-mode path. DeepEP kernel times were otherwise similar or slightly
better. Keep the gather mode as an opt-in diagnostic, but leave the default on
the May285 scatter path. The next useful optimization is still a fused
collapse/combine boundary or a process-per-GPU-safe lower-level local assignment
bridge, not a pure JAX gather rewrite.

### 2026-06-21 16:20 PT: May285 succeeded; dispatch-pack and JAX assignment-gradient improve DeepEP

May285 was the direct May284 retry after keeping fused internode dispatch-pack
but replacing the assignment-gradient FFI in internode dispatch backward with a
JAX segment-sum implementation:

```text
parent=/dlwh/iris-run-job-20260621-230952
run_id=MAY285-DEEPEP-EP16-TK4-FUSEDPACK-JAXGRAD-N2-20260621-2309
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY285-DEEPEP-EP16-TK4-FUSEDPACK-JAXGRAD-N2-20260621-2309
```

Observed result:

- Iris parent and child reached `JOB_STATE_SUCCEEDED`;
- W&B finished successfully;
- the profile artifact uploaded as
  `MAY285-DEEPEP-EP16-TK4-FUSEDPACK-JAXGRAD-N2-20260621-2309-profiler:v0`.

Final W&B summary:

```text
global_step=5
train/loss=11.791558265686035
throughput/duration=0.05659385211765766
throughput/tokens_per_second=1158005.6410323821
throughput/mfu=18.67137165787524
throughput/mean_mfu=11.231206167919053
```

Compared to May272:

```text
May272 DeepEP baseline: duration=0.07006779499351978, tokens/s=935322.7114691007, mfu=15.080891963809387
May285 fused-pack/JAX-grad: duration=0.05659385211765766, tokens/s=1158005.6410323821, mfu=18.67137165787524
```

This is roughly a 24% throughput/MFU improvement and clears the May284
`AssignmentGradRecvXKernel: invalid resource handle` boundary.

Profile artifacts and reports:

```text
scratch/profile_summaries/may285_deepep_fusedpack_jaxgrad.json
scratch/profile_summaries/may285_deepep_fusedpack_jaxgrad_process0.json
scratch/profile_reports/may285_deepep_fusedpack_jaxgrad.md
scratch/profile_reports/may285_deepep_fusedpack_jaxgrad_process0.md
scratch/profile_reports/may272_vs_may285_deepep_process0_compare.json
```

The all-process May285 profile has eight local worker traces and should not be
directly compared to the older May272 process-0 summary. The process-0
comparison is the useful apples-to-apples readout.

May285 process-0 profile versus May272 process-0:

- the old `input_reduce_fusion_1` disappeared (`~21.1 ms` in May272);
- the old `input_scatter_fusion_8`, `input_scatter_fusion_5`, and
  `input_scatter_fusion_7` disappeared;
- the remaining largest local staging kernel is
  `input_scatter_fusion_4`, `~20.9 ms` across the captured profile window;
- DeepEP internode kernels are now smaller than that remaining scatter:
  - `notify_dispatch`: `~5.6 ms`;
  - forward/cached dispatch: `~1.2 ms` and `~1.0 ms`;
  - combine: `~2.7 ms`;
  - cached notify: `~2.0 ms`;
- process-0 all-reduce time is still `~23.5 ms` across the profile window.

Interpretation:

- The fused dispatch-pack and JAX assignment-gradient path is real progress and
  should be kept as the current working baseline.
- The next DeepEP-specific bottleneck is no longer the dispatch pack or the
  assignment-gradient FFI; it is the combine-side local assignment collapse /
  materialization, visible as the remaining `input_scatter_fusion_4`.
- DeepEP still trails the best ring baseline at this small one-layer shape, but
  the gap is narrower than May272.

### 2026-06-21 16:30 PT: May286 proves standalone CUDA collapse is still not usable here

May286 attempted the obvious next patch: keep May285's fused internode
dispatch-pack, but replace the remaining JAX local assignment collapse with
the existing CUDA `deepep_collapse_local_assignments` helper:

```text
parent=/dlwh/iris-run-job-20260621-232450
run_id=MAY286-DEEPEP-EP16-TK4-FUSEDPACK-FFICOLLAPSE-N2-20260621-2324
```

Local checks before launch passed:

```text
uv run python -m py_compile lib/levanter/src/levanter/grug/_moe/ep_deepep.py lib/levanter/src/levanter/kernels/deepep/transport_ffi.py
uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py -q -k 'internode and (dispatch or combine)'
uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'deepep_internode or tokens_per_rdma_rank'
```

Runtime result:

```text
jax.errors.JaxRuntimeError: INTERNAL: CollapseLocalAssignmentsKernel: invalid resource handle
```

The failed May286 parent and child were stopped to free the two H100 nodes. The
experiment patch was reverted locally, so the worktree is back to the May285
JAX-collapse baseline.

Interpretation:

- The local-assignment CUDA helper failure is not limited to the count-seeded
  pack path. The standalone collapse helper also cannot be dropped into the
  process-per-GPU internode Grug graph as-is.
- The next optimization should not be "call the existing collapse FFI." It
  should either fuse collapse semantics directly into
  `deepep_combine_internode`, or add a new process-per-GPU-safe lower-level
  collapse/combine bridge with its own stream/resource contract.

### 2026-06-21 15:38 PT: May280 killed before diagnostic and fused internode assignment patch

May280 did not produce a useful diagnostic:

```text
parent=/dlwh/iris-run-job-20260621-222520
run_id=MAY280-DEEPEP-EP16-TK4-FFIPACK5-CLEARERR-N2-20260621-2225
state=killed
wandb_state=failed
metrics=none
```

The parent was killed after about 6.5 minutes, and W&B recorded no train
metrics. The bounded tail did not show `DEEPEP_CUDA_LAST_ERROR_CLEARED` or
`PrefixLocalAssignment...`, so this run does not answer whether the previous
invalid-resource error was stale CUDA state or the local-assignment launch
itself.

The next change removes the failing boundary instead of relaunching the same
shape:

- extend `deepep_dispatch_internode` to emit capacity-capped local group sizes
  and packed expert assignment buffers (`x_dispatch`, assignment weights,
  token indices, destinations);
- perform the count-seeded local assignment pack inside the internode dispatch
  FFI, after `deep_ep::internode::dispatch` and before returning to XLA;
- route internode MoE directly through these dispatch-owned packed buffers, so
  `ep_deepep.py` no longer calls the separate
  `deepep_pack_local_assignments_from_counts` FFI for the internode path;
- add the assignment-gradient contribution to the internode dispatch custom
  VJP before the normal DeepEP combine gradient.

Local validation:

```text
uv run python -m py_compile \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py

uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py \
  -k 'dispatch_internode_exposes_static_jax_contract or dispatch_internode_backward_uses_combine_contract'
```

Result: both selected tests passed locally. The remaining risk is C++ compile
or runtime behavior on the CoreWeave CUDA image.

May281 result:

```text
parent=/dlwh/iris-run-job-20260621-223955
run_id=MAY281-DEEPEP-EP16-TK4-FUSEDPACK-N2-20260621-2239
state=failed
failure=remote nvcc compile error
```

The CUDA compile failed on both tasks:

```text
deepep_transport_ffi.cu(3672): error: no instance of overloaded function "std::max" matches the argument list
```

This was from clamping `runtime.moe_recv_expert_counter[expert]` directly with
`std::max(..., 0)` while the DeepEP counter type is not an exact `int`. The
fix casts the counter to `int` first and uses an explicit negative clamp before
the existing `std::min(raw_count, remaining_assignments)`.

May282 result:

```text
parent=/dlwh/iris-run-job-20260621-224434
run_id=MAY282-DEEPEP-EP16-TK4-FUSEDPACK2-N2-20260621-2244
state=stopped after worker fatal
failure=INVALID_ARGUMENT: DeepEP local assignment collapse destination map shape is invalid
```

This run is useful: it cleared the May281 remote CUDA compile failure and also
got past the earlier `PrefixLocalAssignmentCursorsKernel(counts)` invalid
resource failure. The new failure was in `deepep_collapse_local_assignments`.

Root cause: `_dispatch_internode_impl` returned `results[:21]` into
`DeepEPInternodeDispatch`. The FFI result tuple includes internal scratch
buffers:

```text
20 = local_group_cursors
21 = recv_assignment_indices
22 = assignment_destinations
```

So Python exposed the length-`local_experts` cursor buffer as
`assignment_destinations`; collapse correctly rejected it because the
destination map must be divisible by `recv_capacity`. The fix projects
`DeepEPInternodeDispatch(*results[:20], results[22])` and adds static-shape test
coverage for `assignment_destinations.shape == (max_recv_tokens * topk,)`.

May283 result:

```text
parent=/dlwh/iris-run-job-20260621-225244
run_id=MAY283-DEEPEP-EP16-TK4-FUSEDPACK3-N2-20260621-2252
state=stopped after worker fatal
failure=INTERNAL: CollapseLocalAssignmentsKernel: invalid resource handle
```

This got farther again: remote CUDA compile passed, W&B appeared, train step 0
compiled/dispatched, and the fused dispatch-owned pack path got past the
previous shape error. The next failing boundary is the separate
`deepep_collapse_local_assignments` FFI. This mirrors the earlier standalone
pack failure: local assignment kernels launched as a separate FFI after
internode DeepEP transport can hit invalid CUDA resources.

Temporary fallback for May284: keep fused internode dispatch-pack but switch
only collapse back to the existing JAX segment-add helper. This should test
whether removing the JAX pack/scatter side alone is enough to move throughput,
while the final optimized direction remains fusing collapse into internode
combine or replacing that boundary with a lower-level bridge.

May284 result:

```text
parent=/dlwh/iris-run-job-20260621-230048
run_id=MAY284-DEEPEP-EP16-TK4-FUSEDPACK-JAXCOLLAPSE-N2-20260621-2300
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY284-DEEPEP-EP16-TK4-FUSEDPACK-JAXCOLLAPSE-N2-20260621-2300
state=failed before first metrics
failure=INTERNAL: AssignmentGradRecvXKernel: invalid resource handle
```

This run proved that fused dispatch-owned local assignment pack plus JAX
collapse gets past the previous standalone collapse FFI failure. The next
remaining failing boundary was the standalone assignment-gradient FFI called
from the internode dispatch custom VJP during backward.

Patch for May285:

- added a pure-JAX `_assignment_gradients_jax(...)` helper that uses
  `assignment_destinations` to gather packed assignment cotangents back into
  `recv_x` and `recv_topk_weights`;
- changed only the internode dispatch VJP to use this JAX helper;
- left the older intranode/standalone local assignment VJPs on the existing FFI
  path for now.

Local validation:

```text
uv run python -m py_compile \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py

uv run pytest lib/levanter/tests/kernels/test_deepep_availability.py \
  -k 'dispatch_internode_exposes_static_jax_contract or dispatch_internode_backward_uses_combine_contract'
```

Result: both selected tests passed locally. May285 was submitted as the direct
retry:

```text
parent=/dlwh/iris-run-job-20260621-230952
run_id_prefix=MAY285-DEEPEP-EP16-TK4-FUSEDPACK-JAXGRAD-N2-20260621-2309
launcher=scratch/launch_may285_deepep_ep16_fusedpack_jaxgrad_n2.sh
```

### 2026-06-21 15:22 PT: May279 aux-stream patch did not move the failure

May279 built the DeepEP layout/intranode/internode FFI libraries, started the
process-per-GPU supervisors on both tasks, created W&B, initialized the EP16
mesh, and entered the first train step:

```text
parent=/dlwh/iris-run-job-20260621-221757
child=/dlwh/iris-run-job-20260621-221757/grug-train-MAY279-DEEPEP-EP16-TK4-FFIPACK4-AUXSTREAM-N2-20260621-2217
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY279-DEEPEP-EP16-TK4-FFIPACK4-AUXSTREAM-N2-20260621-2217
mesh={'replica_dcn': 1, 'data': 1, 'expert': 16, 'model': 1}
```

It still failed before first metrics:

```text
jax.block_until_ready(metrics["train/loss"])
jax.errors.JaxRuntimeError: INTERNAL: PrefixLocalAssignmentCursorsKernel(counts): invalid resource handle
```

The error did not move to any of the explicit aux-stream synchronization
contexts added around local assignment pack/collapse/gradients. The failed
parent and child were stopped after W&B marked the run failed. The aux-stream
patch is therefore not sufficient; either the standalone local-assignment FFI
is still using an invalid launch resource, or the error being reported by the
first local-assignment kernel is stale CUDA last-error state from an earlier
DeepEP call.

Next patch:

- add `ClearCudaLastError(...)` and call it at local assignment pack/collapse
  and assignment-gradient helper boundaries before those helpers launch their
  own kernels;
- when `LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG=1`, log any cleared stale CUDA
  error as `DEEPEP_CUDA_LAST_ERROR_CLEARED`;
- relaunch one EP16 two-node diagnostic run. If the cleared error is from
  internode dispatch, the log should now show the stale source explicitly; if
  the local assignment kernel itself is bad, the run should continue failing
  after the clear.

May278 result:

```text
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY278-DEEPEP-EP16-TK4-FFIPACK3-SYNCDBG-N2-20260621-2212
state=failed
error=jax.errors.JaxRuntimeError: INTERNAL: PrefixLocalAssignmentCursorsKernel(counts): invalid resource handle
```

The host-dispatch synchronization did not move the error into
`deepep_dispatch_internode`. The failure still lands at the count-seeded local
assignment prefix kernel. That makes a stale async DeepEP dispatch error less
likely; the local assignment FFI launch context itself is the current target.
The failed parent was stopped.

Relaunch with the aux-stream local assignment patch:

```text
parent=/dlwh/iris-run-job-20260621-221757
run_id=MAY279-DEEPEP-EP16-TK4-FFIPACK4-AUXSTREAM-N2-20260621-2217
```

Expected outcome: if the XLA platform stream was the invalid resource for the
standalone local assignment FFI, May279 should pass step 0. If not, the error
context should move to one of the explicit `cudaStreamSynchronize(before/after
internode local assignment ...)` boundaries or remain on the prefix kernel.

Relaunch:

```text
parent=/dlwh/iris-run-job-20260621-220414
run_id=MAY277-DEEPEP-EP16-TK4-FFIPACK3-PROFREM-N2-20260621-2204
state=scratch/20260621-1504_may277_deepep_ffipack_deviceguard_state.json
```

At first check, the parent was running and W&B had not appeared yet.

### 2026-06-21 15:09 PT: May277 still fails in count-seeded local pack

May277 reached W&B and started the first train step on both nodes, but failed
before first metrics:

```text
parent=/dlwh/iris-run-job-20260621-220414
child=/dlwh/iris-run-job-20260621-220414/grug-train-MAY277-DEEPEP-EP16-TK4-FFIPACK3-PROFREM-N2-20260621-2204
wandb=https://wandb.ai/marin-community/marin_moe/runs/MAY277-DEEPEP-EP16-TK4-FFIPACK3-PROFREM-N2-20260621-2204
error=jax.errors.JaxRuntimeError: INTERNAL: PrefixLocalAssignmentCursorsKernel(counts): invalid resource handle
```

This reproduces the May276 failure even after wrapping the runtime-free local
assignment pack/collapse/gradient helpers in a pointer-derived CUDA device
guard. That means the problem is not just stale process current device. The
standalone local assignment FFI path is still launching `cudaLaunchKernel` with
an invalid resource under XLA's FFI stream context.

The failed parent and child were stopped to avoid leaving the run consuming
capacity. The next fix should avoid reusing this FFI helper shape as-is for the
internode path, or add more precise stream/resource diagnostics around
`PrefixLocalAssignmentCursorsKernel(counts)`.

### 2026-06-21 15:12 PT: May278 sync-debug launch and aux-stream patch

Launched a diagnostic rerun with host dispatch synchronization enabled:

```text
parent=/dlwh/iris-run-job-20260621-221207
run_id=MAY278-DEEPEP-EP16-TK4-FFIPACK3-SYNCDBG-N2-20260621-2212
extra_env=LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG=1
```

Purpose: determine whether the `invalid resource handle` is actually an async
error from `deepep_dispatch_internode` that only surfaces in the next local
assignment FFI call. With host dispatch debug enabled, the internode dispatch
handler synchronizes after the DeepEP notify/dispatch launches.

Prepared the next patch while May278 runs:

- added `InternodeRuntimeManager::RuntimeForCurrentDeviceIfInitialized()`;
- when the internode runtime is active, local assignment pack/collapse and
  assignment-gradient helpers now synchronize the XLA stream, launch on the
  internode runtime `aux_stream`, then synchronize that stream before returning;
- this is deliberately a correctness/diagnostic patch, not the final
  low-overhead boundary. If it works, replace the syncs with events or fuse the
  local assignment pack into the internode dispatch FFI.

Local validation:

```text
uv run python -m py_compile \
  experiments/grug/dispatch.py \
  experiments/grug/moe/train.py \
  lib/levanter/src/levanter/kernels/deepep/transport_ffi.py \
  lib/levanter/src/levanter/grug/_moe/ep_deepep.py
```
