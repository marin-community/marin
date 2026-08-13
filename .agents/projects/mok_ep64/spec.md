# MoK EP64 Contract

## Public configuration

```python
class MokLikeTopology(enum.StrEnum):
    LOCAL_EP4 = "local_ep4"
    NVLINK_EP64 = "nvlink_ep64"


@dataclasses.dataclass(frozen=True)
class MokLikeSymmetricWorkspaceConfig:
    topology: MokLikeTopology
    workspace_slots: int
    forward_x_storage: MokLikeForwardXStorage
    backward_peer_storage: MokLikeBackwardPeerStorage
    schedule_capacity_factor: float
```

`NVLINK_EP64` requires expert-axis size 64, exactly one local JAX device per process, exactly 64
JAX processes in the expert group, one workspace slot, and runtime-staged forward/backward
storage. It rejects direct XLA peer storage. `schedule_capacity_factor=64` is the strict all-to-one
capacity contract; smaller values are capacity-limited.

## Symmetric workspace owner

File: `lib/levanter/src/levanter/kernels/mixture_of_kittens/symmetric_memory.py`

```python
@dataclasses.dataclass(frozen=True)
class MokLikeSymmetricArenaLayout:
    total_bytes: int
    offsets: Mapping[str, int]
    sizes: Mapping[str, int]


@dataclasses.dataclass(eq=False)
class MokLikeSymmetricWorkspace:
    rank: int
    world_size: int
    local_pointer: int
    peer_pointers: tuple[int, ...]
    layout: MokLikeSymmetricArenaLayout

    def close(self) -> None:
        """Collectively release the workspace after all native calls are quiescent."""


def initialize_mok_like_symmetric_workspace(
    *,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    workspace_slots: int,
    timeout: float = 300.0,
) -> MokLikeSymmetricWorkspace:
    """Create one identical symmetric arena per expert rank and expose every peer alias.

    JAX distributed initialization must already be complete. Every expert rank calls this function
    once, in identical order and with identical shape arguments. The function raises before native
    initialization if world/rank/device ordering differs from the EP64 contract.
    """
```

The workspace holds strong references to the Torch allocation, rendezvous handle, and process
group. `peer_pointers[i]` addresses rank `i`'s arena from the caller's CUDA context. Pointer values
need not be symmetric across processes. Arena offsets and sizes are 256-byte aligned and identical
on every rank.

## Native initialization

```c
int levanter_mok_init_runtime_ep64(
    int ep_rank,
    int ep_size,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int workspace_slots,
    const uint64_t* peer_arena_pointers,
    int64_t peer_pointer_count,
    const uint64_t* arena_offsets,
    int64_t arena_offset_count);
```

`ep_size` must be 64, `peer_pointer_count` must equal 64, and `workspace_slots` must equal one for
the first release. The function validates arena schema/version and selects the EP64 BF16 kernel.
EP4 retains its existing target and ABI until all call sites migrate.

FFI targets are distinct:

- `levanter_mok_forward_bf16_64`
- `levanter_mok_backward_bf16_64`
- `levanter_mok_failure_fence`

The native build schema and cache key include EP size and workspace schema version.

## Probe

File: `experiments/grug/moe_hero_ep/mok_ep64_symmetric_memory_probe.py`

```python
def run_symmetric_memory_probe(*, arena_bytes: int, iterations: int) -> None:
    """Verify JAX/Torch coexistence, every peer mapping, repeated reuse, and clean teardown."""
```

The probe emits one JSON record per process plus a process-zero summary. Acceptance requires exact
world/rank/device identity, every peer pointer nonzero, exact values from all peers (including ranks
31, 32, and 63 when present), no timeout, and exit zero from every supervised process.

## Errors

- `MokLikeTopologyError`: JAX process/device/expert-axis order does not match the runtime group.
- `MokLikeSymmetricMemoryUnavailable`: PyTorch symmetric memory cannot create or rendezvous the
  requested NVLink-domain workspace.
- `MokLikeOperationStampMismatch`: a peer published a different operation stamp for the active
  workspace slot; native code cancels the operation before surfacing this error.
- Existing synchronous native failures propagate through the uniform typed failure fence. Sticky
  CUDA context errors abort the process.

## Launcher identity

File: `experiments/grug/moe_hero_ep/launch_mok_ep64.py`

The launcher contract is 16 tasks, four GB200s/task, four processes/task, E128/top4/d6144/i3072,
48 layers, batch1024, expert64/DP1, one slot, symmetric runtime staging, zero retry/failure budgets,
and profile steps 80-84 for a 100-step seal. W&B tags include `mok-ep64`, `ep-64`, `dp-1`,
`jax-processes-64`, `processes-per-task-4`, `local-experts-2`, the capacity policy, workspace
schema, and source commit.

## Out of scope

- Cross-process zero-copy of XLA-owned input/output buffers.
- More than one workspace slot at EP64.
- EP sizes other than 4 and 64 in the public training launcher.
- Replacing JAX route all-gather with CUDA multicast.
- Recovering in-process from asynchronous illegal-address or poisoned-context failures.
