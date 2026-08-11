# Peer-visible JAX FFI buffer contract

## Public JAX API

```python
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

FfiMemorySpace = Literal["default", "collective"]

@dataclass(frozen=True)
class FfiPeerAccess:
    instance_id: int
    participant_groups: tuple[tuple[int, ...], ...]
    operand_indices: tuple[int, ...]
    result_indices: tuple[int, ...]

def ffi_call(
    target_name: str,
    result_shape_dtypes,
    *,
    operand_memory_spaces: Sequence[FfiMemorySpace | None] | None = None,
    result_memory_spaces: Sequence[FfiMemorySpace | None] | None = None,
    peer_access: FfiPeerAccess | None = None,
    **existing_options,
): ...

def ffi_lowering(
    call_target_name: str,
    *,
    operand_memory_spaces: Sequence[FfiMemorySpace | None] | None = None,
    result_memory_spaces: Sequence[FfiMemorySpace | None] | None = None,
    peer_access: FfiPeerAccess | None = None,
    **existing_options,
): ...
```

Memory-space sequences correspond to JAX's flattened array operand/result
order; tokens and keyword attributes do not consume indices. A missing sequence
or a `None` entry means normal buffer assignment. An explicit `"default"`
requires memory space zero, while `"collective"` maps to GPU memory space one.
Length mismatch, unknown values, or an unsupported backend is a compile-time
error.

`FfiPeerAccess` identifies one static custom-call site. Participant groups are
disjoint tuples of logical device IDs within one PJRT client and process. Every
executing device belongs to exactly one group and every group member invokes
the site once per dynamic instance. Operand/result indices use the same
flattening order as the memory-space sequences. The coordinator identity is
`(client_epoch, RunId, instance_id, participant_group)`.

Every referenced buffer remains in HLO memory space zero and must be directly
read/write accessible from each device in its participant group. A pointer
published for one owner allocation is dereferenceable unchanged from all group
devices under CUDA UVA/VMM; separate rank-owned allocations need not have the
same pointer value. Operand contents remain semantically read-only even when
the VMM mapping grants hardware write permission.

Combining `peer_access` indices with a non-default memory space is invalid in
the first API version. Callers choose either ordinary peer-visible device
memory or an explicit collective memory space.

## PJRT memory kind and external buffers

The GPU PJRT client advertises a `peer_visible` memory kind. Existing JAX
sharding APIs select it through `sharding.with_memory_kind("peer_visible")` for
`device_put`, jitted parameters, donation, and constrained outputs.

External parameters and donated buffers referenced by `FfiPeerAccess` must
already carry this memory kind for the exact participant group. Executable load
or execution validates the capability before enqueue. Donation cannot upgrade
an ordinary allocation, and the implementation must not insert a capability
copy.

Peer visibility propagates through the complete HLO alias set and reused
`BufferAllocation`. If any aliased value requires it, the allocation requires
it for its whole lifetime. An alias between capable and incapable external
buffers is rejected; compiler-owned aliases may be unioned onto one capable
allocation.

## StableHLO/OpenXLA attributes

JAX lowers memory-space requests using the existing string frontend attributes:

```text
operands_memory_spaces = "{operand_index:memory_space,...}"
results_memory_spaces = "{result_index:memory_space,...}"
```

The proposed peer-access descriptor lowers as checked frontend attributes:

```text
peer_access_instance_id = "<non-negative integer>"
peer_access_participant_groups = "{{device,...},{device,...}}"
peer_visible_operands = "{operand_index,...}"
peer_visible_results = "{result_index,...}"
```

Indices are zero-based flattened array positions. Duplicate, negative, or
out-of-range indices are invalid. Participant groups must be nonempty, disjoint,
and contain the executing device exactly once. A non-GPU backend rejects the
descriptor unless it defines an equivalent capability.

## GPU allocation and lifetime

A peer-visible allocation has read/write access enabled for its group before
its first peer access. External buffers are validated before enqueue;
compiler-owned allocations are mapped when allocated.

For each remotely read operand, the owner producer completes before owner-ready
publication; every accessor waits for readiness before reading; every accessor
publishes done; and the owner stream waits for all accessors before reuse. For a
remotely written result, pointer publication precedes remote writes and the
owner stream waits for every writer before local consumption.

The handler may return after these closing waits are enqueued on its local XLA
stream. The call is active until those waits complete. Afterward, remote access
is forbidden and OpenXLA may reuse the allocation immediately. This does not
require the numerical pointer value to become invalid.

If a participant fails after remote work begins, all ranks either prove remote
quiescence before returning or poison the executable/client. A healthy rank may
not return normally while failed-peer work could still access its buffers.

## Errors

- `INVALID_ARGUMENT`: malformed sequence length, memory-space value,
  peer-access descriptor, frontend-attribute index, or alias declaration.
- `UNIMPLEMENTED`: backend does not implement the requested memory space or
  peer-visible capability.
- `FAILED_PRECONDITION`: topology, external-buffer memory kind, participant
  group, alias set, or donation cannot satisfy the capability. This may surface
  at compilation, executable load, or execution, depending on when the device
  assignment and allocation provenance become known.
- `RESOURCE_EXHAUSTED`: the peer-visible allocator cannot reserve or map the
  requested bytes.
- Runtime FFI errors remain responsible for missing-rank rendezvous, remote
  completion timeout, and pointer-exchange failure.

## Marin integration boundary

The public Levanter backend remains `mok_like` with staged runtime storage. A
future experimental storage selector may add `storage="xla_peer_visible"` only
after the upstream API exists. It must preserve the normal Grug model boundary,
canonical parameter leaves, custom-VJP residual, and all current numerical
contracts.

## Out of scope

- Removing or weakening the staged `mok_like` fallback.
- Treating successful P2P access to undocumented default-space buffers as a
  supported contract.
- Equal rank-relative virtual addresses, NCCL window registration, multi-host
  RDMA, pointer translation, or CUDA IPC across processes.
- Changing MoK math, scheduling, routing, parameter layout, or tolerances.
- Allowing a remote pointer to outlive the custom-call stream dependency.
