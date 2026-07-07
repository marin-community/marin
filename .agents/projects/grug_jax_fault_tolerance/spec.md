# Grug JAX Fault Tolerance Spec

This spec pins the public surfaces proposed by `design.md`. It does not prescribe implementation sequencing beyond file ownership and contracts.

## New Shared Transfer Package

New package: `lib/marin/src/marin/transfer/`

### `lib/marin/src/marin/transfer/base.py`

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from jaxtyping import PyTree


class TransferMode(StrEnum):
    CHECKPOINT = "checkpoint"
    ARROW_FLIGHT = "arrow_flight"
    JAX_TRANSFER = "jax_transfer"


@dataclass(frozen=True)
class TransferConfig:
    mode: TransferMode = TransferMode.CHECKPOINT
    namespace: str = "default"
    coordinator_name: str = "transfer_coordinator"
    timeout_seconds: float = 600.0
    checkpoint_dir: str = ""
    max_retained_payloads: int | None = 5
    flight_host: str = "0.0.0.0"
    flight_port: int = 0
    preserve_dtypes: bool = True


@dataclass(frozen=True)
class TransferManifest:
    namespace: str
    payload_id: int
    backend: TransferMode
    rank_locations: Mapping[int, str] = field(default_factory=dict)
    keys: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class TransferResult:
    payload_id: int
    tree: PyTree
    manifest: TransferManifest
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass
class TransferMetrics:
    # Cumulative counters.
    total_publishes: int = 0
    successful_publishes: int = 0
    failed_publishes: int = 0
    total_fetches: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    total_transfer_bytes: int = 0
    # Last-transfer gauges.
    transfer_bytes: int = 0
    largest_leaf_bytes: int = 0
    leaf_count: int = 0
    materialize_time: float = 0.0
    serialize_time: float = 0.0
    fetch_time: float = 0.0
    decode_time: float = 0.0


class TransferPublisher(Protocol):
    def publish(
        self,
        payload_id: int,
        tree: PyTree,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> TransferManifest:
        """Publish one exact pytree payload.

        `payload_id` is caller-owned and may move backward after recovery. Backends must
        preserve array dtype and shape when `TransferConfig.preserve_dtypes` is true.
        A successful return means later subscribers in the same namespace can observe the
        manifest. Implementations may retain only the latest payload unless configured
        otherwise.
        """

    def close(self) -> None:
        """Release backend resources. Multiple calls must be safe."""

    def metrics(self) -> TransferMetrics:
        """Return cumulative local publisher metrics."""


class TransferSubscriber(Protocol):
    def fetch(
        self,
        template: PyTree,
        *,
        after: int | None = None,
        timeout_seconds: float | None = None,
    ) -> TransferResult | None:
        """Fetch the latest payload for this subscriber's namespace.

        `template` defines the expected pytree structure and target sharding. If `after`
        is provided, a manifest with `payload_id == after` is treated as already
        consumed. Rollback is accepted only when the coordinator's latest manifest has
        moved to a different payload ID, including a lower one. Return `None` when no
        compatible payload is available before the timeout.
        """

    def close(self) -> None:
        """Release backend resources. Multiple calls must be safe."""

    def metrics(self) -> TransferMetrics:
        """Return cumulative local subscriber metrics."""
```

### `lib/marin/src/marin/transfer/coordinator.py`

```python
class TransferCoordinator(Protocol):
    def publish(self, manifest: TransferManifest) -> None:
        """Record the latest manifest for `manifest.namespace`.

        New manifests replace older manifests, including rollback payload IDs.
        Exact duplicate `(namespace, payload_id, backend, rank_locations, keys)` updates are ignored.
        Publish is atomic: subscribers must not observe a manifest until the backend
        payload is complete enough to fetch.
        """

    def latest(self, namespace: str) -> TransferManifest | None:
        """Return the most recently published manifest for `namespace`, if any."""
```

The first concrete implementation can be a Fray/Iris actor, matching the current
RL Arrow Flight coordinator shape. The interface must support cross-process
discovery and rank-address mappings; in-process dictionaries are only valid for
unit tests.

### `lib/marin/src/marin/transfer/checkpoint.py`

```python
class CheckpointTransferPublisher(TransferPublisher):
    def __init__(
        self,
        config: TransferConfig,
        *,
        axis_mapping: object | None = None,
        mesh: object | None = None,
    ) -> None: ...


class CheckpointTransferSubscriber(TransferSubscriber):
    def __init__(
        self,
        config: TransferConfig,
        *,
        axis_mapping: object | None = None,
        mesh: object | None = None,
    ) -> None: ...
```

Checkpoint payload layout is:

```text
<checkpoint_dir>/<namespace>/payload_<payload_id>/
```

The backend uses Levanter checkpoint save/load. It must accept arbitrary pytrees matching the subscriber template. It must not require RL model types.

For checkpoint mode, `TransferConfig.checkpoint_dir` is required. Publishers write
to a temporary payload directory, finalize Levanter checkpoint metadata, and only
then publish the manifest. Subscribers load from manifests returned by
`TransferCoordinator`; they must not discover payloads by raw directory listing
except in test-only fallback utilities.

### `lib/marin/src/marin/transfer/arrow_flight.py`

```python
class ArrowFlightTransferPublisher(TransferPublisher):
    def __init__(self, config: TransferConfig, *, coordinator_handle=None) -> None: ...


class ArrowFlightTransferSubscriber(TransferSubscriber):
    def __init__(self, config: TransferConfig, *, coordinator_handle=None) -> None: ...
```

The shared Arrow Flight backend transfers exact flattened pytree leaves with dtype and shape metadata. It must not downcast by default. RL-specific bfloat16 inference conversion remains outside this backend.

### `lib/marin/src/marin/transfer/jax_transfer.py`

```python
class JaxTransferUnavailableError(RuntimeError):
    """Raised when `jax.experimental.transfer` or its jaxlib symbols are unavailable."""


class JaxTransferPublisher(TransferPublisher):
    def __init__(self, config: TransferConfig, *, coordinator_handle=None) -> None: ...


class JaxTransferSubscriber(TransferSubscriber):
    def __init__(self, config: TransferConfig, *, coordinator_handle=None) -> None: ...
```

The JAX backend starts a `jax.experimental.transfer` server and publishes server addresses through `TransferCoordinator`. It raises `JaxTransferUnavailableError` at construction time if the runtime cannot import and start the JAX transfer server. It uses the subscriber template to build the shape/dtype/sharding tree passed to `TransferConnection.pull`.

For JAX transfer mode, `payload_id` is the integer UUID passed to JAX transfer. A publisher calls
`await_pull(payload_id, tree, absolute_timeout)` after publishing a manifest with
`rank_locations`. A subscriber connects to each rank required by the template
sharding and calls `pull(payload_id, shape_dtype_tree, absolute_timeout)`.
Backend code must convert the relative `timeout_seconds` config into the absolute
timeout expected by JAX.

### `lib/marin/src/marin/transfer/__init__.py`

```python
def create_transfer_publisher(
    config: TransferConfig,
    *,
    mesh: object | None = None,
    axis_mapping: object | None = None,
    coordinator_handle=None,
) -> TransferPublisher:
    """Construct a publisher for `config.mode`."""


def create_transfer_subscriber(
    config: TransferConfig,
    *,
    mesh: object | None = None,
    axis_mapping: object | None = None,
    coordinator_handle=None,
) -> TransferSubscriber:
    """Construct a subscriber for `config.mode`."""
```

## RL Compatibility Surface

Existing imports from `marin.rl.weight_transfer` remain as adapters while RL call sites are updated. The adapter must preserve existing public imports, defaults, method names, and metric field names until call sites are removed. `WeightTransferConfig.mode` continues to default to `WeightTransferMode.ARROW_FLIGHT` and maps explicitly to `TransferMode.ARROW_FLIGHT`. RL-only fields such as `sync_interval_steps`, `max_weight_transfer_wait_time`, and `convert_to_bfloat16` remain in RL config.

## Levanter / Iris JAX Initialization

### `lib/levanter/src/levanter/distributed.py`

```python
@dataclass(frozen=True)
class DistributedConfig:
    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: int | list[int] | None = None
    initialize_jax_distributed: bool = True
    enable_recoverability: bool = False
    heartbeat_timeout_seconds: int | None = None
```

When `enable_recoverability` is true, `DistributedConfig.initialize()` must call `jax.config.update("jax_enable_recoverability", True)` before any `jax.distributed.initialize` call. If `heartbeat_timeout_seconds` is not `None`, it is passed to all explicit distributed-init calls.

### `lib/iris/src/iris/runtime/jax_init.py`

```python
def initialize_jax(
    port: int = 8476,
    endpoint_name: str = "jax_coordinator",
    poll_timeout: float = 300.0,
    poll_interval: float = 2.0,
    *,
    enable_recoverability: bool = False,
    heartbeat_timeout_seconds: int | None = None,
) -> None:
    """Initialize JAX distributed runtime using Iris endpoint discovery.

    Recoverability must be configured before distributed initialization. TPU jobs
    reject `enable_recoverability=True` until JAX supports the same semantics there.
    GPU and supervised multigpu paths pass `heartbeat_timeout_seconds` to
    `jax.distributed.initialize` when provided.
    """
```

New error:

```python
class JaxRecoverabilityUnsupportedError(RuntimeError):
    """Raised when recoverability is requested for a runtime that cannot support it."""
```

## Grug Fault Tolerance Surface

### `experiments/grug/fault_tolerance.py`

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GrugFaultToleranceConfig:
    enabled: bool = False
    heartbeat_timeout_seconds: int = 10


class GrugStepAtomicityError(RuntimeError):
    """Raised when a fault-tolerant Grug step cannot be committed atomically."""
```

`experiments/grug/base/train.py` and `experiments/grug/moe/train.py` both use
this helper module. Their `GrugTrainerConfig` types gain:

```python
fault_tolerance: GrugFaultToleranceConfig = field(default_factory=GrugFaultToleranceConfig)
```

Contract:

- Disabled mode preserves current Grug behavior.
- Enabled mode requires a GPU backend and JAX recoverability. TPU raises `JaxRecoverabilityUnsupportedError`.
- Before `trainer.initialize()`, Grug constructs a replaced `TrainerConfig` whose `distributed` config has `enable_recoverability=True` and `heartbeat_timeout_seconds=config.trainer.fault_tolerance.heartbeat_timeout_seconds`.
- If JAX distributed is already initialized without recoverability, Grug raises `JaxRecoverabilityUnsupportedError`.
- A train step is committed only after the `live_devices` context exits successfully and the loss has been blocked.
- On liveness failure, Grug raises `GrugStepAtomicityError` after preserving the pre-step state and skipping all callbacks/checkpoint hooks.
- The first recovery mode is abort-to-checkpoint: Grug relies on Fray/job retry to restart from the last durable checkpoint and does not continue in-process.

## Out Of Scope

- TPU fault-tolerant continuation through JAX recoverability.
- Continuing on a smaller live-device mesh in the first Grug milestone.
- In-process Grug `transfer_restore`; this needs a follow-up spec that defines which ranks fetch, from whom, and when.
- Changing Grug checkpoint file formats beyond the shared checkpoint-transfer layout.
- Replacing Fray retry semantics.
- Making Arrow Flight the default Grug recovery backend before exact-pytree tests exist.

## File Summary

| Path | Change |
|---|---|
| `lib/marin/src/marin/transfer/base.py` | New shared transfer protocols, config, manifest, result, metrics |
| `lib/marin/src/marin/transfer/coordinator.py` | New latest-manifest coordinator |
| `lib/marin/src/marin/transfer/checkpoint.py` | New checkpoint transfer backend |
| `lib/marin/src/marin/transfer/arrow_flight.py` | New exact-pytree Arrow Flight backend |
| `lib/marin/src/marin/transfer/jax_transfer.py` | New JAX transfer server backend |
| `lib/marin/src/marin/transfer/__init__.py` | Backend factory exports |
| `lib/marin/src/marin/rl/weight_transfer/` | Compatibility adapters over `marin.transfer` |
| `lib/levanter/src/levanter/distributed.py` | Recoverability and heartbeat init config |
| `lib/iris/src/iris/runtime/jax_init.py` | Iris distributed init plumbing |
| `experiments/grug/fault_tolerance.py` | Shared Grug fault-tolerance config, errors, and helper contracts |
| `experiments/grug/base/train.py` | Atomic step commit integration |
| `experiments/grug/moe/train.py` | Atomic step commit integration |
