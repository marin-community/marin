# Background Research Brief

- Effort: medium
- Stop rule: stopped after JAX fault-tolerance semantics, Grug train-loop commit points, RL transfer prior art, JAX transfer-server shape, and distributed-init ownership converged on the same design.
- Date: 2026-07-07

## Question

How should Grug integrate JAX fault-tolerant distributed execution, and where should the state-transfer abstraction live?

## Current Marin Context

Grug is template-first. The canonical edit surface is `experiments/grug/base/`, not a shared trainer stack. The base train loop initializes Levanter/JAX, builds a compact Grug mesh, restores checkpoints, executes a jitted `train_step`, blocks on loss, runs callbacks, and checkpoints the new state. The load-bearing lines are the initialization and loop in [`experiments/grug/base/train.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/experiments/grug/base/train.py#L418), mesh construction in [`train.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/experiments/grug/base/train.py#L447), train-step commit path in [`train.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/experiments/grug/base/train.py#L570), and checkpoint callback in [`train.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/experiments/grug/base/train.py#L621).

Grug dispatch already goes through Fray and forwards runtime-tuning environment variables in [`experiments/grug/dispatch.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/experiments/grug/dispatch.py#L40). The MoE variant already has GPU `processes_per_task` launch plumbing, which makes it the natural first runtime for GPU fault-tolerance experiments. `compact_grug_mesh` currently assumes a static all-device mesh from `jax.process_count()`, `jax.local_device_count()`, and `jax.devices()` in [`lib/levanter/src/levanter/grug/sharding.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/lib/levanter/src/levanter/grug/sharding.py#L144).

Distributed JAX initialization is not Grug-owned. Levanter calls `DistributedConfig.initialize()`, which delegates Iris jobs to `iris.runtime.jax_init.initialize_jax()`. Iris TPU initialization currently calls `jax.distributed.initialize()` with TPU runtime autodiscovery, while GPU paths pass coordinator/process arguments but not heartbeat timeout in [`lib/iris/src/iris/runtime/jax_init.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/lib/iris/src/iris/runtime/jax_init.py#L278) and [`jax_init.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/lib/iris/src/iris/runtime/jax_init.py#L291).

Both Grug base and MoE already carry a Levanter `TrainerConfig` at `config.trainer.trainer` and call `trainer.initialize()` on it before building the mesh. That existing nested config should remain the source of truth for recoverability and heartbeat settings.

## Prior Art

JAX's fault-tolerant distributed guide says multi-controller JAX is fault-intolerant by default: one failed process causes other processes to fail intentionally. It also warns the feature is experimental and currently fully supported only on GPUs. The public shape is `jax.config.update("jax_enable_recoverability", True)`, `jax.distributed.initialize(..., heartbeat_timeout_seconds=...)`, and `jax.experimental.multihost_utils.live_devices(jax.devices())` around collective work. The key semantic constraint is atomicity: a step must only commit after the post-work liveness check succeeds, because collectives may raise, return, or produce different results on different processes if a process fails mid-collective.

RL already has a transfer mechanism in `lib/marin/src/marin/rl/weight_transfer/`. The current base abstraction is weight-oriented: `WeightTransferServer.serve_weights(...)` and `WeightTransferClient.receive_weights(...)` in [`base.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/lib/marin/src/marin/rl/weight_transfer/base.py#L99). The checkpoint implementation saves and loads pytrees through Levanter checkpointing in [`checkpoint.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/lib/marin/src/marin/rl/weight_transfer/checkpoint.py#L44), which is directly applicable to full Grug train-state transfer. The Arrow Flight implementation starts multiple Flight servers and publishes model weights through a coordinator in [`arrow_flight.py`](https://github.com/marin-community/marin/blob/e45473443aa963ef86ce6e613aaf55dc32b04702/lib/marin/src/marin/rl/weight_transfer/arrow_flight.py#L397). It is fast, but currently model-oriented: it uses Haliax state dicts and can downcast floats for inference, which is not acceptable for optimizer-state recovery.

Installed `jax==0.10.1` also contains `jax.experimental.transfer`. Its source exposes `start_transfer_server`, `TransferServer.await_pull`, `TransferServer.connect`, and `TransferConnection.pull` over a pytree of shape/dtype structs in `.venv/lib/python3.12/site-packages/jax/experimental/transfer.py`. Locally, importing that module fails on the macOS wheel because the expected jaxlib C++ symbol is missing. That should be treated as platform availability to verify on target GPU jobs, not as a reason to ignore the backend.

## Negative Leads

- Do not promise TPU fault tolerance through this JAX API. Upstream marks full support GPU-only.
- Do not add only environment variables. The Grug loop must move callback/checkpoint commit behind `live_devices` atomicity.
- Do not leave the transfer abstraction owned by RL. RL and Grug have different payloads and consumers.
- Do not treat existing Arrow Flight code as exact train-state transfer. It is model-weight oriented and may perform lossy dtype conversion.
- Do not retry a failed donated step in-process with the old `GrugTrainState`. Base and MoE Grug donate the state argument, so the pre-step buffers may be invalid after the jitted step launches.
- Do not wire transfer service into M0 just to have it present. Transfer recovery only makes sense when Grug either disables donation at the recovery boundary or publishes a complete backup state before failure.
- Do not add a second Grug recoverability/heartbeat config. Grug already has `config.trainer.trainer: TrainerConfig`, and recoverability belongs in that existing distributed config.

## Evidence Map

### Claim: The abstraction should be shared Marin infrastructure, not Grug or RL code.

- Support:
  - RL already has server/client/coordinator patterns, metrics, checkpoint transfer, and Arrow Flight transfer.
  - Grug needs the same shape for recovery but with full `GrugTrainState` payloads.
  - JAX transfer server is naturally another backend under the same abstraction.
- Contradictions:
  - Extracting the abstraction creates churn before Grug has a working recovery run.
- Directness to Marin: high
- Confidence: strong
- Action: create a neutral `marin.transfer` package and adapt RL to consume it.

### Claim: Step atomicity is the first Grug train-loop change.

- Support:
  - JAX docs explicitly require liveness checks around collective work to prevent divergent commits.
  - Grug currently commits state, callbacks, and checkpointing after `block_until_ready`, without a liveness barrier.
  - Grug donates its train-state argument, so liveness failure must abort to durable state rather than retry with old in-memory buffers.
- Contradictions:
  - Existing Fray retries already restore after full job failure.
- Directness to Marin: high
- Confidence: stable
- Action: add opt-in Grug fault-tolerance config and wrap steps with `live_devices`.

### Claim: JAX transfer server should be a backend, but not the first dependency.

- Support:
  - Its API transfers device-array pytrees by pull, which matches recovering-rank state transfer.
  - It avoids making Grug depend on host serialization details.
  - Transfer service is only useful for Grug recovery once the design has a committed source of recoverable state: no-donation live state or periodic backup offload.
- Contradictions:
  - Local import fails due missing jaxlib C++ symbols.
  - API is experimental.
  - A live transfer backend cannot recover a dead rank's unique shards unless those shards were already externalized or replicated.
- Directness to Marin: medium-high
- Confidence: exploratory until tested on GPU
- Action: keep transfer service out of M0; specify it for M1 no-donation or backup-offload recovery and require a GPU smoke before Grug depends on JAX transfer.

## Recommended Plan

1. M0: add Levanter/Iris JAX recoverability plumbing on the existing `TrainerConfig.distributed`, then have Grug read that config for GPU-only `live_devices` step atomicity and durable checkpoint fallback.
2. Extract `marin.transfer`: neutral payload IDs, manifests, metrics, publisher/subscriber protocols, checkpoint backend, and adapters for RL's existing weight transfer.
3. M1 option A: add no-donation transfer recovery, where Grug compiles a no-donation train-step variant and restores from a complete committed transfer payload.
4. M1 option B: add periodic backup offload, where Grug keeps donation enabled and publishes complete train-state backups through `marin.transfer` every configured interval.
5. Add JAX transfer service backend under `marin.transfer` after a GPU smoke proves the C++ symbols and pull semantics work.
6. Add exact-pytree Arrow Flight backend only if checkpoint or JAX-transfer backup restore is too slow for Grug recovery.

## Open Questions

- Should the first runtime proof target Grug MoE on CoreWeave H100, since it already has `processes_per_task`, or should base Grug get the launch plumbing first?
- For M1, should we first spend memory by disabling donation at the recovery boundary or spend bandwidth by periodically offloading a backup state?
- Should the shared transfer package live at `marin.transfer` or in a lower-level package if Levanter should eventually use it directly?

## Source Ledger

| Source | Type | Location | Claim used for |
|---|---|---|---|
| JAX fault tolerance docs | official docs | https://docs.jax.dev/en/latest/fault_tolerance.html | GPU-only warning, recoverability flags, atomic `live_devices` semantics |
| Grug base train loop | Marin code | `experiments/grug/base/train.py` | Commit/checkpoint ordering |
| Grug dispatch | Marin code | `experiments/grug/dispatch.py` | Fray dispatch and env forwarding |
| Grug sharding | Marin code | `lib/levanter/src/levanter/grug/sharding.py` | Static all-device mesh assumption |
| Iris JAX init | Marin code | `lib/iris/src/iris/runtime/jax_init.py` | Distributed init ownership |
| RL weight transfer | Marin code | `lib/marin/src/marin/rl/weight_transfer/` | Existing transfer abstraction and implementations |
| JAX transfer server source | dependency source | `.venv/lib/python3.12/site-packages/jax/experimental/transfer.py` | Device-array transfer backend shape |
