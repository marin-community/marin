# Grug JAX Fault Tolerance

Grug should use JAX's fault-tolerant multi-controller support without making recovery logic Grug-specific. The useful abstraction is a shared Marin transfer service: RL weight transfer becomes one consumer of it, checkpoint and Arrow Flight are implementations, and JAX transfer server can be resurrected as the device-array backend. With that in place, Grug can add GPU-only step atomicity and recovery without copying RL-specific weight-transfer code into the Grug template.

## Challenges

JAX fault tolerance is experimental and currently fully supported only on GPUs. Base Grug defaults to TPU, while the Grug MoE path already has GPU `processes_per_task` launch wiring. That means the design has to be explicit about GPU-only support and avoid implying TPU recovery through this API.

The hard part is commit atomicity. Grug currently runs a jitted train step, blocks on loss, then runs callbacks and checkpoints the resulting state. JAX's fault-tolerance guide warns that a collective interrupted by process failure can raise, return, or produce divergent values on different processes unless the work is wrapped in liveness-barrier semantics. For Grug, checkpointing a step is the commit point, so callback and checkpoint side effects must move behind the successful liveness check.

The other hard part is ownership. RL has the closest transfer machinery today, but its public names and semantics are weight-transfer oriented. Grug recovery needs full train-state transfer, including optimizer state, EMA state, and scalar step, with exact dtype preservation. The shared interface should therefore move out of RL before Grug depends on it.

## Costs / Risks

- Extracting a neutral transfer package creates churn in RL before Grug gets visible recovery behavior.
- JAX transfer server is an experimental backend and may be unavailable on some wheels; the local macOS wheel exposes the Python module but lacks the C++ symbol.
- Fault-tolerant Grug will initially be GPU-only, so TPU runs continue to rely on Fray retry plus checkpoint restore.
- Wrapping steps in `live_devices` adds runtime overhead and another failure mode in the hot training loop.

## Design

The long-lived abstraction is a neutral `marin.transfer` package. It owns payload IDs, manifests, metrics, publisher/subscriber protocols, and backend factories. RL call sites should move to that package, keeping RL-specific scheduling and inference conversion outside the shared backend. The checkpoint backend can be lifted almost directly from RL's `GCSCheckpointServer` and `GCSCheckpointClient`, which already save and load pytrees through Levanter checkpointing. Arrow Flight stays available, but the shared Arrow backend must preserve exact dtypes; RL can opt into inference-specific bfloat16 conversion before publishing.

JAX transfer server becomes another backend of this same interface, not a separate Grug mechanism. Its backend starts a `jax.experimental.transfer` server, publishes each rank's address through the shared coordinator, and transfers shape/dtype-described pytrees with `await_pull` and `pull`. Because local import fails on the macOS wheel, this backend is gated behind runtime capability detection and a GPU smoke test before Grug uses it.

JAX recoverability belongs in distributed initialization, not a new Grug-side config path. Levanter's `DistributedConfig`, already reachable through Grug's existing `config.trainer.trainer: TrainerConfig`, should grow recoverability and heartbeat fields. When enabled, it sets `jax_enable_recoverability` before distributed init and passes `heartbeat_timeout_seconds` through Iris's `initialize_jax` GPU paths. TPU paths should reject this mode for now rather than silently running without the promised semantics.

Grug should read that existing `TrainerConfig` instead of adding a second recoverability switch. When `config.trainer.trainer.distributed.enable_recoverability` is false, the loop is unchanged. When it is true, Grug checks that the runtime is GPU, builds the normal compact mesh, and wraps each train step with `jax.experimental.multihost_utils.live_devices(jax.devices())`. The train step still returns a new `GrugTrainState`, but that state is only committed to callbacks and checkpointing if the liveness context exits successfully after `jax.block_until_ready`. If the liveness check fails, Grug raises a step-atomicity error, skips callbacks/checkpointing, and relies on Fray retry to restart from the last durable checkpoint. It does not continue in-process in the first milestone.

Argument donation makes that abort-only contract necessary. Base and MoE Grug donate the train-state argument into the jitted step, so a failed step cannot assume the pre-step device buffers are still reusable. `live_devices` provides side-effect atomicity, not memory-state rollback: after a liveness failure the process should treat the in-memory state as poisoned and exit through the retry path.

This makes the milestone split important. M0 is a Grug-only abort-to-checkpoint change and does not require `marin.transfer`. Transfer service starts paying for itself in M1, where Grug can choose one of two recovery modes:

- Turn off train-state argument donation at the recovery boundary and recover through `marin.transfer`. In this mode the old in-memory state remains eligible to publish after a failed step, but disabling donation alone is not enough: a dead rank's unique shards are gone unless they were already externalized or replicated elsewhere. The transfer backend must therefore either hold all rank-local shards from the latest committed step or be restricted to states whose shards can be reconstructed from survivors.
- Keep donation in the hot step and periodically offload a full backup state through `marin.transfer`. Recovery uses the newest transfer payload, falling back to the durable checkpoint if no complete payload is available. This preserves the memory benefit of donation most steps, but introduces a tunable backup interval and transfer bandwidth cost.

This design allows an atomic Grug-only patch to land before the transfer extraction. The shared transfer package is required before M1 transfer recovery modes or a reusable JAX transfer-server backend. Dynamic reduced-device meshes also stay out of the first milestone. `compact_grug_mesh` currently reshapes all `jax.devices()`, so continuing with fewer live devices requires careful divisibility checks and data-loader behavior.

## Testing

The transfer extraction should preserve RL behavior first. Existing RL weight-transfer tests should move with the updated RL call sites, plus new tests that publish and fetch an arbitrary pytree through the shared checkpoint backend without importing `marin.rl`.

The JAX init change needs focused unit tests around argument passing: recoverability is set before distributed init, heartbeat timeout reaches GPU `jax.distributed.initialize`, and TPU recoverability fails fast. The Grug loop needs a behavior test with a fake liveness context proving callbacks and checkpoint hooks run only after successful atomic completion.

The live proof should run on a small GPU Grug/MoE job with one process per GPU, using a shared helper so base and MoE do not diverge. Kill a nonzero process during a step and verify that no checkpoint metadata advances past the last atomic successful step. A separate GPU smoke should prove the JAX transfer backend can start, publish rank addresses, and pull a small sharded pytree before Grug recovery depends on it.

For M1, tests should distinguish the two transfer modes. The no-donation mode needs a fault-injection test proving the donated-state jitted variant is not used and recovery never depends on invalidated buffers. The backup-offload mode needs a test proving only fully published payloads are accepted, incomplete payloads fall back to checkpoint restore, and recovery can lose at most `backup_interval_steps - 1` committed steps relative to the latest checkpoint.

## Open Questions

- Should the first Grug proof land in the MoE variant, where GPU `processes_per_task` already exists, or should base Grug receive GPU launch plumbing first?
- For M1, is the right first transfer mode no-donation live recovery, periodic backup offload, or both behind one config surface?
- Should `marin.transfer` stay in `lib/marin`, or should the interface live lower if Levanter should eventually consume it without depending on Marin?
