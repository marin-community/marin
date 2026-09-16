# Debugging log for stratified-520m multihost sync failure

Investigate and fix the new 520M stratified failure:

`RuntimeError: multihost_broadcast_sync requires jax distributed client to be initialized`

## Initial status

Claude Code reported the failure on the region-agnostic 520M stratified run. The error points at Levanter's
multihost metadata sync, not at scheduling or TPU capacity. Recent startup changes already moved Iris jobs onto
`iris.runtime.jax_init.initialize_jax()`, so the likely regression is a mismatch between TPU runtime initialization and
Levanter's multihost sync assumptions.

## Hypothesis 1

TPU jobs can report `jax.process_count() > 1` before the unpublished JAX distributed client is populated, so
`multihost_broadcast_sync()` can fail even though TPU runtime itself is healthy.

## Changes to make

- Update `lib/levanter/src/levanter/utils/jax_utils.py` so multihost sync falls back to supported
  `jax.experimental.multihost_utils` primitives when the low-level distributed client is absent.
- Add focused tests in `lib/levanter/tests/test_jax_utils.py` covering the client-missing fallback for both
  broadcast and barrier sync.

## Future Work

- [ ] Add an end-to-end startup test that exercises tracker initialization under TPU-style multihost conditions.
- [ ] Audit other uses of `jax._src.distributed.global_state.client` in Levanter for the same assumption.

## Results

Code inspection showed:

- `lib/levanter/src/levanter/distributed.py` now defers Iris TPU jobs to `iris.runtime.jax_init.initialize_jax()`.
- `lib/iris/src/iris/runtime/jax_init.py` intentionally no-ops on TPU.
- `lib/levanter/src/levanter/tracker/wandb.py` calls `multihost_broadcast_sync()` whenever `jax.process_count() > 1`.
- `lib/levanter/src/levanter/utils/jax_utils.py` hard-failed when `distributed.global_state.client` was `None`.

That matches the reported failure signature. The fix is to preserve the current client-based path when available, but
fall back to `jax.experimental.multihost_utils.broadcast_one_to_all()` and
`jax.experimental.multihost_utils.sync_global_devices()` when the client is missing.
