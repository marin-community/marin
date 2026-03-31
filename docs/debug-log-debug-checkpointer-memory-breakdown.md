# Debugging log for debug_checkpointer memory breakdown

Expand `debug_checkpointer` so the next long-running RL checkpoint incident tells
us more than just aggregate RSS and checkpoint phase.

## Initial status

The current checkpoint instrumentation already narrows failures to
`tensorstore_serialize` vs `async_commit_in_flight`, and it samples trainer RSS
while the checkpoint is running. That was enough to prove where the v5p-8 run
was dying, but not enough to prove what owned the memory.

Claude's logbook correctly called out the blind spot: one RSS number cannot
separate checkpoint staging buffers from weight-transfer host copies, GC-lagged
NumPy arrays, or previous async-commit overlap.

## Hypothesis 1

The missing visibility is inside the trainer process, not just at the Iris job
level.

To make the next crash decisive, `debug_checkpointer` needs to log:

- the Python heap view (`tracemalloc`) during checkpoint phases
- whether a previous async commit thread is still alive
- whether GC can reclaim anything before serialize begins
- RL-specific host memory that sits alongside checkpointing, especially Arrow
  Flight's resident weight store
- immediately flushed logs so a SIGKILL/OOM does not strand the last useful
  debug lines in buffers

## Changes to make

- `lib/levanter/src/levanter/checkpoint.py`
  - add a debug state-provider registry
  - add manager-thread / pending-commit introspection
  - add `tracemalloc` current/peak and top-growth logging
  - add forced `gc.collect()` logging before serialize
  - add aggressive handler/stdout/stderr flushing on debug checkpoint logs
- `lib/levanter/src/levanter/tensorstore_serialization.py`
  - flush immediately after the high-value serialize boundary logs
- `lib/marin/src/marin/rl/train_worker.py`
  - register an RL-specific checkpoint debug provider while the trainer is live
- `lib/marin/src/marin/rl/weight_transfer/base.py`
  - add a default `get_debug_snapshot()` hook for server-side providers
- `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
  - expose resident Arrow Flight store size / latest transfer snapshot

## Results

Implemented under `debug_checkpointer`:

- `Checkpoint debug snapshot` logs now include:
  - RSS
  - Python `tracemalloc` current/peak
  - GC counts / garbage length
  - previous async-commit thread state
  - pending commit-future counts
  - registered provider state
- `Checkpoint debug gc` logs now run before serialize and record:
  - objects collected by `gc.collect()`
  - RSS before/after GC
  - traced Python heap before/after GC
  - tracked-object counts before/after GC
- tracemalloc growth since the post-GC baseline is logged at phase transitions
  and periodic checkpoint progress intervals
- RL trainers now register a provider that exposes:
  - replay-buffer stats and current step
  - Arrow Flight resident store snapshot and latest transfer metrics
- Arrow Flight now reports:
  - latest stored weight id
  - stored parameter count
  - stored record-batch count
  - stored Arrow bytes / MiB
  - latest transfer byte/time metrics
- TensorStore serialize boundary logs and checkpoint progress logs are flushed
  immediately

This does not yet *fix* the checkpoint memory issue. It makes the next incident
much more attributable:

- if the previous async commit is still alive, we will see it
- if GC can reclaim a large amount before serialize, we will see it
- if Arrow Flight resident bytes are nontrivial at checkpoint start, we will see
  them in the same debug snapshot
- if Python-heap growth is relevant, tracemalloc growth will show where

## Verification

- `cd lib/levanter && uv run pytest -q tests/test_checkpoint.py`
- `uv run pytest -q tests/rl/test_train_worker.py tests/rl/test_weight_transfer.py tests/rl/test_orchestration.py`

## Future Work

- [ ] If the next failure still dies too abruptly, add a lower-level native log
      sink for JAX/TensorStore/libtpu rather than relying only on Python logs.
- [ ] If Arrow Flight resident bytes are consistently large at checkpoint start,
      test clearing or rotating the store before serialize.
- [ ] If previous async commit overlap is common, consider an opt-in mode that
      waits for commit completion before starting the next serialize.
