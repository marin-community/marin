# Debugging log for qsplit240 parent SIGSEGV during Iris fan-out

Investigate repeated parent-process `SIGSEGV` crashes while launching the
qsplit240 300M/6B many-domain sweep on Iris.

## Initial status

The parent Iris orchestrator for the qsplit240 300M/6B sweep crashed multiple
times:

- `8GB / 256 concurrent`: parent OOM
- `32GB / 64 concurrent`: parent `SIGSEGV` after fan-out
- `64GB / 32 concurrent`: parent `SIGSEGV` after longer fan-out
- `64GB / 16 concurrent`: parent `SIGSEGV` after even longer fan-out

Two distinct crash strings were observed:

- `Failed to compress input file`
- crash during Fray/Iris client initialization (`current_client: using Iris backend`)

Lower concurrency increased time-to-failure, which suggested orchestration
pressure rather than a deterministic one-shot config error.

## Hypothesis 1

The parent is repeatedly re-compressing the workspace bundle for each child,
eventually exhausting memory or disk.

## Changes to make

- Inspect the Iris bundle creation path.
- Check whether child jobs inherit a parent `bundle_id` or recreate bundles.

## Results

- Iris bundles are created by
  [bundle.py](/Users/calvinxu/Projects/Work/Marin/marin/lib/iris/src/iris/cluster/client/bundle.py),
  which writes a single temp zip and caps it at `25MB`.
- Child jobs submitted from within an Iris job inherit the parent `bundle_id`
  through
  [client.py](/Users/calvinxu/Projects/Work/Marin/marin/lib/iris/src/iris/client/client.py)
  and
  [remote_client.py](/Users/calvinxu/Projects/Work/Marin/marin/lib/iris/src/iris/cluster/client/remote_client.py).
- So the parent is **not** rebuilding a new workspace bundle per child. This
  makes “insufficient local disk for repeated workspace compression” an unlikely
  primary cause.

## Hypothesis 2

Each StepRunner worker thread is auto-detecting and creating its own Iris
client, instead of reusing one shared parent client. Under heavy threaded
fan-out, repeated Iris client initialization can explain:

- the `SIGSEGV` during Fray/Iris client init
- the fact that lower concurrency extends survival time

## Changes to make

- Patch the executor path so that, when any resolved step is remote, the parent
  captures one Fray client with `current_client()` and seeds it into context
  with `set_current_client(...)` before `StepRunner().run(...)`.
- Add a regression test asserting that `Executor.run(...)` seeds the
  auto-detected Fray client before remote-step execution.

## Results

- Patched
  [executor.py](/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/execution/executor.py)
  to capture one shared Fray client for remote steps and scope it across the
  step runner.
- Added a regression test in
  [test_executor.py](/Users/calvinxu/Projects/Work/Marin/marin/tests/execution/test_executor.py)
  to verify that remote executor runs inherit the auto-detected client instead
  of leaving worker threads to auto-detect independently.

## Future Work

- [ ] Relaunch the qsplit240 sweep from the patched workspace and confirm the
      parent survives past the prior failure window.
- [ ] If parent crashes persist, shard the 240-run sweep into multiple parent
      jobs instead of relying on one long-lived orchestrator.
- [ ] If crashes remain even with a shared client, inspect the native library
      path behind the `Failed to compress input file` message directly.
