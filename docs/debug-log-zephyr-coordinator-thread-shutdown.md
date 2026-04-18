# Debugging log for zephyr coordinator thread shutdown

Fix the PR 3963 CI failure where the Zephyr test suite finished successfully, then crashed during interpreter shutdown because a background coordinator thread was still logging to a closed stderr stream.

## Initial status

CI reported a fatal shutdown crash after the Zephyr tests passed:

- `ValueError: I/O operation on closed file` from `logging`
- stack frames in `zephyr.execution.ZephyrCoordinator._coordinator_loop()` and `_log_status()`
- fatal Python shutdown error mentioning daemon threads and a locked `stderr`

The failure happened after normal test completion, which pointed to a teardown leak rather than a functional pipeline error.

## Hypothesis 1

The real `ZephyrContext.execute()` teardown path leaves the coordinator actor's background thread alive on the local backend.

## Changes to make

- Inspect `_run_coordinator_job()` in `lib/zephyr/src/zephyr/execution.py`
- Inspect `LocalClient.host_actor()` in `lib/fray/src/fray/v2/local_backend.py`
- Add a regression test in `lib/zephyr/tests/test_execution.py` that exercises `execute()` and waits for `zephyr-coordinator-loop` to disappear

## Future Work

- [ ] Consider teaching `LocalClient.host_actor()` to attach a generic stop hook for hosted actors that implement `shutdown()`
- [ ] Audit other daemon-thread actors to make sure their lifecycle is owned by the caller

## Results

`LocalClient.host_actor()` returns `HostedActor(handle)` with no stop callback, so `hosted.shutdown()` is a no-op on the local backend. `ZephyrCoordinator.initialize()` starts a daemon `zephyr-coordinator-loop` thread, and `_run_coordinator_job()` previously never called `coordinator.shutdown()`. The fix is to explicitly invoke `coordinator.shutdown.remote().result()` in the coordinator job's `finally` block before `hosted.shutdown()`.

The regression test covers the real `execute()` path and asserts that no extra live `zephyr-coordinator-loop` thread remains after execution returns.
