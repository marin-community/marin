# Debugging log for Zephyr Iris Test Failures

## Goal
Fix 3 failing iris backend tests:
- test_shared_data[iris] - returns [] instead of [10, 20, 30]
- test_context_manager[iris] - returns [] instead of expected results
- test_flat_map[iris] - returns [] instead of expected results

## Initial status

Running `uv run pytest tests/test_execution.py -v` shows:
- 15/18 tests passing
- 3 iris tests failing with empty results
- All local backend tests pass
- Workers are executing (tasks show COMPLETE state in logs)
- No exceptions thrown, just empty results

## Hypothesis 1: Results not being collected from coordinator

The workers are executing (logs show tasks completing), but `ctx.execute()` returns empty lists. This suggests the issue is in how results are being collected after execution completes.

Checking the execution flow:
1. Workers pull tasks and execute them
2. Workers report results to coordinator via `report_result()`
3. After stage completes, `collect_results()` should return the results
4. Results should be materialized and returned

The issue is likely in one of these steps.

## Changes to investigate

Files examined:
- lib/zephyr/tests/test_execution.py - test code ✓
- lib/zephyr/src/zephyr/execution.py - ZephyrContext.execute() and result collection ✓

Key observations:
- Workers call `coordinator.report_result.remote(worker_id, task.shard_idx, result)` without awaiting
- Main thread polls `get_status()` and collects when `completed >= total`
- Actor calls should be serialized, so if status shows completed, report_result should have processed
- But results are empty, suggesting report_result isn't storing results correctly

Hypothesis: The worker is calling `report_result.remote()` but not awaiting it. Since actor futures in Iris use thread pools, there might be a race where the main thread calls `collect_results()` before `report_result` completes.

## Testing - Add debug logging

Adding debug output to see what workers are returning and what coordinator receives.

## Results

Found the root cause! The issue was that several async actor calls were not being awaited:

1. `coordinator.set_shared_data.remote()` - not awaited
2. `coordinator.start_stage.remote()` - not awaited (CRITICAL!)
3. `coordinator.signal_done.remote()` - not awaited

The most critical issue was `start_stage.remote()` not being awaited. This caused a race condition:

1. Main thread calls `start_stage.remote()` - returns immediately (async)
2. Main thread polls `get_status()` - coordinator hasn't processed start_stage yet, so `total=0`
3. Main thread sees `completed=0, total=0` → `0 >= 0` is True → exits polling immediately
4. Main thread collects empty results
5. Coordinator processes start_stage later (too late)
6. Workers start and see `done=True`, exit without processing

## Solution

Changed all critical async calls to wait for completion by calling `.result()`:

```python
# Before
coordinator.set_shared_data.remote(self._shared_data)
coordinator.start_stage.remote(stage.stage_name(), tasks)
coordinator.signal_done.remote()

# After
coordinator.set_shared_data.remote(self._shared_data).result()
coordinator.start_stage.remote(stage.stage_name(), tasks).result()
coordinator.signal_done.remote().result()
```

This ensures the coordinator has processed each operation before the main thread proceeds.

## Test Results

✅ All 18 tests passing (9 local + 9 iris)
- test_simple_map[local/iris]
- test_filter[local/iris]
- test_shared_data[local/iris]
- test_multi_stage[local/iris]
- test_context_manager[local/iris]
- test_write_jsonl[local/iris]
- test_dry_run[local/iris]
- test_flat_map[local/iris]
- test_empty_dataset[local/iris]

## Completed Work

- ✅ Removed excessive debug logging (kept error logging only)
- ✅ Documented control flow in lib/zephyr/README.md
- ✅ All async coordinator calls now properly awaited
- ✅ All 18 tests passing (9 local + 9 iris)

## Key Changes Made

1. **lib/zephyr/src/zephyr/execution.py**:
   - Added `.result()` to `set_shared_data.remote()`
   - Added `.result()` to `start_stage.remote()` (critical fix!)
   - Added `.result()` to `signal_done.remote()`
   - Cleaned up verbose debug logging

2. **lib/zephyr/README.md**:
   - Added comprehensive "Execution Control Flow" section
   - Documented data flow between stages
   - Explained why data is in-memory, not files
   - Documented critical synchronization points
   - Explained shard data format
   - Documented error handling strategy

3. **lib/zephyr/docs/iris-test-failures.md**:
   - Complete debugging log of investigation
   - Root cause analysis
   - Solution details

## Future Considerations

- [ ] Consider adding timeout guards for actor calls
- [ ] Review other async calls in codebase to ensure they're properly awaited
- [ ] Consider adding assertions to detect `total=0` polling exit as a guard against future regressions
