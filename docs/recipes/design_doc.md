# Recipe: Writing Design Documents

## Purpose

Design docs should be **tight, readable 1-pagers** that explain the problem, reference key code, propose an approach, and reflect on trade-offs. Avoid exhaustive code dumps and detailed implementation plans.

**Target length**: ~200-400 lines including example

---

# Example Design Doc: Step Execution Time Tracking

N.B.

This design doc was machine generated and is primarily used for agent task tracking.
DO NOT use it as a canonical source of information about the project.
It is included in this commit for reviewer reference and feedbacka.
It may be deleted in the future.

## Problem

The Executor framework (`src/marin/execution/executor.py`) provides no visibility into step execution times. When pipelines run slow (e.g., 6 hours for 50 steps), developers can't identify bottlenecks without manual instrumentation.

**Current behavior**: `_launch_step()` at line 664 launches Ray tasks but never records timing. `ExecutorStepInfo` (line 324) has no timing fields.

## Goals

- Capture execution time per step with <1% overhead
- Expose timing in `.executor_info` JSON and status events
- Distinguish Ray scheduling overhead from actual execution time
- Backwards compatible (existing experiments unchanged)

**Non-goals**: Distributed tracing, real-time metrics, CPU/memory profiling

## Proposed Solution

### Core Approach

1. **Extend status events** - Add `execution_start_time` and `execution_end_time` fields to `StatusEvent` (executor_step_status.py:15)
2. **Wrap step functions** - In `_launch_step()`, wrap functions to capture `time.time()` before/after execution and write to status events
3. **Compute on read** - `get_infos()` (line 854) reads status events and calculates durations

**Key principle**: Use existing status infrastructure rather than creating parallel timing logs. Avoids filesystem races and integrates naturally with step lifecycle (RUNNING → SUCCESS/FAILED).

### Data Schema

```python
# executor_step_status.py:15 - Add to existing StatusEvent
execution_start_time: float | None = None  # Unix timestamp
execution_end_time: float | None = None

# executor.py:324 - Add to ExecutorStepInfo
scheduled_time: str | None = None           # When STATUS_RUNNING recorded
execution_start_time: str | None = None     # When step function began
execution_end_time: str | None = None       # When step function completed
duration_seconds: float | None = None       # Execution time only
total_duration_seconds: float | None = None # Including Ray scheduling
```

### Timing Capture

Modify `_launch_step()` to wrap step functions:

```python
def _create_timed_wrapper(self, original_fn: Callable, output_path: str) -> Callable:
    def timed_wrapper(config):
        start = time.time()
        try:
            result = original_fn(config)
            end = time.time()
            append_status(status_path, STATUS_SUCCESS,
                         execution_start_time=start, execution_end_time=end)
            return result
        except Exception as e:
            append_status(status_path, STATUS_FAILED, message=str(e),
                         execution_start_time=start, execution_end_time=time.time())
            raise
    return timed_wrapper
```

Use wrapper when launching Ray tasks (both remote and normal functions).

### Timing Extraction

Add `_extract_timing_from_events(events)` method to parse status events:
- Find first RUNNING event → `scheduled_time`
- Find execution_start_time/execution_end_time from event fields
- Compute durations if data available

Call from `get_infos()` when building ExecutorStepInfo objects.

## Implementation Outline

1. **Extend schema** - Add timing fields to StatusEvent dataclass and append_status() signature
2. **Capture timing** - Add _create_timed_wrapper() method and modify _launch_step() to use it
3. **Expose timing** - Add _extract_timing_from_events() and update get_infos() to populate timing fields
4. **Test** - Verify timing captured (test_step_timing), persisted to JSON (test_timing_in_executor_json), and works on failures (test_timing_on_failure)

No breaking changes. Fields are optional (None for old data). No deprecation needed.

## Notes

- Wrapper overhead <1% for typical multi-second steps
- Uses existing status infrastructure (no new systems)
- `total_duration_seconds` includes Ray scheduling overhead; `duration_seconds` is execution only

## Future Work

- Pipeline visualization (Gantt charts)
- Integration with Ray tracing for distributed view
- CPU/memory metrics per step
- Alerting on timing regressions

---

# Guidelines for Agents

When writing a design doc:

## Structure

**Required sections**:
1. **Problem** - Concrete scenario with file:line references
2. **Goals** - Specific objectives and explicit non-goals
3. **Proposed Solution** - Core approach with minimal code snippets (10-30 lines max)
4. **Implementation Outline** - Brief bullet points (4-6 items)
5. **Notes** - Important details about the approach
6. **Future Work** - What's deliberately excluded

**Target length**: 200-400 lines total

## Writing Style

- Reference actual files with line numbers (`executor.py:664`)
- Show concrete examples (not placeholders like "[describe problem]")
- Include small code snippets showing **core idea** (10-30 lines max)
- Explain **why** you made design choices
- State backwards compatibility upfront
- Be concise - every line should add value
- No massive code dumps (>30 lines)
- No detailed implementation plans with phases/substeps/validation commands
- No abstract placeholders ("update the configuration")

## Code Snippets

Show the key idea in 10-30 lines, not complete implementations:

```python
# Shows core concept
def timed_wrapper(fn):
    start = time.time()
    result = fn()
    append_status(path, SUCCESS, execution_start_time=start, execution_end_time=time.time())
    return result
```

Skip exhaustive details (error handling, edge cases, etc.) - those belong in implementation.

## Implementation Outline

4-6 high-level bullet points:

```markdown
1. Extend StatusEvent schema with timing fields
2. Wrap step functions to capture time.time()
3. Parse timing from status events in get_infos()
4. Test timing capture, persistence, and failure cases
```

Include test approach as one bullet. Don't include detailed phases, substeps, file paths, or validation commands.

## See Also

- `AGENTS.md` - Coding guidelines
- @docs/recipes - Recipes for how to operate on the codebase.
