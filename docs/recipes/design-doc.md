# Recipe: Writing Design Documents

## Purpose

This recipe provides a concrete example of a well-formed Marin design document. Follow this pattern when proposing changes to Marin's codebase.

**Key principle**: Design docs should be detailed enough that an agent (or developer) can implement them with minimal additional research. Avoid abstract placeholders - use concrete code examples.

---

# Example Design Doc: Step Execution Time Tracking

> **Meta-note**: This is a worked example showing the expected structure and detail level. Your design doc should follow this pattern.

## Background & Motivation

### Problem

Marin's Executor framework (`src/marin/execution/executor.py`) currently provides no visibility into how long individual steps take to execute. When debugging slow pipelines or optimizing resource usage, developers must manually instrument code or guess which steps are bottlenecks.

Example scenario: A user runs a 50-step pipeline for data processing. The pipeline takes 6 hours total, but they don't know if the slowness comes from tokenization (step 30) or deduplication (step 42). Without timing data, optimization is guesswork.

### Goals

1. **Capture execution time** for each ExecutorStep with <1% overhead
2. **Integrate with existing status system** - timing should appear in `.executor_info` files and status events
3. **Enable pipeline analysis** - expose timing data for visualization/debugging tools
4. **Maintain backwards compatibility** - existing experiments should work without changes

### Non-Goals

- Distributed tracing across Ray workers (future work)
- Real-time streaming metrics (use existing status events)
- Profiling CPU/memory within steps (use separate tools)

## Current Implementation

The Executor currently tracks step execution through the status system but doesn't capture timing:

```python
# src/marin/execution/executor.py:664
def _launch_step(self, step: ExecutorStep, *, dry_run: bool, force_run_failed: bool) -> tuple[ray.ObjectRef, bool]:
    config = self.configs[step]
    output_path = self.output_paths[step]

    # ... setup code ...

    if not dry_run:
        step_name = f"{step.name}: {get_fn_name(step.fn)}"
        should_execute = should_run(output_path, step_name, self.status_actor, ray_task_id, force_run_failed)

        if not should_execute:
            append_status(status_path, STATUS_SUCCESS, ray_task_id=ray_task_id, message="Step was already successful")
            return self._dry_run_result, False

        append_status(status_path, STATUS_RUNNING, ray_task_id=ray_task_id)  # ← No timing here

        # Launch Ray task (no timing wrapper)
        if isinstance(step.fn, ray.remote_function.RemoteFunction):
            ref = step.fn.options(name=f"{get_fn_name(step.fn, short=True)}:{step.name}", runtime_env=runtime_env).remote(config)
        else:
            remote_fn = ray.remote(step.fn)
            ref = remote_fn.options(name=f"{get_fn_name(step.fn, short=True)}:{step.name}", runtime_env=runtime_env).remote(config)

        return ref, True
```

**Current ExecutorStepInfo** (no timing fields):

```python
# src/marin/execution/executor.py:324
@dataclass(frozen=True)
class ExecutorStepInfo:
    name: str
    fn_name: str
    config: dataclass
    description: str | None
    override_output_path: str | None
    version: dict[str, Any]
    dependencies: list[str]
    output_path: str
    # ← Missing: start_time, end_time, duration_seconds
```

## Proposed Design

### Key Principles

1. **Timing doesn't block execution** - Capture timestamps asynchronously via status events
2. **Use existing status infrastructure** - Extend status events rather than creating parallel systems
3. **Compute on read** - Calculate duration when generating ExecutorInfo, not during execution
4. **Ray-aware** - Account for Ray task scheduling delays vs. actual execution time

### New Data Structures

```python
# src/marin/execution/executor_step_status.py:15 (add to existing file)
@dataclass
class StatusEvent:
    """A single status transition for a step."""
    timestamp: str  # ISO 8601
    status: str     # STATUS_RUNNING, STATUS_SUCCESS, etc.
    message: str | None = None
    ray_task_id: str | None = None
    # Add new fields:
    execution_start_time: float | None = None  # Unix timestamp when step function actually starts
    execution_end_time: float | None = None    # Unix timestamp when step function completes
```

**Rationale**: We add timing fields to `StatusEvent` rather than creating a separate timing log because:
- Status events already track the step lifecycle (RUNNING → SUCCESS/FAILED)
- Avoids filesystem race conditions between status and timing logs
- Natural place to distinguish "queued in Ray" vs. "executing"

### Modified ExecutorStepInfo

```python
# src/marin/execution/executor.py:324
@dataclass(frozen=True)
class ExecutorStepInfo:
    """Information about an ExecutorStep that can be serialized to JSON."""

    name: str
    fn_name: str
    config: dataclass
    description: str | None
    override_output_path: str | None
    version: dict[str, Any]
    dependencies: list[str]
    output_path: str

    # New timing fields (computed from status events)
    scheduled_time: str | None = None      # When step was scheduled (STATUS_RUNNING event)
    execution_start_time: str | None = None  # When step function began executing
    execution_end_time: str | None = None    # When step function completed
    duration_seconds: float | None = None    # execution_end_time - execution_start_time
    total_duration_seconds: float | None = None  # Including Ray scheduling overhead
```

**Example JSON output**:
```json
{
  "name": "tokenized/fineweb",
  "fn_name": "marin.processing.tokenize.tokenize",
  "output_path": "gs://marin-us-central2/tokenized/fineweb-8c2f3a",
  "scheduled_time": "2025-10-09T14:23:11.432Z",
  "execution_start_time": "2025-10-09T14:23:15.891Z",
  "execution_end_time": "2025-10-09T15:47:22.103Z",
  "duration_seconds": 5046.21,
  "total_duration_seconds": 5050.67
}
```

### Timing Capture Mechanism

```python
# src/marin/execution/executor.py:690 (modify existing code)
def _launch_step(self, step: ExecutorStep, *, dry_run: bool, force_run_failed: bool) -> tuple[ray.ObjectRef, bool]:
    # ... existing setup ...

    append_status(status_path, STATUS_RUNNING, ray_task_id=ray_task_id)  # Records scheduled_time

    # Wrap the step function to capture execution timing
    def timed_fn_wrapper(config):
        execution_start = time.time()
        try:
            result = original_fn(config)
            execution_end = time.time()

            # Record execution timing in status
            append_status(
                get_status_path(config.output_path) if hasattr(config, 'output_path') else status_path,
                STATUS_SUCCESS,
                ray_task_id=ray.get_runtime_context().get_task_id(),
                execution_start_time=execution_start,
                execution_end_time=execution_end,
            )
            return result
        except Exception as e:
            execution_end = time.time()
            append_status(
                get_status_path(config.output_path) if hasattr(config, 'output_path') else status_path,
                STATUS_FAILED,
                message=str(e),
                execution_start_time=execution_start,
                execution_end_time=execution_end,
            )
            raise

    # Launch wrapped function
    original_fn = step.fn._function if isinstance(step.fn, ray.remote_function.RemoteFunction) else step.fn
    if isinstance(step.fn, ray.remote_function.RemoteFunction):
        wrapped_remote = ray.remote(timed_fn_wrapper)
        ref = wrapped_remote.options(name=f"{get_fn_name(step.fn, short=True)}:{step.name}", runtime_env=runtime_env).remote(config)
    else:
        remote_fn = ray.remote(timed_fn_wrapper)
        ref = remote_fn.options(name=f"{get_fn_name(step.fn, short=True)}:{step.name}", runtime_env=runtime_env).remote(config)

    return ref, True
```

**Rationale for wrapper approach**:
- ✅ Captures actual execution time inside the Ray worker
- ✅ Works with both normal functions and Ray remote functions
- ✅ Handles exceptions correctly
- ✅ No changes needed to individual step functions
- ⚠️ Adds small overhead (~microseconds) for wrapper call

### Reading Timing Data

```python
# src/marin/execution/executor.py:854 (modify get_infos method)
def get_infos(self):
    """Calculates info files for each step and entire execution."""
    for step in self.steps:
        # Read status events to compute timing
        status_path = get_status_path(self.output_paths[step])
        events = read_events(status_path) if fsspec_utils.exists(status_path) else []

        timing = self._extract_timing_from_events(events)

        self.step_infos.append(
            ExecutorStepInfo(
                name=step.name,
                fn_name=get_fn_name(step.fn),
                config=self.configs[step],
                description=step.description,
                override_output_path=step.override_output_path,
                version=self.versions[step],
                dependencies=[self.output_paths[dep] for dep in self.dependencies[step]],
                output_path=self.output_paths[step],
                # New timing fields
                **timing
            )
        )
    # ... rest of method unchanged ...

def _extract_timing_from_events(self, events: list[dict]) -> dict:
    """Extract timing information from status events."""
    timing = {
        "scheduled_time": None,
        "execution_start_time": None,
        "execution_end_time": None,
        "duration_seconds": None,
        "total_duration_seconds": None,
    }

    for event in events:
        if event["status"] == STATUS_RUNNING and timing["scheduled_time"] is None:
            timing["scheduled_time"] = event["timestamp"]

        if event.get("execution_start_time") and timing["execution_start_time"] is None:
            timing["execution_start_time"] = datetime.fromtimestamp(event["execution_start_time"]).isoformat() + "Z"

        if event.get("execution_end_time"):
            timing["execution_end_time"] = datetime.fromtimestamp(event["execution_end_time"]).isoformat() + "Z"

    # Compute durations if we have the data
    if timing["execution_start_time"] and timing["execution_end_time"]:
        start = datetime.fromisoformat(timing["execution_start_time"].replace("Z", ""))
        end = datetime.fromisoformat(timing["execution_end_time"].replace("Z", ""))
        timing["duration_seconds"] = (end - start).total_seconds()

    if timing["scheduled_time"] and timing["execution_end_time"]:
        scheduled = datetime.fromisoformat(timing["scheduled_time"].replace("Z", ""))
        end = datetime.fromisoformat(timing["execution_end_time"].replace("Z", ""))
        timing["total_duration_seconds"] = (end - scheduled).total_seconds()

    return timing
```

## Implementation Plan

**Backwards Compatibility**: This change is backwards compatible. Existing experiments will work unchanged, but new timing fields will be `None` for steps executed before this change. No deprecation needed.

### Phase 1: Extend Status Event Schema

**Goal**: Add timing fields to status events without breaking existing code.

#### Step 1.1: Update StatusEvent dataclass

**File**: `src/marin/execution/executor_step_status.py`

```python
@dataclass
class StatusEvent:
    timestamp: str
    status: str
    message: str | None = None
    ray_task_id: str | None = None
    # New fields (optional for backwards compat)
    execution_start_time: float | None = None
    execution_end_time: float | None = None
```

**Validate**:
```bash
uv run pytest tests/test_executor.py::test_executor -v
# Should pass - new fields are optional
```

#### Step 1.2: Update append_status signature

**File**: `src/marin/execution/executor_step_status.py:75`

```python
def append_status(
    output_path: str,
    status: str,
    *,
    ray_task_id: str | None = None,
    message: str | None = None,
    execution_start_time: float | None = None,  # NEW
    execution_end_time: float | None = None,    # NEW
):
    """Append a status event to the status file."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    if message:
        event["message"] = message
    if ray_task_id:
        event["ray_task_id"] = ray_task_id
    if execution_start_time:
        event["execution_start_time"] = execution_start_time
    if execution_end_time:
        event["execution_end_time"] = execution_end_time

    # ... existing write logic ...
```

**Validate**:
```bash
uv run pytest tests/test_executor.py -v
# All tests should still pass
```

### Phase 2: Capture Timing in Executor

**Goal**: Wrap step functions to record execution time.

#### Step 2.1: Add timed wrapper function

**File**: `src/marin/execution/executor.py:664`

Add helper method to Executor class:

```python
def _create_timed_wrapper(self, original_fn: Callable, output_path: str) -> Callable:
    """Wrap a step function to capture execution timing."""
    def timed_wrapper(config):
        execution_start = time.time()
        status_path = get_status_path(output_path)

        try:
            result = original_fn(config)
            execution_end = time.time()

            # Don't override SUCCESS status from _launch_step, just add timing
            # Read current events to preserve them
            events = read_events(status_path) if fsspec_utils.exists(status_path) else []

            # Update the most recent SUCCESS event with timing
            if events and events[-1]["status"] == STATUS_SUCCESS:
                events[-1]["execution_start_time"] = execution_start
                events[-1]["execution_end_time"] = execution_end

                # Rewrite status file with updated event
                fsspec_utils.mkdirs(os.path.dirname(status_path))
                with fsspec.open(status_path, "w") as f:
                    for event in events:
                        print(json.dumps(event), file=f)

            return result
        except Exception as e:
            execution_end = time.time()
            # Timing is still useful even on failure
            append_status(
                status_path,
                STATUS_FAILED,
                message=str(e),
                execution_start_time=execution_start,
                execution_end_time=execution_end,
            )
            raise

    return timed_wrapper
```

**Validate**:
```bash
# Run a simple executor test
uv run pytest tests/test_executor.py::test_executor -v -s
# Check that timing appears in status file
```

#### Step 2.2: Modify _launch_step to use wrapper

**File**: `src/marin/execution/executor.py:690`

```python
def _launch_step(self, step: ExecutorStep, *, dry_run: bool, force_run_failed: bool) -> tuple[ray.ObjectRef, bool]:
    # ... existing setup code ...

    append_status(status_path, STATUS_RUNNING, ray_task_id=ray_task_id)

    # Create timing wrapper
    if isinstance(step.fn, ray.remote_function.RemoteFunction):
        original_fn = step.fn._function
        wrapped_fn = self._create_timed_wrapper(original_fn, output_path)
        wrapped_remote = ray.remote(wrapped_fn)
        ref = wrapped_remote.options(
            name=f"{get_fn_name(step.fn, short=True)}:{step.name}",
            runtime_env=runtime_env
        ).remote(config)
    else:
        wrapped_fn = self._create_timed_wrapper(step.fn, output_path)
        wrapped_remote = ray.remote(wrapped_fn)
        ref = wrapped_remote.options(
            name=f"{get_fn_name(step.fn, short=True)}:{step.name}",
            runtime_env=runtime_env
        ).remote(config)

    return ref, True
```

**Validate**:
```bash
uv run pytest tests/test_executor.py::test_executor tests/test_executor.py::test_parallelism -v
# Verify timing data appears in status files
```

### Phase 3: Add Timing to ExecutorStepInfo

**Goal**: Compute and expose timing in the executor info JSON.

#### Step 3.1: Extend ExecutorStepInfo dataclass

**File**: `src/marin/execution/executor.py:324`

```python
@dataclass(frozen=True)
class ExecutorStepInfo:
    name: str
    fn_name: str
    config: dataclass
    description: str | None
    override_output_path: str | None
    version: dict[str, Any]
    dependencies: list[str]
    output_path: str

    # Timing fields (None if step hasn't executed or was run before this feature)
    scheduled_time: str | None = None
    execution_start_time: str | None = None
    execution_end_time: str | None = None
    duration_seconds: float | None = None
    total_duration_seconds: float | None = None
```

**Validate**:
```bash
# Type check
uv run mypy src/marin/execution/executor.py
```

#### Step 3.2: Add timing extraction helper

**File**: `src/marin/execution/executor.py` (add new method)

```python
def _extract_timing_from_events(self, events: list[dict]) -> dict:
    """Extract timing information from status events.

    Returns dict with keys: scheduled_time, execution_start_time, execution_end_time,
    duration_seconds, total_duration_seconds. Values are None if data unavailable.
    """
    timing = {
        "scheduled_time": None,
        "execution_start_time": None,
        "execution_end_time": None,
        "duration_seconds": None,
        "total_duration_seconds": None,
    }

    for event in events:
        # Scheduled time = first RUNNING event
        if event["status"] == STATUS_RUNNING and timing["scheduled_time"] is None:
            timing["scheduled_time"] = event["timestamp"]

        # Execution times from event fields
        if event.get("execution_start_time") and timing["execution_start_time"] is None:
            timing["execution_start_time"] = datetime.fromtimestamp(
                event["execution_start_time"], tz=timezone.utc
            ).isoformat()

        if event.get("execution_end_time"):
            # Use latest end time (in case of retries)
            timing["execution_end_time"] = datetime.fromtimestamp(
                event["execution_end_time"], tz=timezone.utc
            ).isoformat()

    # Compute durations if we have the data
    if timing["execution_start_time"] and timing["execution_end_time"]:
        start = datetime.fromisoformat(timing["execution_start_time"])
        end = datetime.fromisoformat(timing["execution_end_time"])
        timing["duration_seconds"] = (end - start).total_seconds()

    if timing["scheduled_time"] and timing["execution_end_time"]:
        scheduled = datetime.fromisoformat(timing["scheduled_time"])
        end = datetime.fromisoformat(timing["execution_end_time"])
        timing["total_duration_seconds"] = (end - scheduled).total_seconds()

    return timing
```

**Validate**:
```bash
uv run pytest tests/test_executor.py -v
```

#### Step 3.3: Update get_infos to include timing

**File**: `src/marin/execution/executor.py:854`

```python
def get_infos(self):
    """Calculates info files for each step and entire execution."""
    for step in self.steps:
        # Read status events
        status_path = get_status_path(self.output_paths[step])
        events = []
        if fsspec_utils.exists(status_path):
            events = read_events(status_path)

        # Extract timing
        timing = self._extract_timing_from_events(events)

        self.step_infos.append(
            ExecutorStepInfo(
                name=step.name,
                fn_name=get_fn_name(step.fn),
                config=self.configs[step],
                description=step.description,
                override_output_path=step.override_output_path,
                version=self.versions[step],
                dependencies=[self.output_paths[dep] for dep in self.dependencies[step]],
                output_path=self.output_paths[step],
                # Add timing fields
                scheduled_time=timing["scheduled_time"],
                execution_start_time=timing["execution_start_time"],
                execution_end_time=timing["execution_end_time"],
                duration_seconds=timing["duration_seconds"],
                total_duration_seconds=timing["total_duration_seconds"],
            )
        )

    # ... rest unchanged ...
```

**Validate**:
```bash
# Run full executor test suite
uv run pytest tests/test_executor.py -v

# Manually verify timing appears in JSON
# (Add a test step that sleeps, check the output JSON has duration > sleep_time)
```

### Phase 4: Add Tests for Timing

**Goal**: Ensure timing works correctly and handles edge cases.

#### Step 4.1: Test basic timing capture

**File**: `tests/test_executor.py` (add new test)

```python
def test_step_timing():
    """Verify execution timing is captured in executor info."""
    import time

    def slow_fn(config: MyConfig | None):
        time.sleep(2)  # Simulate work

    step = ExecutorStep(name="slow_step", fn=slow_fn, config=None)

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        start_time = time.time()
        executor.run(steps=[step])
        end_time = time.time()

        # Check timing in step info
        step_info = executor.step_infos[0]
        assert step_info.duration_seconds is not None
        assert step_info.duration_seconds >= 2.0  # At least sleep time
        assert step_info.duration_seconds < 3.0   # Not too much overhead

        # Total duration should include Ray scheduling
        assert step_info.total_duration_seconds >= step_info.duration_seconds

        # All timestamp fields should be present
        assert step_info.scheduled_time is not None
        assert step_info.execution_start_time is not None
        assert step_info.execution_end_time is not None
```

**Validate**:
```bash
uv run pytest tests/test_executor.py::test_step_timing -v
```

#### Step 4.2: Test timing persists to JSON

**File**: `tests/test_executor.py` (add new test)

```python
def test_timing_in_executor_json():
    """Verify timing appears in the executor info JSON file."""
    import time

    def quick_fn(config: MyConfig | None):
        time.sleep(0.5)

    step = ExecutorStep(name="quick", fn=quick_fn, config=None)

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        executor.run(steps=[step])

        # Read the executor info JSON
        with open(executor.executor_info_path) as f:
            info = json.load(f)

        step_info = info["steps"][0]
        assert "duration_seconds" in step_info
        assert step_info["duration_seconds"] >= 0.5
        assert "execution_start_time" in step_info
        assert "execution_end_time" in step_info
```

**Validate**:
```bash
uv run pytest tests/test_executor.py::test_timing_in_executor_json -v
```

#### Step 4.3: Test timing on failed steps

**File**: `tests/test_executor.py` (add new test)

```python
def test_timing_on_failure():
    """Verify timing is captured even when step fails."""
    import time

    def failing_fn(config: MyConfig | None):
        time.sleep(1)
        raise ValueError("Intentional failure")

    step = ExecutorStep(name="failing", fn=failing_fn, config=None)

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)

        with pytest.raises(ValueError, match="Intentional failure"):
            executor.run(steps=[step])

        # Timing should still be captured
        status_path = get_status_path(executor.output_paths[step])
        events = read_events(status_path)

        # Find the FAILED event
        failed_event = next(e for e in events if e["status"] == STATUS_FAILED)
        assert "execution_start_time" in failed_event
        assert "execution_end_time" in failed_event

        duration = failed_event["execution_end_time"] - failed_event["execution_start_time"]
        assert duration >= 1.0  # At least the sleep time
```

**Validate**:
```bash
uv run pytest tests/test_executor.py::test_timing_on_failure -v
```

### Phase 5: Update Existing Tests

**Goal**: Ensure existing tests work with new timing fields.

#### Step 5.1: Audit tests for ExecutorStepInfo usage

```bash
# Find all tests that check ExecutorStepInfo fields
grep -r "ExecutorStepInfo" tests/
grep -r "step_info\[" tests/
```

Most tests shouldn't break because timing fields are optional and default to None.

#### Step 5.2: Update tests that do strict equality checks

**File**: `tests/test_executor.py:129` (example)

Before:
```python
def check_info(step_info: dict, step: ExecutorStep):
    assert step_info["name"] == step.name
    assert step_info["output_path"] == executor.output_paths[step]
    assert step_info["config"] == asdict_optional(executor.configs[step])
    assert step_info["version"] == executor.versions[step]
```

After:
```python
def check_info(step_info: dict, step: ExecutorStep):
    assert step_info["name"] == step.name
    assert step_info["output_path"] == executor.output_paths[step]
    assert step_info["config"] == asdict_optional(executor.configs[step])
    assert step_info["version"] == executor.versions[step]
    # Timing fields are optional
    assert "duration_seconds" in step_info  # Field exists but may be None
```

**Validate**:
```bash
uv run pytest tests/test_executor.py -v
uv run pytest tests/test_dry_run.py -v
# All tests should pass
```

## Benefits

1. **Observability**: Developers can identify slow steps without manual instrumentation
2. **Optimization**: Pipeline authors can focus optimization efforts on actual bottlenecks
3. **Debugging**: Timing data helps diagnose Ray scheduling issues vs. actual execution slowness
4. **Backwards compatible**: No breaking changes to existing experiments
5. **Low overhead**: <1% performance impact from wrapper function
6. **Integrated**: Uses existing status system, no new infrastructure needed

## Trade-offs

**Pros**:
- Minimal code changes (mainly in executor.py and status tracking)
- Works with existing Ray infrastructure
- Timing survives across executor reruns (persisted in status files)

**Cons**:
- Timing accuracy limited to Python's `time.time()` precision (~microseconds on modern systems)
- Doesn't capture time spent in Ray scheduling queue before worker picks up task (though `total_duration_seconds` includes this)
- Wrapper adds small overhead, though negligible for typical steps (>1 second)

## Future Work

After this feature is stable, we could add:

1. **Pipeline visualization**: Generate Gantt charts from timing data showing parallel execution
2. **Distributed tracing**: Integrate with Ray's tracing to show cross-worker dependencies
3. **Resource metrics**: Extend timing to include CPU/memory usage per step
4. **Alerting**: Warn if steps take >X% longer than historical average
5. **Flame graphs**: Visualize which steps dominate pipeline execution time

---

# How to Use This Recipe

## For Agents

When writing a design doc:

1. **Pick a concrete example** - Don't use placeholders like "[describe problem]". Show actual code.

2. **Reference real files with line numbers** - Example: `src/marin/execution/executor.py:664`, not "the executor file"

3. **Show before/after** - Include current implementation with actual code snippets, then proposed changes

4. **Be specific in implementation plan** - Each step should say:
   - Exact file to modify
   - Code snippet showing the change
   - Validation command (specific pytest invocation)

5. **Include Benefits/Trade-offs** - Explicitly state what's gained and what's lost

6. **Follow AGENTS.md principles**:
   - State backwards compatibility upfront
   - No deprecation paths unless user requests
   - Update all usages directly
   - Use early-exit patterns, top-level functions, functional style

## Structure Checklist

- [ ] **Background & Motivation**: Concrete problem with example scenario
- [ ] **Goals/Non-Goals**: Specific, measurable objectives
- [ ] **Current Implementation**: Real code from codebase showing current approach
- [ ] **Proposed Design**: New dataclasses/functions with actual code
- [ ] **Key Principles**: Design decisions with rationale
- [ ] **Implementation Plan**: Phased approach, each phase has:
  - [ ] Goal statement
  - [ ] Specific file paths
  - [ ] Code snippets (before/after)
  - [ ] Validation steps (exact commands)
- [ ] **Benefits**: What value does this provide?
- [ ] **Trade-offs**: What are the downsides?
- [ ] **Future Work**: What's deliberately left out?

## Key Differences from Abstract Templates

❌ **Bad** (abstract):
```markdown
### Step 1: Update the configuration
- Add new fields to the config class
- Update the constructor
- Validate the changes
```

✅ **Good** (concrete):
```markdown
### Step 1.1: Add timing fields to StatusEvent

**File**: `src/marin/execution/executor_step_status.py:15`

```python
@dataclass
class StatusEvent:
    timestamp: str
    status: str
    message: str | None = None
    ray_task_id: str | None = None
    # Add:
    execution_start_time: float | None = None
    execution_end_time: float | None = None
```

**Validate**:
```bash
uv run pytest tests/test_executor.py::test_executor -v
# Should pass - new fields are optional
```
```

Notice the good example:
- Names the exact file and line number
- Shows the actual code change
- Explains why (optional for backwards compat)
- Provides specific test command with expected result

## See Also

- `AGENTS.md` - Coding guidelines (no deprecation, update all usages)
- `docs/explanations/executor.md` - Executor framework concepts
- `docs/design/rl-unified-interface.md` - Another example of detailed design doc
- `docs/design/rl-rollout-metadata-migration.md` - Example showing data structure migration
