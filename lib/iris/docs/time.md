# Time Handling Design for Iris

## Motivation

Time handling in distributed systems is error-prone. Common issues include:

- **Unit confusion**: Mixing seconds, milliseconds, and nanoseconds leads to bugs (e.g., `timeout=5` could mean 5ms or 5s)
- **Monotonic vs wall-clock confusion**: Using wall-clock time for timeouts breaks when system clocks adjust
- **Type safety**: Raw `int` or `float` time values can be accidentally used as durations when they're timestamps, or vice versa
- **Proto conversion**: Manual conversion between Python time representations and proto fields is error-prone and inconsistent

This design introduces type-safe time primitives that:
1. Make time semantics explicit in code
2. Prevent mixing incompatible time concepts
3. Use appropriate clock sources (monotonic for timeouts, wall-clock for timestamps)
4. Provide consistent proto serialization

## Current State

**Already implemented** (as of PR #2567):
- Core time types: `Timestamp`, `Duration`, `Deadline`, `Timer`, `RateLimiter`, `ExponentialBackoff`
- Proto messages: `iris.common.Timestamp`, `iris.common.Duration`
- Partial migration in: `state.py`, `managed_vm.py`, `ssh.py`, `task_attempt.py`, `worker.py`

**Still needs migration**:
- Remaining files using `now_ms()`, `time.time()`, or raw time values
- Proto fields still using raw `int64 timestamp_ms` or `int32 timeout_seconds`
- Helper functions and configuration objects that accept raw time values
- SSH timeout handling, VM lifecycle timeouts, retry logic

## Design Principles

### 1. Type Safety First

**Never** use raw numeric types for time in public APIs:

```python
# ❌ BAD: Ambiguous units, easy to mix up
def connect(timeout: int) -> None: ...
def created_at() -> int: ...

# ✅ GOOD: Clear semantics, type-safe
def connect(timeout: Duration) -> None: ...
def created_at() -> Timestamp: ...
```

### 2. Consistent Proto Representation

All time values serialize to proto using **milliseconds**:
- Timestamps: milliseconds since Unix epoch (`int64 epoch_ms`)
- Durations: milliseconds interval (`int64 milliseconds`)

This provides:
- Sufficient precision (1ms) for distributed systems
- Consistent units across the codebase
- Compatibility with common time representations

### 3. Appropriate Clock Sources

- **Wall-clock time** (`time.time()`): Use for absolute timestamps (logging, persistence, user display)
- **Monotonic time** (`time.monotonic()`): Use for relative timing (timeouts, deadlines, performance measurement)

Types enforce correct usage:
- `Timestamp`: Wall-clock (epoch-based)
- `Deadline`, `Timer`: Monotonic (immune to clock adjustments)
- `Duration`: Clock-agnostic (pure interval)

### 4. Class Methods for Construction

Prefer class methods over constructor parameters for clarity:

```python
# ✅ GOOD: Intent is clear at call site
timeout = Duration.from_seconds(30.0)
deadline = Deadline.from_seconds(5.0)
created = Timestamp.now()

# ❌ BAD: Less clear, easy to forget units
timeout = Duration(30000)  # is this ms? seconds?
deadline = Deadline(5.0)   # what unit?
```

### 5. No Raw Proto Wrappers

Don't add wrapper methods to maintain old APIs:

```python
# ❌ BAD: Creates parallel APIs, delays real migration
class VM:
    def created_at_ms(self) -> int:
        return self._created_at.epoch_ms()

    def created_at(self) -> Timestamp:
        return self._created_at

# ✅ GOOD: Single clear API
class VM:
    def created_at(self) -> Timestamp:
        return self._created_at
```

When migrating, **update all call sites** to use the typed API.

## Core Types

### Timestamp

Represents a **point in wall-clock time** (milliseconds since Unix epoch).

**Use for**: Event timestamps, creation times, logging, user-facing dates.

```python
from iris.time_utils import Timestamp

# Construction
now = Timestamp.now()
from_ms = Timestamp.from_ms(1234567890123)
from_sec = Timestamp.from_seconds(1234567890.123)
from_proto = Timestamp.from_proto(proto_msg.timestamp)

# Conversion
epoch_ms: int = ts.epoch_ms()
epoch_sec: float = ts.epoch_seconds()
iso_string: str = ts.as_formatted_date()  # "2009-02-13T23:31:30.123000+00:00"
proto_value: int = ts.to_proto()

# Arithmetic
age_ms: int = ts.age_ms()
future: Timestamp = ts.add_ms(5000)
future: Timestamp = ts.add(Duration.from_seconds(5))

# Comparison
if ts1.before(ts2): ...
if ts1.after(ts2): ...
if ts1 == ts2: ...
if ts1 < ts2: ...
```

### Duration

Represents a **time interval** (milliseconds).

**Use for**: Configuration timeouts, retry intervals, age calculations.

```python
from iris.time_utils import Duration

# Construction
timeout = Duration.from_seconds(30.0)
interval = Duration.from_ms(5000)
delay = Duration.from_minutes(5)
long_timeout = Duration.from_hours(2)

# Conversion
ms: int = dur.to_ms()
sec: float = dur.to_seconds()

# Arithmetic
total = dur1 + dur2
doubled = dur * 2.0
halved = dur * 0.5

# Comparison
if dur1 < dur2: ...
if dur == Duration.from_seconds(10): ...
```

### Deadline

Represents a **point in monotonic time** for timeout checking.

**Use for**: Operation timeouts, retry loops, polling limits.

```python
from iris.time_utils import Deadline

# Construction
deadline = Deadline.from_seconds(30.0)
deadline = Deadline.from_ms(5000)

# Checking
if deadline.expired():
    raise TimeoutError("Operation timed out")

deadline.raise_if_expired("Connection timeout")

# Remaining time
remaining_sec: float = deadline.remaining_seconds()
remaining_ms: int = deadline.remaining_ms()

# Usage in loops
deadline = Deadline.from_seconds(10.0)
while not deadline.expired():
    if try_operation():
        break
    time.sleep(0.1)
deadline.raise_if_expired("Operation failed")
```

**Important**: Deadline uses monotonic time internally, so it's immune to system clock adjustments.

### Timer

Measures **elapsed time** using monotonic clock.

**Use for**: Performance measurement, timing operations.

```python
from iris.time_utils import Timer

timer = Timer()
do_work()
elapsed_ms = timer.elapsed_ms()
elapsed_sec = timer.elapsed_seconds()

# Reset for reuse
timer.reset()
```

### RateLimiter

Ensures operations don't run more frequently than a specified interval.

**Use for**: Logging throttling, API rate limiting, polling frequency.

```python
from iris.time_utils import RateLimiter

limiter = RateLimiter(interval_seconds=1.0)

while True:
    if limiter.should_run():
        expensive_logging()
    time.sleep(0.1)
```

### ExponentialBackoff

Implements exponential backoff with jitter for retry loops.

**Use for**: Retry logic, polling with backoff, connection attempts.

```python
from iris.time_utils import ExponentialBackoff

# Configuration
backoff = ExponentialBackoff(
    initial=0.1,      # Start at 100ms
    maximum=5.0,      # Cap at 5 seconds
    factor=2.0,       # Double each time
    jitter=0.1,       # ±10% randomness
)

# Manual control
while not success:
    try_operation()
    time.sleep(backoff.next_interval())

# Wait for condition
success = backoff.wait_until(
    condition=lambda: server.is_ready(),
    timeout=30.0
)

# Wait for condition or raise
backoff.wait_until_or_raise(
    condition=lambda: connection.established(),
    timeout=60.0,
    error_message="Connection failed"
)
```

**Update plan**: Migrate `ExponentialBackoff` to accept `Duration` for parameters:

```python
# Future API (after migration)
backoff = ExponentialBackoff(
    initial=Duration.from_ms(100),
    maximum=Duration.from_seconds(5.0),
    factor=2.0,
    jitter=0.1,
)

# Or keep float seconds for simplicity since it's always relative
# Decision: Keep float seconds for ergonomics (they're always intervals, not absolute)
```

## Proto Integration

### Proto Message Definitions

```protobuf
// In common.proto
message Timestamp {
  int64 epoch_ms = 1;
}

message Duration {
  int64 milliseconds = 1;
}
```

### Proto <-> Python Conversion

**Timestamp**:
```python
# Proto -> Python
created_at = Timestamp.from_proto(proto_msg.created_at.epoch_ms)

# Python -> Proto
proto_msg.created_at.epoch_ms = created_at.to_proto()

# Or construct proto message
from iris.rpc.common_pb2 import Timestamp as ProtoTimestamp
proto_ts = ProtoTimestamp(epoch_ms=created_at.to_proto())
```

**Duration**:
```python
# Proto -> Python
timeout = Duration.from_ms(proto_msg.timeout.milliseconds)

# Python -> Proto
proto_msg.timeout.milliseconds = timeout.to_ms()

# Or construct proto message
from iris.rpc.common_pb2 import Duration as ProtoDuration
proto_dur = ProtoDuration(milliseconds=timeout.to_ms())
```

### Migration: Raw Proto Fields

Many proto files still use raw time fields. Migrate these incrementally:

**Before**:
```protobuf
message WorkerInfo {
  int64 registered_at_ms = 1;
  int32 heartbeat_timeout_seconds = 2;
}
```

**After**:
```protobuf
import "common.proto";

message WorkerInfo {
  iris.common.Timestamp registered_at = 1;
  iris.common.Duration heartbeat_timeout = 2;
}
```

**Python code update**:
```python
# Before
registered_ms = proto_msg.registered_at_ms
timeout_sec = proto_msg.heartbeat_timeout_seconds

# After
registered = Timestamp.from_proto(proto_msg.registered_at.epoch_ms)
timeout = Duration.from_ms(proto_msg.heartbeat_timeout.milliseconds)
```

## Migration Strategy

### Phase 1: Core Infrastructure ✅ (PR #2567)
- [x] Implement `Timestamp`, `Duration`, `Deadline`, `Timer`, `RateLimiter`, `ExponentialBackoff`
- [x] Add proto messages `iris.common.Timestamp`, `iris.common.Duration`
- [x] Migrate core modules: `state.py`, `managed_vm.py`, `ssh.py`, `task_attempt.py`, `worker.py`

### Phase 2: API Surface Migration ✅
- [x] Migrate public APIs to accept typed time parameters
- [x] Update `SshConnection` protocol: `timeout: int` → `timeout: Duration`
- [x] Update VM lifecycle: boot/init timeouts → `Duration`
- [x] Update autoscaler: backoff timestamps → `Timestamp`, intervals → `Duration`
- [x] Update scheduler: deadlines → `Deadline`, timestamps → `Timestamp`

### Phase 3: Proto Field Migration ✅
- [x] Audit all proto files for raw time fields
- [x] Migrate to `iris.common.Timestamp` and `iris.common.Duration`
- [x] Update serialization/deserialization code
- [x] Regenerate proto files: `uv run scripts/generate_protos.py`

### Phase 4: Helper Function Migration ✅
- [x] Configuration classes: accept `Duration` instead of seconds/milliseconds
- [x] Retry/backoff utilities: return and accept `Duration`
- [x] Remove `now_ms()` from public APIs (keep internal for `Timestamp.now()` implementation)

### Phase 5: Testing and Validation ✅
- [x] Add integration tests for time handling
- [x] Verify monotonic vs wall-clock usage is correct
- [x] Check proto serialization round-trips correctly
- [x] Performance testing (no regression from type overhead)

## Usage Patterns

### Pattern: SSH Connections

```python
# Current (Phase 1)
class SshConnection(Protocol):
    def run(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        ...

# Target (Phase 2)
class SshConnection(Protocol):
    def run(
        self,
        command: str,
        timeout: Duration = Duration.from_seconds(30)
    ) -> subprocess.CompletedProcess:
        ...

# Implementation
def run(self, command: str, timeout: Duration = Duration.from_seconds(30)) -> subprocess.CompletedProcess:
    deadline = Deadline.from_seconds(timeout.to_seconds())
    # Use deadline.remaining_seconds() when calling subprocess
    result = subprocess.run(
        cmd_args,
        timeout=deadline.remaining_seconds(),
        ...
    )
    return result
```

### Pattern: VM Lifecycle Timeouts

```python
@dataclass
class TimeoutConfig:
    # Current: raw int seconds
    boot_timeout_seconds: int = 300
    init_timeout_seconds: int = 600

# Target
@dataclass
class TimeoutConfig:
    boot_timeout: Duration = Duration.from_minutes(5)
    init_timeout: Duration = Duration.from_minutes(10)

    @classmethod
    def from_proto(cls, proto: ProtoTimeoutConfig) -> TimeoutConfig:
        return cls(
            boot_timeout=Duration.from_ms(proto.boot_timeout.milliseconds),
            init_timeout=Duration.from_ms(proto.init_timeout.milliseconds),
        )
```

### Pattern: Retry with Backoff and Deadline

```python
def connect_with_retry(
    address: str,
    timeout: Duration,
) -> Connection:
    """Connect with exponential backoff until timeout."""
    backoff = ExponentialBackoff(
        initial=0.1,
        maximum=2.0,
    )
    deadline = Deadline.from_seconds(timeout.to_seconds())

    while not deadline.expired():
        try:
            return Connection(address)
        except ConnectionError:
            if deadline.expired():
                break
            interval = backoff.next_interval()
            # Don't sleep past deadline
            sleep_time = min(interval, deadline.remaining_seconds())
            time.sleep(sleep_time)

    raise TimeoutError(f"Failed to connect to {address}")
```

### Pattern: Event Timestamps

```python
@dataclass
class TaskAttempt:
    # Current: raw int
    started_at_ms: int
    finished_at_ms: int | None

# Target
@dataclass
class TaskAttempt:
    started_at: Timestamp
    finished_at: Timestamp | None

    @classmethod
    def from_proto(cls, proto: ProtoTaskAttempt) -> TaskAttempt:
        return cls(
            started_at=Timestamp.from_proto(proto.started_at.epoch_ms),
            finished_at=(
                Timestamp.from_proto(proto.finished_at.epoch_ms)
                if proto.HasField("finished_at")
                else None
            ),
        )

    def to_proto(self) -> ProtoTaskAttempt:
        proto = ProtoTaskAttempt()
        proto.started_at.epoch_ms = self.started_at.to_proto()
        if self.finished_at is not None:
            proto.finished_at.epoch_ms = self.finished_at.to_proto()
        return proto

    def duration(self) -> Duration | None:
        """Calculate how long the attempt ran."""
        if self.finished_at is None:
            return None
        elapsed_ms = self.finished_at.epoch_ms() - self.started_at.epoch_ms()
        return Duration.from_ms(elapsed_ms)
```

### Pattern: Heartbeat Tracking

```python
@dataclass
class WorkerState:
    last_heartbeat: Timestamp
    heartbeat_timeout: Duration

    def is_heartbeat_expired(self) -> bool:
        """Check if worker has missed its heartbeat deadline."""
        age = self.last_heartbeat.age_ms()
        return age > self.heartbeat_timeout.to_ms()

    def heartbeat_deadline(self) -> Timestamp:
        """When will the heartbeat expire?"""
        return self.last_heartbeat.add(self.heartbeat_timeout)
```

### Pattern: Rate-Limited Logging

```python
class Controller:
    def __init__(self):
        self._status_limiter = RateLimiter(interval_seconds=5.0)

    def periodic_status(self):
        if self._status_limiter.should_run():
            logger.info(f"Status: {self.get_status()}")
```

## Anti-Patterns

### ❌ Don't: Mix Raw and Typed Time

```python
# BAD: Inconsistent API
class VM:
    def created_at(self) -> Timestamp: ...
    def boot_time_ms(self) -> int: ...  # Should return Duration
    def timeout_seconds(self) -> float: ...  # Should return Duration
```

### ❌ Don't: Manual Proto Conversion

```python
# BAD: Error-prone, units unclear
proto_msg.timeout = int(timeout * 1000)

# GOOD: Use Duration
proto_msg.timeout.milliseconds = timeout.to_ms()
```

### ❌ Don't: Use Wall-Clock for Timeouts

```python
# BAD: Breaks if system clock changes
start = time.time()
while time.time() - start < timeout:
    ...

# GOOD: Use Deadline with monotonic time
deadline = Deadline.from_seconds(timeout)
while not deadline.expired():
    ...
```

### ❌ Don't: Store Deadline/Timer in Dataclasses

```python
# BAD: Deadline is stateful and monotonic, doesn't serialize
@dataclass
class Task:
    timeout: Deadline  # Can't persist to proto!

# GOOD: Store Duration, create Deadline when needed
@dataclass
class Task:
    timeout: Duration

    def create_deadline(self) -> Deadline:
        return Deadline.from_seconds(self.timeout.to_seconds())
```

### ❌ Don't: Implicit Unit Conversions

```python
# BAD: Silent conversion hides intent
def set_timeout(self, seconds: float):
    self._timeout_ms = int(seconds * 1000)

# GOOD: Explicit types
def set_timeout(self, timeout: Duration):
    self._timeout = timeout
```

## Testing Guidelines

### Test Time-Sensitive Code

Use `time.sleep()` sparingly in tests. Prefer short timeouts and verify behavior:

```python
def test_deadline_expires():
    deadline = Deadline.from_ms(50)
    time.sleep(0.1)
    assert deadline.expired()

def test_deadline_not_expired():
    deadline = Deadline.from_seconds(10.0)
    assert not deadline.expired()
```

### Test Proto Round-Trips

```python
def test_timestamp_proto_roundtrip():
    original = Timestamp.now()
    proto_value = original.to_proto()
    restored = Timestamp.from_proto(proto_value)
    assert original == restored
```

### Don't Test Implementation Details

```python
# ❌ BAD: Tests internal representation
def test_duration_internal_ms():
    dur = Duration.from_seconds(1.5)
    assert dur._ms == 1500  # Brittle!

# ✅ GOOD: Tests public behavior
def test_duration_conversion():
    dur = Duration.from_seconds(1.5)
    assert dur.to_ms() == 1500
    assert dur.to_seconds() == 1.5
```

## Summary

**Key principles**:
1. Use typed time classes (`Timestamp`, `Duration`, `Deadline`) everywhere
2. Never expose raw time values in public APIs
3. Use class methods for construction (`Duration.from_seconds(5)`)
4. Migrate consistently—no backward-compatibility wrappers
5. Use monotonic time (`Deadline`, `Timer`) for timeouts and measurements
6. Use wall-clock time (`Timestamp`) for event timestamps and logging
7. Serialize everything as milliseconds in proto

**Migration order**:
1. ✅ Core types and partial migration (PR #2567)
2. ✅ Public APIs (protocols, signatures)
3. ✅ Proto field definitions
4. ✅ Helper functions and config
5. ✅ Testing and validation

This design ensures time handling is **type-safe**, **semantically clear**, and **correct** across the Iris codebase.

## Migration Complete

### Completion Status

The migration to typed time primitives was completed in January 2026 across multiple PRs:

- **Phase 1** (PR #2567): Core infrastructure and initial migrations
- **Phase 2**: API surface migration across SSH, VM lifecycle, autoscaler, scheduler
- **Phase 3**: Proto field migration to `iris.common.Timestamp` and `iris.common.Duration`
- **Phase 4**: Helper function and configuration migration
- **Phase 5**: Testing and validation (complete)

### Files Modified

The following files were migrated to use typed time primitives:

**Core infrastructure**:
- `lib/iris/src/iris/time_utils.py` - New type-safe time primitives
- `lib/iris/tests/test_time_utils.py` - Comprehensive test coverage

**Proto definitions**:
- `lib/iris/src/iris/rpc/common.proto` - Added `Timestamp` and `Duration` messages
- `lib/iris/src/iris/rpc/cluster.proto` - Migrated to typed time fields (`timeout`, `scheduling_timeout`, `LogEntry.timestamp`, `TransactionAction.timestamp`)
- `lib/iris/src/iris/rpc/vm.proto` - Migrated to typed time fields
- `lib/iris/src/iris/rpc/errors.proto` - Migrated `ErrorDetails.timestamp` to `iris.common.Timestamp`
- Generated files: `*_pb2.py`, `*_pb2.pyi` for all proto files

**Application code**:
- `lib/iris/src/iris/cluster/controller/state.py` - Controller state timestamps
- `lib/iris/src/iris/cluster/controller/service.py` - Transaction action timestamp serialization
- `lib/iris/src/iris/cluster/vm/managed_vm.py` - VM lifecycle timeouts and timestamps
- `lib/iris/src/iris/cluster/vm/ssh.py` - SSH connection timeouts
- `lib/iris/src/iris/cluster/worker/task_attempt.py` - Task execution timestamps
- `lib/iris/src/iris/cluster/worker/worker.py` - Worker heartbeat tracking
- `lib/iris/src/iris/cluster/worker/service.py` - Log entry timestamp filtering
- `lib/iris/src/iris/client/client.py` - LogEntry dataclass migrated to use `Timestamp`
- `lib/iris/src/iris/rpc/errors.py` - Error detail timestamp construction

### Key Changes

**Removed patterns**:
- Raw `int` timestamps (e.g., `created_at_ms: int`)
- Raw `float` or `int` timeouts (e.g., `timeout: int = 30`)
- Manual millisecond conversions (e.g., `int(timeout * 1000)`)
- Wall-clock time for timeouts (e.g., `time.time()` in loops)

**Replaced with**:
- `Timestamp` for all event times (creation, completion, heartbeat)
- `Duration` for all timeout configurations and intervals
- `Deadline` for all timeout checking in loops
- `Timer` for performance measurement
- `RateLimiter` for throttling operations

### Deviations from Original Plan

**ExponentialBackoff parameters**: The original plan suggested migrating `ExponentialBackoff` to accept `Duration` parameters. After evaluation, we decided to keep `float` seconds for ergonomic reasons:
- Backoff parameters are always relative intervals, never absolute times
- Float seconds are more natural for backoff configuration (e.g., `initial=0.1`)
- No ambiguity since these can never be confused with timestamps

**Migration scope**: The migration was more comprehensive than initially planned:
- All proto definitions were migrated (not just selected files)
- All time-related configurations were updated consistently
- Rate-limited logging was migrated to use `RateLimiter` utility

### Lessons Learned

1. **Proto migrations are disruptive**: Changing proto field types requires updating all serialization code at once. Plan for coordinated updates across client and server code.

2. **Type safety catches bugs early**: The migration revealed several places where timeout units were ambiguous or incorrectly converted. Strong typing prevented these from becoming runtime errors.

3. **Monotonic time matters**: Several timeout implementations were using wall-clock time (`time.time()`), which would break if the system clock adjusted. Switching to `Deadline` (monotonic) fixed these latent bugs.

4. **Default parameters help migration**: Adding default timeout values (e.g., `Duration.from_seconds(30)`) made migration smoother by reducing the need to update all call sites immediately.

5. **Documentation and examples are critical**: The comprehensive examples in this document helped maintain consistency across the migration and provided clear guidance for future code.

### Completion Date

Migration completed: January 2026

Performance validation: Complete (January 2026)
