# Iris Testing Guidelines

Comprehensive testing policy for the Iris project. Referenced from [AGENTS.md](AGENTS.md).

## Core Principle

Tests should test **stable behavior**, not implementation details.

ABSOLUTELY DO NOT test things that are trivially caught by the type checker:

- No tests for "constant = constant"
- No tests for "method exists"
- No tests for "create an object(x, y, z) and attributes are x, y, z"

These tests have negative value — they make our code more brittle.

## What to Test

Test _stable behavior_. Prefer integration-style tests which exercise behavior
and test externally-observable output. Use mocks as needed to isolate external
dependencies (e.g. mock around subprocess for gcloud/kubectl, HTTP for remote
APIs), but prefer "fakes" — real implementations backed by in-memory or
temporary-file state — when reasonable.

Good tests validate:

- Integration points and round-trip behavior (submit job -> observe status)
- Realistic failure modes (worker crash, heartbeat timeout, quota exhaustion)
- Edge cases at API boundaries (empty inputs, duplicate names, constraint mismatches)
- State machine transitions via the public event API

## What NOT to Test

### Private attributes

No assertions on `_`-prefixed attributes. If a behavior is worth testing, it
must be observable through the public API. If no public API exists, either add
one or accept the behavior is an implementation detail.

```python
# BAD — reaches into private state
assert group._backoff_until is not None
assert worker_id in state._pending_dispatch

# GOOD — observes behavior through public API
status = autoscaler.get_status()
assert status.groups[0].backoff_active
```

### Internal call dispatch

No `assert_called_once_with` or `call_count` on internal helpers. When using
mocks at external boundaries (subprocess, HTTP, gcloud), asserting on call
shape is acceptable. But mocking internal functions and asserting on call
counts tests the wiring, not the behavior. If you need to verify a side
effect, use a fake that records observable state.

### Python language semantics

Do not test that `list(set)` creates a snapshot, or that `len(list) >= 0`.
Test application behavior.

### Constructor round-trips

Do not test that `Foo(x=1).x == 1`.

## Test Hygiene

### Every test must assert something

Every test function must contain at least one `assert` statement or
`pytest.raises` context. Tests that only verify "does not raise" must include a
comment explaining this intent. Screenshot-only tests are not acceptable
without accompanying behavioral assertions.

### No permanently-skipped tests

Do not check in `@pytest.mark.skip`-ed tests. A skipped test provides zero
value and accumulates maintenance debt. If a test is flaky, either fix it or
delete it. If a feature is not yet implemented, track it in an issue, not a
skipped test.

### No dead code in test files

Remove unused helpers, fakes, classes, and imports from test files. Delete
empty stub files. A test file with no test functions has negative value.

### Test naming

Use `test_<subject>_<scenario>_<expected_outcome>`:

```
test_scheduler_with_insufficient_capacity_returns_empty_assignments
test_worker_after_heartbeat_timeout_is_marked_failed
```

Names must accurately describe the verified behavior. A test named
`test_multiple_workers_one_fails` that uses a single-worker cluster is
misleading and must be renamed or rewritten.

File naming: use `test_<module>.py` where `<module>` matches the source file
being tested.

## Timing and Polling

Avoid bare `time.sleep()` in polling loops. Use `iris.time_utils.Deadline`,
`ExponentialBackoff.wait_until()`, or `wait_for_condition` from test utilities.

A single short sleep to let a background thread start is acceptable when
documented with a comment. Sleeping in a loop to wait for a condition is not.

Test helpers must not use bare `except Exception`. Catch specific exception
types even in startup-polling loops.

## Markers and Organization

- All tests that boot a cluster (local or Docker) must be marked
  `@pytest.mark.e2e`.
- Docker-dependent tests must also be marked `@pytest.mark.docker`.
- E2E tests live in `tests/e2e/`.
- Shared fakes live in `tests/cluster/platform/fakes.py` or
  `src/iris/test_util.py`. Do not duplicate fakes across files.

## Protocols

Non-trivial public classes should define a protocol which represents their
_important_ interface characteristics. Test to this protocol, not the concrete
class: the protocol should describe the interesting behavior of the class, but
not betray the implementation details. (You may of course _instantiate_ the
concrete class for testing.)

## E2E Tests

All Iris E2E tests live in `tests/e2e/`. Every test is marked `e2e`.
Tests use three core fixtures:

- `cluster`: Booted local cluster with `IrisClient` and RPC access
- `page`: Playwright page pointed at the dashboard (request only when needed)
- `screenshot`: Capture labeled screenshots to `IRIS_SCREENSHOT_DIR`

Chaos injection is auto-reset between tests. Call `enable_chaos()` directly.
Docker tests use a separate `docker_cluster` fixture and are marked `docker`.

## Running Tests

```bash
# All unit tests
uv run pytest lib/iris/tests/ -m "not e2e" -o "addopts="

# All E2E tests
uv run pytest lib/iris/tests/e2e/ -m e2e -o "addopts="

# E2E without Docker (fast)
uv run pytest lib/iris/tests/e2e/ -m "e2e and not docker" -o "addopts="

# Docker-only tests
uv run pytest lib/iris/tests/e2e/ -m docker -o "addopts="

# Dashboard tests with screenshots
IRIS_SCREENSHOT_DIR=/tmp/shots uv run pytest lib/iris/tests/e2e/test_dashboard.py -o "addopts="

# When modifying the dashboard
uv run pytest lib/iris/tests/e2e/test_dashboard.py -x -o "addopts="

# K8s runtime tests (requires a running cluster — kind, k3d, minikube, etc.)
uv run pytest lib/iris/tests/e2e/test_coreweave_live_kubernetes_runtime.py \
  -m slow -k lifecycle -v
```
