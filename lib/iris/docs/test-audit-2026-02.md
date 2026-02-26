# Iris Test Suite Audit — February 2026

An exhaustive review of every test file in `lib/iris/` against the testing
policy in `AGENTS.md`. Each test function was categorized as:

- **GOOD** — Tests stable behavior, integration points, or realistic failure modes
- **BAD** — Tests implementation details, is tautological, or tests things the type checker catches
- **UGLY** — Poor naming, missing assertions, permanently skipped, resource leaks, or misleading intent

---

## Executive Summary

| Metric | Count |
|--------|-------|
| Total test files reviewed | 60 |
| Total test functions reviewed | ~340 |
| GOOD | ~295 (87%) |
| BAD | ~25 (7%) |
| UGLY | ~20 (6%) |

The suite is **generally high quality**. Most tests exercise real behavior via
integration-style execution with fakes over mocks. The problems cluster around:

1. **Private-attribute access** (`_backoff_until`, `_pending_dispatch`, `_slices`, `_remote_exec`)
2. **Mock call-count assertions** instead of observable-behavior checks
3. **Permanently-skipped or assertion-free tests** that accumulate maintenance debt
4. **Misleading test names** that promise behavior the test does not verify
5. **Redundant tests** that duplicate coverage without adding signal

---

## Per-File Summary Tables

### `tests/actor/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_actor_e2e.py` | `test_basic_actor_call` | GOOD | |
| `test_actor_e2e.py` | `test_actor_exception_propagation` | GOOD | |
| `test_actor_pool.py` | `test_pool_round_robin` | GOOD | Assertion proves coverage, not even distribution |
| `test_actor_pool.py` | `test_pool_broadcast` | GOOD | |
| `test_actor_retry.py` | `FailingResolver` (class) | **REMOVE** | Defined but never used — dead code |
| `test_actor_retry.py` | `test_actor_client_retries_on_transient_rpc_error` | GOOD | |
| `test_actor_retry.py` | `test_actor_client_does_not_retry_on_application_error` | GOOD | |
| `test_actor_retry.py` | `test_actor_pool_retries_on_transient_rpc_error` | GOOD | |
| `test_actor_retry.py` | `test_actor_client_exhausts_retries` | GOOD | |
| `test_actor_retry.py` | `test_actor_client_clears_cache_on_final_retryable_failure` | GOOD | Excellent regression test |
| `test_resolver.py` | `test_client_with_resolver` | UGLY | Poor name; redundant with `test_basic_actor_call` |
| `test_resolver.py` | `test_gcs_resolver_finds_actors` | GOOD | |
| `test_resolver.py` | `test_gcs_resolver_ignores_non_running` | GOOD | |
| `test_resolver.py` | `test_gcs_resolver_multiple_instances` | GOOD | |
| `test_resolver.py` | `test_gcs_resolver_no_matching_actor` | GOOD | |
| `test_resolver.py` | `test_gcs_resolver_missing_internal_ip` | GOOD | |

### `tests/cli/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_image_tag_parsing.py` | `test_parse_ghcr_tag` | GOOD | |
| `test_image_tag_parsing.py` | `test_parse_ghcr_tag_returns_none_for_non_ghcr` | GOOD | |
| `test_build_config_regions.py` | `test_build_cluster_images_pushes_worker_controller_and_task_to_ghcr` | BAD | Tests internal call dispatch with mock `assert_called`; tautological dict assertion; dead zone-config setup |
| `test_local_cluster.py` | `test_cli_local_cluster_e2e` | GOOD | Missing `@pytest.mark.e2e`; should live in `tests/e2e/` |

### `tests/client/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_worker_pool.py` | `test_submit_executes_task` | GOOD | |
| `test_worker_pool.py` | `test_map_executes_tasks` | GOOD | |
| `test_worker_pool.py` | `test_exception_propagates_to_caller` | GOOD | |
| `test_worker_pool.py` | `test_shutdown_prevents_new_submissions` | UGLY | Direct `__enter__()` call; resource leak; no `__exit__` cleanup |
| `test_worker_pool.py` | `test_multiple_sequential_tasks` | BAD | Redundant with existing tests; adds no distinct coverage |

### `tests/cluster/` (root)

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_attempt_logs.py` | `test_multiple_attempts_preserve_logs` | GOOD | |
| `test_attempt_logs.py` | `test_superseding_attempt_logs_info` | UGLY | Name promises log content check; no such assertion exists |
| `test_attempt_logs.py` | `test_attempt_specific_log_fetch` | GOOD | |
| `test_attempt_logs.py` | `test_task_status_shows_attempts` | GOOD | |
| `test_client.py` | All 7 tests | GOOD | Solid integration tests |
| `test_manager.py` | All 3 tests | GOOD | Proper use of fakes |
| `test_env_propagation.py` | `test_child_job_inherits_parent_env` | UGLY | Never asserts env vars are inherited; only checks job name |
| `test_env_propagation.py` | `test_env_propagates_through_job_chain` | GOOD | Best test in the file; full E2E |
| `test_env_propagation.py` | Other 5 tests | GOOD | But 4 use brittle `_cluster_client` monkey-patching |
| `test_pickle_version_mismatch.py` | All 3 tests | GOOD | Proper regression tests |
| `test_types.py` | `test_entrypoint_command` | BAD | "Constructor args == attributes" |
| `test_types.py` | `test_entrypoint_callable_has_workdir_files` | BAD | Tests internal filenames (`_callable.pkl`) |
| `test_types.py` | All other ~30 tests | GOOD | Especially JobName and constraint tests |

### `tests/cluster/runtime/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_kubernetes_runtime.py` | All 12 tests | GOOD | Right level of subprocess mocking for K8s manifests |

### `tests/cluster/controller/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_autoscaler.py` | ~35 tests (waterfall routing, quota, scaling) | GOOD | Excellent coverage |
| `test_autoscaler.py` | `test_get_init_log_returns_bootstrap_output` | BAD | Calls private `_register_slice_workers` |
| `test_autoscaler.py` | `test_execute_records_failure_on_scale_up_error` | BAD | Asserts on private `_backoff_until` |
| `test_autoscaler.py` | 2 cascade E2E tests | UGLY | Use `time.sleep(0.1)` — flaky on slow CI |
| `test_bundle_store.py` | All 3 tests | GOOD | |
| `test_dashboard.py` | ~20 tests | GOOD | Solid RPC-level integration |
| `test_dashboard.py` | `test_get_autoscaler_status_returns_status_when_enabled` | BAD | Tests mock passthrough, not logic |
| `test_dashboard.py` | `test_get_autoscaler_status_includes_slice_details` | BAD | Same: tests mock serialization |
| `test_heartbeat.py` | `test_worker_heartbeat_expired_check` | GOOD | |
| `test_job.py` | All 9 tests | GOOD | |
| `test_pending_diagnostics.py` | All 3 tests | GOOD | |
| `test_scheduler.py` | ~40 tests | GOOD | Very comprehensive |
| `test_scheduler.py` | `test_scheduler_detects_timed_out_tasks` | BAD | Re-implements controller logic inline |
| `test_scheduler.py` | `test_scheduler_no_timeout_when_zero` | BAD | Name misleading; tests wrong thing |
| `test_scheduler.py` | `test_scheduler_reports_coscheduling_capacity_details` | UGLY | Weak substring assertions |
| `test_service.py` | All 15 tests | GOOD | |
| `test_state.py` | ~45 tests | GOOD | Excellent event-driven tests |
| `test_state.py` | `test_fail_heartbeat_clears_dispatch_when_worker_fails` | BAD | Accesses `state._pending_dispatch` |
| `test_state.py` | `test_fail_heartbeat_requeues_dispatch_when_worker_healthy` | BAD | Accesses `state._pending_dispatch` |
| `test_state.py` | `test_running_tasks_safe_iteration_prevents_race` | BAD | Tests Python `set` semantics, not app behavior |
| `test_state.py` | `test_stale_attempt_error_log_for_non_terminal` | BAD | Calls `task.create_attempt()` directly |
| `test_state.py` | 3 coscheduled failure tests | UGLY | Assert on `txn.tasks_to_kill` (internal) — the state assertions are fine |
| `test_vm_lifecycle.py` | All 8 tests | GOOD | |

### `tests/cluster/platform/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_bootstrap.py` | 12 pure-function tests | GOOD | |
| `test_bootstrap.py` | `test_build_controller_bootstrap_script_from_config_rewrites_ghcr_to_ar` | BAD | Uses `MagicMock.assert_called_once_with` |
| `test_bootstrap.py` | 3 `test_gcp_platform_resolve_image_*` | BAD | `GcpPlatform.__new__` + private `_project_id` bypass |
| `test_config.py` | All ~45 tests | GOOD | |
| `test_platform.py` | ~28 tests | GOOD | Excellent parameterized protocol tests |
| `test_platform.py` | `test_gcp_reload_uses_full_stop_then_start` | BAD | Tests internal delegation via mock |
| `test_platform.py` | `test_gcp_create_vm_slice_mode_with_long_prefix_uses_valid_slice_id` | BAD | Accesses `_remote_exec` private attr |
| `test_platform.py` | `test_gcp_list_all_slices_multi_zone` | UGLY | Duplicate assertions (copy-paste error) |
| `test_remote_exec.py` | Both tests | BAD | Test private `_build_cmd` method |
| `test_scaling_group.py` | ~40 tests | GOOD | |
| `test_scaling_group.py` | `test_scale_up_passes_tags_as_labels` | BAD | Inspects mock `call_args` |
| `test_scaling_group.py` | 3 `TestVerifySliceIdle` tests | BAD | Test private `_verify_slice_idle` |
| `test_scaling_group.py` | 2 `TestMarkSliceLockDiscipline` tests | UGLY | Access `_slices_lock` and `_slices` directly |
| `test_coreweave_platform.py` | ~40 tests | GOOD | Excellent FakeKubectl |
| `test_coreweave_platform.py` | `test_tunnel_parses_address` | BAD | Redundant address re-parsing |
| `test_coreweave_platform.py` | `test_worker_pod_has_s3_env_vars` | UGLY | Sets `platform._s3_enabled = True` directly |

### `tests/cluster/worker/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_builder.py` | (empty file) | **REMOVE** | Dead stub |
| `test_bundle_cache.py` | All 6 tests | GOOD | Excellent real-filesystem integration |
| `test_dashboard.py` | ~12 tests | GOOD | |
| `test_dashboard.py` | `test_task_detail_page_loads` | BAD | Only checks HTTP 200 + content-type |
| `test_dashboard.py` | `test_rpc_heartbeat_via_connect_client` | **REMOVE** | Permanently skipped + latent assertion bug |
| `test_env_probe.py` | All 11 tests | GOOD | Best file in the directory |
| `test_task_logging.py` | All 16 tests | GOOD | |
| `test_worker.py` | ~18 tests | GOOD | |
| `test_worker.py` | `test_list_tasks` | BAD | Only checks list length |
| `test_worker.py` | `test_health_check_rpc` | BAD | `healthy=True` hardcoded; `uptime >= 0` trivially true |
| `test_worker.py` | `test_fetch_logs_tail` | UGLY | `sleep(2)` + `len >= 0` trivially true |

### `tests/e2e/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_task_lifecycle.py` | All 7 tests | GOOD | |
| `test_vm_lifecycle.py` | All 3 tests | GOOD | |
| `test_worker_failures.py` | All 6 tests | GOOD | |
| `test_rpc_failures.py` | All 5 tests | GOOD | |
| `test_heartbeat.py` | 4 tests | GOOD | |
| `test_heartbeat.py` | `test_multiple_workers_one_fails` | UGLY | Name says multi-worker; uses single-worker cluster |
| `test_heartbeat.py` | `test_heartbeat_failure_with_pending_kills` | UGLY | Docstring claims "pending kills" but no kills issued; duplicate |
| `test_scheduling.py` | All 7 tests | GOOD | |
| `test_high_concurrency.py` | `test_128_tasks_concurrent_scheduling` | GOOD | Race condition regression test |
| `test_endpoints.py` | All 7 tests | GOOD | |
| `test_dashboard.py` | 5 tests | GOOD | |
| `test_dashboard.py` | `test_autoscaler_tab` | UGLY | Zero assertions; screenshot only |
| `test_dashboard.py` | `test_controller_logs` | UGLY | Near-no-op without Playwright |
| `test_profiling.py` | All 3 tests | GOOD | |
| `test_building_logs.py` | `test_build_logs_visible_during_building_state` | **REMOVE** | Permanently skipped as flaky |
| `test_multi_region.py` | All 3 tests | GOOD | |
| `test_gpu_worker_metadata.py` | `test_gpu_worker_metadata_visible_to_controller` | GOOD | |
| `test_autoscaler_dashboard.py` | All 5 tests | GOOD | |
| `test_docker.py` | Both tests | GOOD | |
| `test_coreweave_live_kubernetes_runtime.py` | All 4 tests | GOOD | |

### `tests/rpc/`

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_proto_utils.py` | All 5 tests | GOOD | Model of behavioral testing |
| `test_interceptors.py` | Both tests | GOOD | |
| `test_errors.py` | All 12 tests | GOOD | |

### `tests/` (root)

| File | Test | Verdict | Issue |
|------|------|---------|-------|
| `test_iris_run.py` | All 11 tests | GOOD | Good mix of unit and integration |
| `test_logging.py` | 4 tests | GOOD | |
| `test_logging.py` | `test_configure_logging_captures_records` | UGLY | Mutates `_configured` private; local import |
| `test_marin_fs.py` | All 18 tests | GOOD | Comprehensive fallback-path coverage |
| `test_time_utils.py` | All 14 tests | GOOD | Excellent edge cases |
| `test_utils.py` | All 7 tests | GOOD | |
| `test_demo_notebook_submit.py` | All 3 tests | GOOD | |

### Misc

| File | Verdict | Issue |
|------|---------|-------|
| `tests/e2e/benchmark_controller.py` | **MOVE** | Not a test file; belongs in `benchmarks/` or `scripts/` |
| `tests/e2e/conftest.py` `chronos` fixture | **REMOVE** | Defined but never used by any e2e test |

---

## Tests to Remove

These tests provide zero or negative value and should be deleted:

| Test | File | Reason |
|------|------|--------|
| `test_rpc_heartbeat_via_connect_client` | `worker/test_dashboard.py` | Permanently skipped; contains latent assertion bug (`"rpc-test-task"` vs actual task ID) |
| `test_build_logs_visible_during_building_state` | `e2e/test_building_logs.py` | Permanently skipped as flaky; overly complex |
| `FailingResolver` class | `actor/test_actor_retry.py` | Dead code — defined but never used |
| `test_builder.py` | `worker/test_builder.py` | Empty file — zero content |
| `test_running_tasks_safe_iteration_prevents_race` | `controller/test_state.py` | Tests Python `list(set)` semantics, not application behavior |
| `test_multiple_sequential_tasks` | `client/test_worker_pool.py` | Fully redundant with `test_submit_executes_task` + `test_map_executes_tasks` |
| `test_client_with_resolver` | `actor/test_resolver.py` | Redundant with `test_basic_actor_call`; misleading name |

---

## Tests to Rewrite

These tests have structural problems worth fixing:

| Test | File | Problem | Fix |
|------|------|---------|-----|
| `test_remote_exec.py` (both) | `platform/test_remote_exec.py` | Tests private `_build_cmd` | Replace with behavioral test via `run_command` + fake subprocess |
| 3 `TestVerifySliceIdle` tests | `platform/test_scaling_group.py` | Tests private `_verify_slice_idle` | Rewrite via `scale_down_if_idle()` or `is_slice_eligible_for_scaledown()` |
| 3 `test_gcp_platform_resolve_image_*` | `platform/test_bootstrap.py` | `__new__` constructor bypass + `_project_id` | Use `FakeGcloud` to construct a real `GcpPlatform` |
| `test_build_cluster_images_...` | `cli/test_build_config_regions.py` | Mock `assert_called_once_with` on private helpers | Replace with CLI-level test or delete |
| `test_health_check_rpc` | `worker/test_worker.py` | `healthy=True` hardcoded; `uptime >= 0` trivially true | Assert on `running_tasks` count under known load |
| `test_fetch_logs_tail` | `worker/test_worker.py` | `sleep(2)` + `len >= 0` trivially true | Poll until logs appear; assert specific content |
| `test_multiple_workers_one_fails` | `e2e/test_heartbeat.py` | Claims multi-worker but uses single-worker cluster | Use `multi_worker_cluster` fixture or rename |
| `test_heartbeat_failure_with_pending_kills` | `e2e/test_heartbeat.py` | No kills issued; duplicate of threshold test | Remove or add explicit kill requests |
| `test_superseding_attempt_logs_info` | `cluster/test_attempt_logs.py` | Never asserts on "Superseding" log text | Add `assert "Superseding" in ...` or rename |
| `test_child_job_inherits_parent_env` | `cluster/test_env_propagation.py` | Never checks env vars are inherited | Capture env (like sibling tests) or delete |

---

## Policy Recommendations

Based on the patterns observed across ~340 tests, here are recommendations to
strengthen the Iris testing policy:

### 1. Ban private-attribute assertions

**Current policy:** "Tests should test stable behavior, not implementation details."

**Proposed addition:**

> **No assertions on private attributes.** Tests must NOT access or assert on
> attributes prefixed with `_` (e.g., `state._pending_dispatch`,
> `group._backoff_until`, `platform._s3_enabled`). If a behavior is worth
> testing, it must be observable through the public API. If no public API
> exists, either add one or accept the behavior is an implementation detail.

This was the single most common violation — found in `test_state.py`,
`test_autoscaler.py`, `test_scaling_group.py`, `test_platform.py`,
`test_coreweave_platform.py`, and `test_env_propagation.py`.

### 2. Ban mock call-count assertions for internal functions

**Current policy:** Prefers fakes over mocks.

**Proposed addition:**

> **No `assert_called_once_with` or `call_count` on internal helpers.** When
> using mocks at external boundaries (subprocess, HTTP, gcloud), asserting on
> call shape is acceptable. But mocking internal functions and asserting on
> call counts tests the wiring, not the behavior. If you need to verify a
> side effect, use a fake that records observable state.

Violations: `test_build_config_regions.py`, `test_bootstrap.py`,
`test_platform.py` (`test_gcp_reload_uses_full_stop_then_start`),
`test_scaling_group.py` (`test_scale_up_passes_tags_as_labels`).

### 3. Require assertions in every test function

**Proposed addition:**

> **Every test function must contain at least one `assert` statement or
> `pytest.raises` context.** Tests that only verify "does not raise" must
> include a comment explaining this intent. Screenshot-only tests are not
> acceptable without accompanying behavioral assertions.

Violations: `test_autoscaler_tab` (e2e dashboard), `test_controller_logs`,
`test_valid_config_accepted`.

### 4. Delete permanently-skipped tests

**Proposed addition:**

> **No permanently `@pytest.mark.skip`-ed tests.** A skipped test provides
> zero value and accumulates maintenance debt. If a test is flaky, either fix
> it or delete it. If a feature is not yet implemented, track it in an issue,
> not a skipped test.

Violations: `test_rpc_heartbeat_via_connect_client`,
`test_build_logs_visible_during_building_state`.

### 5. Enforce naming conventions

**Proposed addition:**

> **Test names must accurately describe the behavior being verified.** Use the
> pattern:
>
> `test_<subject>_<scenario>_<expected_outcome>`
>
> Examples:
> - `test_scheduler_with_insufficient_capacity_returns_empty_assignments`
> - `test_worker_after_heartbeat_timeout_is_marked_failed`
>
> Names must not promise behavior the test does not verify. A test named
> `test_multiple_workers_one_fails` that uses a single-worker cluster is
> misleading and must be renamed.
>
> **File naming:** Use `test_<module>.py` where `<module>` matches the source
> file being tested. Avoid generic names like `test_utils.py` for test
> infrastructure — use `test_util.py` (matching `src/iris/test_util.py`).

Violations: `test_client_with_resolver`, `test_multiple_workers_one_fails`,
`test_heartbeat_failure_with_pending_kills`,
`test_superseding_attempt_logs_info`.

### 6. Remove dead code from test files

**Proposed addition:**

> **No dead code in test files.** Unused test helpers, fixtures, fakes,
> classes, or imports must be removed. Dead test infrastructure (like
> `FailingResolver` defined but never called, or `create_test_entrypoint`
> shadowing a real import) adds confusion and maintenance burden.

Violations: `FailingResolver` in `test_actor_retry.py`,
`create_test_entrypoint` in `worker/test_dashboard.py` and
`worker/test_worker.py`, `chronos` fixture in `e2e/conftest.py`,
`test_builder.py` (empty file).

### 7. Prefer `iris.time_utils` over raw `time.sleep` in tests

**Proposed addition:**

> **Avoid bare `time.sleep()` in test polling loops.** Use
> `iris.time_utils.Deadline` or `ExponentialBackoff.wait_until()` for polling.
> Bare sleeps make tests both slow (sleeping longer than needed) and flaky
> (not sleeping long enough on slow CI). Use `wait_for_condition` from
> `tests/test_utils.py` for simple condition polling.
>
> Exception: A single short sleep to let a background thread start is
> acceptable when documented with a comment.

Violations: `_wait_for_workers` in `test_local_cluster.py`, cascade tests in
`test_autoscaler.py`, `test_fetch_logs_tail` in `worker/test_worker.py`.

### 8. Narrow exception handling in test helpers

**Proposed addition (reinforces existing policy):**

> **Test helpers must not use bare `except Exception`.** Even in polling loops
> where connection errors are expected during startup, use a specific exception
> type (e.g., `ConnectError`, `OSError`). Bare exception swallowing hides real
> bugs during test setup.

Violation: `_wait_for_workers` in `test_local_cluster.py`.

### 9. Mark E2E tests consistently

**Proposed addition:**

> **All tests that boot a cluster (local or Docker) must be marked
> `@pytest.mark.e2e`.** Tests that also require Docker must additionally be
> marked `@pytest.mark.docker`. E2E tests should live in `tests/e2e/` unless
> there is a strong reason to co-locate them elsewhere.

Violation: `test_cli_local_cluster_e2e` lives in `tests/cli/` without an
`e2e` marker.

### 10. Consolidate shared test fakes

**Proposed addition:**

> **Shared test fakes must live in a common location.** Use
> `tests/cluster/platform/fakes.py` for Platform fakes and
> `src/iris/test_util.py` for general test utilities. Do not define
> identically-named fakes in multiple files (e.g., `FakePlatform` exists in
> both `platform/fakes.py` and `controller/test_vm_lifecycle.py`). Move
> `FakeKubectl` from `test_coreweave_platform.py` to `fakes.py` for
> reusability.

---

## Proposed Updated `AGENTS.md` Testing Section

```
## Testing

* Tests should test stable behavior, not implementation details.

ABSOLUTELY DO NOT test things that are trivially caught by the type checker.
Explicitly that means:

- No tests for "constant = constant"
- No tests for "method exists"
- No tests for "create an object(x, y, z) and attributes are x, y, z"

These tests have negative value — they make our code more brittle.

### What to Test

Test _stable behavior_ instead. Prefer integration-style tests which exercise
behavior and test externally-observable output. You can use mocks as needed to
isolate external dependencies (e.g., subprocess for gcloud/kubectl, HTTP for
remote APIs), but prefer "fakes" — real implementations backed by in-memory or
temporary-file state — when reasonable.

### What NOT to Test

- **Private attributes**: No assertions on `_`-prefixed attributes. If a
  behavior is worth testing, it must be observable through the public API.
- **Internal call dispatch**: No `assert_called_once_with` or `call_count`
  on internal helpers. Mock at external boundaries only.
- **Python language semantics**: Do not test that `list(set)` creates a
  snapshot, or that `len(list) >= 0`. Test application behavior.
- **Constructor round-trips**: Do not test that `Foo(x=1).x == 1`.

### Test Hygiene

- **Every test must assert something.** Tests that only verify "does not
  raise" must include a comment. Screenshot-only tests must have behavioral
  assertions alongside.
- **No permanently-skipped tests.** Fix or delete. Track missing features
  in issues, not skipped tests.
- **No dead code in test files.** Remove unused helpers, fakes, classes.
- **Delete empty stub files.** A test file with no test functions has
  negative value.

### Naming

Use `test_<subject>_<scenario>_<expected_outcome>`:

    test_scheduler_with_insufficient_capacity_returns_empty_assignments
    test_worker_after_heartbeat_timeout_is_marked_failed

Names must accurately describe the verified behavior. A test named
"test_multiple_workers_one_fails" that uses a single-worker cluster
is misleading and must be renamed or rewritten.

### Timing and Polling

- Avoid bare `time.sleep()` in polling loops. Use
  `iris.time_utils.Deadline`, `ExponentialBackoff.wait_until()`, or
  `wait_for_condition` from `tests/test_utils.py`.
- Test helpers must not use bare `except Exception`. Catch specific
  exception types even in startup-polling loops.

### Markers and Organization

- All tests that boot a cluster must be marked `@pytest.mark.e2e`.
- Docker-dependent tests must also be marked `@pytest.mark.docker`.
- E2E tests live in `tests/e2e/`.
- Shared fakes live in `tests/cluster/platform/fakes.py` or
  `src/iris/test_util.py`. Do not duplicate fakes across files.

### Protocols

Non-trivial public classes should define a protocol which represents
their _important_ interface characteristics. Test to this protocol,
not the concrete class: the protocol should describe the interesting
behavior of the class, but not betray the implementation details.
(You may of course _instantiate_ the concrete class for testing.)
```
