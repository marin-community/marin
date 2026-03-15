# Iris Smoke Test Structure & CI Workflow

## File Inventory

| File | Role |
|------|------|
| `lib/iris/tests/e2e/test_smoke.py` | Main smoke test file (~850 lines). All tests marked `@pytest.mark.e2e`. Tests dashboard rendering, scheduling, endpoints, logs, profiling, checkpoint/restore, stress, GPU metadata. |
| `lib/iris/tests/e2e/conftest.py` | Shared E2E fixtures: `cluster` (function-scoped), `multi_worker_cluster`, `chronos`, `k8s_cluster`, `k8s_runtime`, Playwright stubs. Also defines `IrisTestCluster`, `ClusterCapabilities`, `discover_capabilities()`. |
| `lib/iris/tests/e2e/helpers.py` | `TestJobs` class with static callables (`quick`, `sleep`, `fail`, `noop`, `busy_loop`, `log_verbose`, etc.) used as job payloads. |
| `lib/iris/examples/smoke.yaml` | Cloud smoke test config: GCP project, controller in `europe-west4-b`, TPU v5e-16 scale groups in two zones, CPU on-demand group. |
| `lib/iris/examples/test.yaml` | Default test config (`DEFAULT_CONFIG`). Used by `conftest.py`'s function-scoped `cluster` fixture and by `test_smoke.py`'s `_make_smoke_config()` as a base. |
| `.github/workflows/iris-cloud-smoke.yaml` | CI workflow for cloud smoke tests. Triggered by `/iris-smoke` comment or manual dispatch. |

## Cluster Setup/Teardown

### Local mode (default)

The `smoke_cluster` fixture in `test_smoke.py:203` is **module-scoped**. In local mode it:

1. Calls `_make_smoke_config()` which loads `examples/test.yaml`, clears scale groups, then adds:
   - 4 CPU workers (`_add_cpu_group`)
   - 2-VM TPU coscheduling group (`_add_coscheduling_group` from conftest)
   - 4-VM TPU coscheduling group (`_add_coscheduling_group_4vm`)
   - 2 multi-region CPU groups (`_add_multi_region_groups`)
   - Total: `SMOKE_WORKER_COUNT = 12` local workers
2. Calls `make_local_config(config)` which sets `controller.local.SetInParent()`, making it an in-process cluster.
3. Uses `connect_cluster(config)` context manager (`lib/iris/src/iris/cluster/manager.py:24`) which creates a `LocalCluster`, calls `cluster.start()`, yields the address, and calls `cluster.close()` on exit.
4. `LocalCluster` (`lib/iris/src/iris/cluster/local_cluster.py:142`) runs Controller + Autoscaler(LocalPlatform) in-process with thread-based workers.
5. Wraps in `IrisTestCluster` dataclass and calls `wait_for_workers(12, timeout=60)`.

### Cloud mode

Three sub-modes controlled by `--iris-mode`:
- **`full`**: `_cloud_smoke_cluster()` handles full lifecycle — stop existing → clear remote state → build images → start controller → tunnel → yield → stop.
- **`keep`**: Same as `full` but skips `platform.stop_all()` in teardown.
- **`redeploy`**: Same as `full`.

The `_cloud_smoke_cluster()` function (line 139):
1. `load_config(config_path)` — loads the YAML
2. Optionally overrides `label_prefix` and `remote_state_dir`
3. `_pin_latest_images(config)` + `_build_cluster_images(config)`
4. `IrisConfig(config).platform()` — creates platform (GCP/CoreWeave)
5. `platform.stop_all(config)` — tear down existing
6. `_clear_remote_state()` — wipe GCS state dir
7. `platform.start_controller(config)` — start controller VM
8. `platform.tunnel(address)` — SSH tunnel to controller
9. Create `IrisClient` + `ControllerServiceClientSync`, wait for workers
10. On exit: `platform.stop_all(config)` (unless `mode == "keep"`)

### Connect to existing controller (`--iris-controller-url`)

Already implemented in `smoke_cluster` fixture at line 222:
```python
if controller_url:
    client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)
    controller_client = ControllerServiceClientSync(address=controller_url, timeout_ms=30000)
    tc = IrisTestCluster(url=controller_url, client=client, ...)
    if is_cloud:
        tc.wait_for_workers(1, timeout=timeout)
    yield tc
    controller_client.close()
    return
```
This path does **no** cluster setup or teardown — it just connects to the given URL and runs tests.

## Config Information Needed

The smoke tests need from the config:
1. **Base config** (`examples/test.yaml`): platform settings, GCP project, autoscaler defaults, worker defaults (docker image, port), storage (remote_state_dir), controller settings.
2. **Scale groups**: CPU/TPU resource specs, min/max slices, regions, device types/variants, coscheduling params. In local mode these are overridden programmatically; in cloud mode they come from `examples/smoke.yaml`.
3. **label_prefix**: Used in cloud mode to isolate CI runs (passed via `--iris-label-prefix`).
4. **Capabilities discovery**: `discover_capabilities()` in `conftest.py:247` probes live workers via `ListWorkersRequest` to determine regions, device types, and coscheduling availability. Tests use `pytest.skip()` if needed capabilities are absent.

## Controller URL Usage Patterns

1. **`--iris-controller-url` pytest option** (`conftest.py:72`): Direct connection to existing controller, skip all setup/teardown.
2. **CLI `--controller-url`** (`src/iris/cli/main.py:26`): `resolve_cluster_name()` uses it for cluster identification.
3. **Worker config** (`test.yaml:21`): `controller_address: "${IRIS_CONTROLLER_ADDRESS}"` — workers use env var to find controller.
4. **Inside jobs** (`helpers.py:88`): `get_job_info().controller_address` — jobs connect back to controller for endpoint registration.

## CI Workflow Structure

`.github/workflows/iris-cloud-smoke.yaml`:

**Trigger**: `/iris-smoke` comment on PR (by MEMBER/COLLABORATOR/OWNER) or manual dispatch.

**Two jobs**:
1. **`cloud-smoke-test`** (45min timeout):
   - Checkout PR head
   - Install Python 3.12 + uv + Playwright
   - GCP auth + gcloud setup + SSH key
   - GHCR login
   - Run: `uv run pytest tests/e2e/test_smoke.py -m e2e --iris-config examples/smoke.yaml --iris-mode full --iris-label-prefix smoke-ci --timeout=1200`
   - Upload screenshots

2. **`cleanup`** (10min, runs even if test job fails/times out):
   - `iris cluster stop --label smoke-ci`
   - Failsafe: delete GCE instances and TPU VMs with `iris-smoke-ci-managed=true` label
   - Post commit status (success/failure)

**Concurrency**: `group: iris-cloud-smoke`, `cancel-in-progress: false` (never kill running test — would leak GCP resources).

## What Would Be Needed to Split Cluster Startup from Testing

The `--iris-controller-url` path already exists and cleanly separates concerns. To split into separate CI steps:

1. **Startup step**: Run something equivalent to `_cloud_smoke_cluster()` lines 139-176 (stop old → build images → start controller → wait for workers), then output the controller URL.

2. **Test step**: Run `pytest ... --iris-controller-url <url> --iris-mode keep` which hits the existing connect-only path.

3. **Teardown step**: Already exists as the separate `cleanup` job.

Key considerations:
- **Tunnel management**: `_cloud_smoke_cluster()` uses `platform.tunnel(address)` for SSH port-forwarding. A startup script would need to establish and maintain this tunnel, or expose the controller directly.
- **Image build**: `_pin_latest_images()` + `_build_cluster_images()` happen inside the fixture currently. These would move to the startup step.
- **Remote state cleanup**: `_clear_remote_state()` must happen before controller start.
- **Worker readiness**: The startup step should wait for workers (`wait_for_workers(1, timeout=600)`) before signaling ready.
- **The `IrisConfig`/`platform` objects** need to be created in both startup and teardown steps, or the teardown can use the CLI (`iris cluster stop --label ...`) as it already does.

The existing `conftest.py` already has all three paths wired through `smoke_cluster`:
- `controller_url` set → connect only (no lifecycle)
- `config_path` + `mode != "local"` → full cloud lifecycle
- neither → local in-process

No new fixture infrastructure is needed — just a script/CLI command to start the cluster and print the URL.
