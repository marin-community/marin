# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from fray.cluster import ResourceConfig

from experiments.grug import dispatch


def _entrypoint(config: str) -> None:
    del config


def _capture_request(monkeypatch, **arguments):
    requests = []
    handle = SimpleNamespace(wait=lambda **kwargs: None)
    monkeypatch.setattr(
        dispatch,
        "current_client",
        lambda: SimpleNamespace(submit=lambda request: requests.append(request) or handle),
    )
    dispatch.dispatch_grug_training_run(
        run_id="probe-test",
        config="config",
        local_entrypoint=_entrypoint,
        resources=ResourceConfig.with_gpu("GB200", count=4),
        **arguments,
    )
    assert len(requests) == 1
    return requests[0]


def test_dispatch_without_probe_preserves_default_environment(monkeypatch) -> None:
    request = _capture_request(monkeypatch)

    assert request.max_retries_failure == 3
    assert request.environment.setup_scripts is None
    assert "LD_PRELOAD" not in request.environment.env_vars


def test_dispatch_with_probe_builds_and_preloads_worker_library(monkeypatch) -> None:
    request = _capture_request(
        monkeypatch,
        env_vars={
            "MARIN_CUDA_MODULE_PROBE_PROFILE": "trace_sync_split",
            "MARIN_CUDA_MODULE_PROBE_REQUIRED": "1",
            "MARIN_CUDA_MODULE_PROBE_LOG_DIR": "/tmp/cubin-probe",
        },
        max_retries_failure=0,
    )

    assert request.max_retries_failure == 0
    assert request.environment.env_vars["LD_PRELOAD"] == "/app/.venv/lib/libmarin_cuda_module_probe.so"
    assert request.environment.env_vars["MARIN_CUDA_MODULE_PROBE_PROFILE"] == "trace_sync_split"
    assert request.environment.setup_scripts is not None
    assert len(request.environment.setup_scripts) == 3
