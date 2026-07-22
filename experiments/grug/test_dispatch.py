# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace

from fray.cluster import ResourceConfig
from marin.execution.lazy import StepContext

from experiments.grug import dispatch
from experiments.grug.moe import launch_cw_scale


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
    resources = arguments.pop("resources", ResourceConfig.with_gpu("GB200", count=4))
    dispatch.dispatch_grug_training_run(
        run_id="probe-test",
        config="config",
        local_entrypoint=_entrypoint,
        resources=resources,
        **arguments,
    )
    assert len(requests) == 1
    return requests[0]


def test_dispatch_preserves_default_environment(monkeypatch) -> None:
    request = _capture_request(monkeypatch)

    assert request.max_retries_failure == 3
    assert request.environment.setup_scripts is None
    assert "LD_PRELOAD" not in request.environment.env_vars
def test_dispatch_with_nvidia_jax_image_protects_container_accelerator_stack(monkeypatch) -> None:
    request = _capture_request(
        monkeypatch,
        resources=ResourceConfig.with_gpu("GB200", count=4, image="nvcr.io/nvidia/jax:26.06-py3"),
    )

    assert request.environment.setup_scripts is not None
    script = request.environment.setup_scripts[0]
    excluded = set(re.findall(r"--no-install-package ([a-z0-9-]+)", script))
    assert {
        "jax",
        "jaxlib",
        "jax-cuda13-pjrt",
        "jax-cuda13-plugin",
        "cuda-python",
        "nvidia-cublas",
        "nvidia-cudnn-cu13",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-nccl-cu13",
    } <= excluded
    assert "nvidia-cutlass-dsl" not in excluded
    assert "nvidia-cutlass-dsl-libs-cu13" not in excluded
    assert "venv --system-site-packages" in script
    assert "--active --inexact" in script
    assert "--package marin-root" in script
    assert "--package marin-root --no-group dev" not in script
    assert not any(line.lstrip().startswith("+") for line in script.splitlines())
    assert "ngc-jax-before.sha256" in script
    assert "ngc-jax-after.sha256" in script
    assert "nvidia_cutlass_dsl_libs_cu13-*.dist-info" in script
    assert "nvidia_cutlass_dsl_libs_base-*.dist-info" in script
    assert "assert cutlass.__file__.startswith(venv)" in script
    assert "assert _cutlass_ir.__file__.startswith(venv)" in script


def test_scale_checkpoint_routes_task_image_to_train_workers(monkeypatch) -> None:
    image = "nvcr.io/nvidia/jax:26.06-py3"
    monkeypatch.setenv("SCALE_TASK_IMAGE", image)
    monkeypatch.setenv("RUN_ID", "ngc-image-test")

    step = launch_cw_scale.build_scale_checkpoint()

    assert step.runtime_args["train_resources"].image == image


def test_scale_checkpoint_can_skip_final_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("SCALE_FINAL_CHECKPOINT", "skip")
    monkeypatch.setenv("RUN_ID", "skip-final-checkpoint-test")

    step = launch_cw_scale.build_scale_checkpoint()
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args.keys(), step.deps))

    assert config.grug_trainer.final_checkpoint.value == "skip"
