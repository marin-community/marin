# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import cloudpickle

from experiments.evaluation.serving_parity import launch


def test_requested_cpus_reach_the_tpu_gang(monkeypatch, tmp_path):
    original_resources = launch.ResourceConfig.with_tpu
    monkeypatch.setattr(
        launch.ResourceConfig,
        "with_tpu",
        lambda *args, **kwargs: replace(original_resources(*args, **kwargs), cpu=25.75),
    )
    client = MagicMock()
    client.submit.return_value = "/test/score"
    monkeypatch.setattr(
        launch,
        "connect_controller",
        lambda **kwargs: contextlib.nullcontext(
            SimpleNamespace(
                url="http://controller",
                credentials=None,
            )
        ),
    )
    monkeypatch.setattr(launch.IrisClient, "remote", lambda *args, **kwargs: contextlib.nullcontext(client))
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch",
            "--hardware=v4",
            "--name=score",
            "--weights=gs://bucket/model",
            "--output=gs://bucket/result",
            "--submit",
        ],
    )
    launch.main()
    entrypoint = client.submit.call_args.kwargs["entrypoint"]
    _function, args, _kwargs = cloudpickle.loads(entrypoint.workdir_files["_callable.pkl"])
    assert args[-1] == 25.75

    observed = {}

    @contextlib.contextmanager
    def gang(_launcher, **kwargs):
        observed.update(kwargs)
        yield SimpleNamespace(leader=False, identity=object(), check_alive=lambda: None)

    monkeypatch.setattr(launch, "tpu_ray_gang", gang)
    monkeypatch.setattr(launch, "wait_for_iris_vllm_shutdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(launch, "notify_iris_vllm_stopped", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    engine = launch.VllmEngineConfig(launcher=launch.VllmLauncherType.TPU)
    with launch.serving_model("v4", None, engine, args[-1]) as model:
        assert model is None
    assert observed["cpus_per_host"] == 25
