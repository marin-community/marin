# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import cloudpickle

from experiments.evaluation.serving_parity import launch


def test_requested_cpus_reach_the_submitted_entrypoint(monkeypatch):
    original_resources = launch.ResourceConfig.with_tpu
    monkeypatch.setattr(
        launch.ResourceConfig,
        "with_tpu",
        lambda *args, **kwargs: replace(original_resources(*args, **kwargs), cpu=25.75),
    )
    client = MagicMock()
    client.submit.return_value = "/test/logprob"
    monkeypatch.setattr(
        launch,
        "connect_controller",
        lambda **kwargs: contextlib.nullcontext(SimpleNamespace(url="http://controller", credentials=None)),
    )
    monkeypatch.setattr(launch.IrisClient, "remote", lambda *args, **kwargs: contextlib.nullcontext(client))
    monkeypatch.setattr(
        "sys.argv",
        ["launch", "--hardware=v4", "--name=logprob", "--output=gs://bucket/result", "--submit"],
    )
    launch.main()
    entrypoint = client.submit.call_args.kwargs["entrypoint"]
    _function, args, _kwargs = cloudpickle.loads(entrypoint.workdir_files["_callable.pkl"])
    assert args[-1] == 25.75
