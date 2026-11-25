# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from fray.cluster import ResourceConfig
from rigging.timing import Duration

from experiments.grug import dispatch


def _noop(_: object) -> None:
    pass


def test_dispatch_forwards_allocator_environment_and_typed_grace(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "libjemalloc.so.2")
    monkeypatch.setenv("MALLOC_CONF", "background_thread:true,narenas:2")
    monkeypatch.setenv("IRIS_PREEMPTION_GRACE_SECONDS", "invalid")
    submitted = []
    job = SimpleNamespace(wait=lambda **_: None)
    client = SimpleNamespace(submit=lambda request: submitted.append(request) or job)
    monkeypatch.setattr(dispatch, "current_client", lambda: client)

    dispatch.dispatch_grug_training_run(
        run_id="allocator-test",
        config=object(),
        local_entrypoint=_noop,
        resources=ResourceConfig.with_cpu(),
        termination_grace=Duration.from_seconds(90),
    )

    assert submitted[0].environment.env_vars["LD_PRELOAD"] == "libjemalloc.so.2"
    assert submitted[0].environment.env_vars["MALLOC_CONF"] == "background_thread:true,narenas:2"

    assert submitted[0].termination_grace == Duration.from_seconds(90)
    assert "IRIS_PREEMPTION_GRACE_SECONDS" not in submitted[0].environment.env_vars
