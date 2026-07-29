# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.cluster import ResourceConfig

from experiments.grug import dispatch


def test_dispatch_grug_training_run_forwards_priority(monkeypatch):
    submitted_requests = []

    class FakeJob:
        def wait(self, *, raise_on_failure: bool) -> None:
            assert raise_on_failure

    class FakeClient:
        def submit(self, request):
            submitted_requests.append(request)
            return FakeJob()

    monkeypatch.setattr(dispatch, "current_client", FakeClient)

    dispatch.dispatch_grug_training_run(
        run_id="priority-regression",
        config=object(),
        local_entrypoint=lambda _: None,
        resources=ResourceConfig.with_cpu(),
        priority=1,
    )

    assert submitted_requests[0].priority == 1
