# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for process profiling CLI options."""

from contextlib import contextmanager

import pytest
from click.testing import CliRunner
from iris.cli.process_status import profile
from iris.rpc import job_pb2


@pytest.mark.parametrize(("args", "expected_native"), [(["threads"], False), (["threads", "--native"], True)])
def test_profile_threads_native_option_controls_rpc_request(monkeypatch, args, expected_native):
    requests: list[job_pb2.ProfileTaskRequest] = []

    class FakeClient:
        def profile_task(self, request: job_pb2.ProfileTaskRequest) -> job_pb2.ProfileTaskResponse:
            requests.append(request)
            return job_pb2.ProfileTaskResponse(profile_data=b"Thread 0x1")

    @contextmanager
    def fake_rpc_client_for_ctx(_ctx, *, url):
        assert url == "http://controller.test"
        yield FakeClient()

    monkeypatch.setattr("iris.cli.process_status.rpc_client_for_ctx", fake_rpc_client_for_ctx)

    result = CliRunner().invoke(profile, args, obj={"controller_url": "http://controller.test"})

    assert result.exit_code == 0, result.output
    assert requests[0].profile_type.threads.native is expected_native
