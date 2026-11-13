# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.preemption import TERMINATION_GRACE_ENV
from iris.rpc import controller_pb2
from iris.testing.controller import make_test_entrypoint


def test_termination_grace_defaults_are_visible_on_submitted_job(controller_service):
    response = controller_service.launch_job(
        controller_pb2.Controller.LaunchJobRequest(name="/test-user/default-grace", entrypoint=make_test_entrypoint()),
        None,
    )
    stored = controller_service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=response.job_id), None
    ).request
    assert stored.termination_grace.milliseconds == 10_000


def test_termination_grace_rejected_at_admission_without_creating_job(controller_service):
    request = controller_pb2.Controller.LaunchJobRequest(
        name="/test-user/excessive-grace", entrypoint=make_test_entrypoint()
    )
    request.termination_grace.milliseconds = 86400_000
    with pytest.raises(ConnectError) as error:
        controller_service.launch_job(request, None)
    assert error.value.code == Code.INVALID_ARGUMENT
    with pytest.raises(ConnectError) as missing:
        controller_service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=request.name), None)
    assert missing.value.code == Code.NOT_FOUND


def test_runtime_termination_grace_cannot_be_supplied_as_user_environment(controller_service):
    request = controller_pb2.Controller.LaunchJobRequest(
        name="/test-user/spoofed-grace", entrypoint=make_test_entrypoint()
    )
    request.environment.env_vars[TERMINATION_GRACE_ENV] = "86400000"
    with pytest.raises(ConnectError) as error:
        controller_service.launch_job(request, None)
    assert error.value.code == Code.INVALID_ARGUMENT
