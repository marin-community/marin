# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that generated proto code can be imported."""


def test_controller_proto_imports():
    """Verify controller proto messages and services are importable."""
    from fluster import controller_pb2

    # Can instantiate messages
    job_spec = controller_pb2.JobSpec(name="test")
    assert job_spec.name == "test"

    # Enums are accessible
    assert controller_pb2.JOB_STATUS_PENDING == 1
    assert controller_pb2.JOB_STATUS_RUNNING == 3


def test_worker_proto_imports():
    """Verify worker proto messages and services are importable."""
    from fluster import worker_pb2

    # Can instantiate messages
    req = worker_pb2.RunJobRequest(job_id="test-123")
    assert req.job_id == "test-123"

    # Enums are accessible
    assert worker_pb2.JOB_STATUS_BUILDING == 2
    assert worker_pb2.JOB_STATUS_SUCCEEDED == 4


def test_actor_proto_imports():
    """Verify actor proto messages and services are importable."""
    from fluster import actor_pb2

    # Can instantiate messages
    call = actor_pb2.ActorCall(method_name="predict")
    assert call.method_name == "predict"

    # Response with oneof
    response = actor_pb2.ActorResponse(serialized_value=b"test_data")
    assert response.serialized_value == b"test_data"
