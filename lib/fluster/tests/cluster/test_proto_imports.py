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


def test_cluster_proto_imports():
    """Verify cluster proto messages and services are importable."""
    from fluster import cluster_pb2

    # Controller messages
    job_spec = cluster_pb2.JobSpec(name="test")
    assert job_spec.name == "test"

    # Worker messages
    req = cluster_pb2.RunJobRequest(job_id="test-123")
    assert req.job_id == "test-123"


def test_actor_proto_imports():
    """Verify actor proto messages and services are importable."""
    from fluster import actor_pb2

    # Can instantiate messages
    call = actor_pb2.ActorCall(method_name="predict")
    assert call.method_name == "predict"

    # Response with oneof
    response = actor_pb2.ActorResponse(serialized_value=b"test_data")
    assert response.serialized_value == b"test_data"
