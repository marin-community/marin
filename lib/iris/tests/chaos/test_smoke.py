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

import time
from iris.rpc import cluster_pb2
from iris.cluster.types import Entrypoint, ResourceSpec, EnvironmentSpec


def _ok():
    return 42


def test_smoke(cluster):
    _url, client = cluster

    # Submit job
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_ok),
        name="chaos-smoke",
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
    )

    # Wait for completion
    terminal_states = {
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
    }
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.status(str(job.job_id))
        if status.state in terminal_states:
            break
        time.sleep(0.5)
    else:
        status = client.status(str(job.job_id))

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
