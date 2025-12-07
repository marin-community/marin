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

import os
import shutil

import pytest
import ray
from pydantic import BaseModel

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


class WorkerConfig(BaseModel):
    worker_count: int = 0
    cluster_address: str | None = None


@pytest.fixture(scope="session")
def ray_tpu_cluster(tmp_path_factory, worker_id):
    """Start a Ray cluster for testing.

    When running under pytest-xdist, we need to ensure each cluster is isolated
    by specifying unique temp directories and ports.

    We additionally set the "RAY_LOCAL_CLUSTER" environment variable to signal to code like
    executor that this is a test cluster. This allows us to skip things like dependency collection
    that are unnecessary in a test environment.
    """
    # make ray less noisy
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    os.environ["RAY_SCHEDULER_EVENTS"] = "0"
    if not worker_id or worker_id == "master":
        worker_id = 0
    else:
        worker_id = int(worker_id.replace("gw", ""))
    worker_offset = 10 * worker_id

    # N.B. We cannot use the default temp directory as Ray will complain with:
    # AF_UNIX path length cannot exceed 103 bytes
    tmp_path = f"/tmp/ray_tests/{os.getpid()}/{worker_id}"

    init_args = {
        "address": "local",
        "namespace": "marin",
        # In case the user is running a Ray cluster already
        "dashboard_port": 10265 + worker_offset,
        "_temp_dir": tmp_path,
        "ignore_reinit_error": True,
    }
    print("Starting on worker_id", worker_id, "with init_args", init_args)
    if os.getenv("START_RAY_TPU_CLUSTER") == "true":
        tpu_resource_name = "TPU-v5litepod-4-head"
        chip_count = 4

        print(f"Starting TPU Ray cluster with resources TPU:{chip_count}, {tpu_resource_name}:1")
        ctx = ray.init(
            resources={"TPU": chip_count, tpu_resource_name: 1, "head_node": 1},
            num_cpus=120,
            **init_args,
        )
    elif os.getenv("START_RAY_CPU_CLUSTER") == "true":
        ctx = ray.init(
            **init_args,
            num_cpus=8,
            resources={"head_node": 1},
        )
    else:
        ctx = ray.init(
            **init_args,
            num_cpus=8,
            resources={"head_node": 1},
        )

    # update environment variable to pass Ray address to subprocesses
    os.environ["RAY_ADDRESS"] = ctx.address_info["address"]
    if ctx.address_info["webui_url"] is not None:
        os.environ["RAY_DASHBOARD_URL"] = ctx.address_info["webui_url"]
    os.environ["RAY_API_SERVER_ADDRESS"] = ctx.address_info["gcs_address"]
    os.environ["RAY_LOCAL_CLUSTER"] = "1"

    print(
        f"Initialized ray with address={ctx.address_info['address']}",
        f"webui_url={ctx.address_info['webui_url']}, gcs_address={ctx.address_info['gcs_address']}",
    )

    yield

    # cleanup temp directory
    # Use _exiting_interpreter=True to force shutdown even if workers are hung
    ray.shutdown(_exiting_interpreter=True)
    shutil.rmtree(tmp_path, ignore_errors=True)
