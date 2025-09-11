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
import sys
import time
from pathlib import Path

import fasteners
import pytest
import ray
from pydantic import BaseModel

from marin.evaluation.evaluators.evaluator import ModelConfig

default_engine_kwargs = {"enforce_eager": True, "max_model_len": 1024}

large_model_engine_kwargs = {"max_model_len": 1024, "tensor_parallel_size": 8}

default_generation_params = {"max_tokens": 16}

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


class WorkerConfig(BaseModel):
    worker_count: int = 0
    cluster_address: str | None = None


@pytest.fixture(scope="module")
def model_config():
    config = ModelConfig(
        name="test-llama-200m",
        path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        engine_kwargs=default_engine_kwargs,
        generation_params=default_generation_params,
    )
    yield config
    config.destroy()


@pytest.fixture
def gcsfuse_mount_model_path():
    return "/opt/gcsfuse_mount/perplexity-models/llama-200m"


@pytest.fixture
def gcsfuse_mount_llama_70b_model_path():
    return "/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct"


@pytest.fixture
def gcsfuse_mount_llama_8b_model_path():
    return "/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-8B-Instruct"


@pytest.fixture
def test_file_path():
    return "gs://marin-us-east5/documents/chris-test/test_50.jsonl.gz"


@pytest.fixture
def current_date_time():
    # Get the current local time and format as MM-DD-YYYY-HH-MM-SS
    formatted_time = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    return formatted_time


@pytest.fixture(scope="session")
def ray_tpu_cluster(tmp_path_factory, worker_id):
    if os.getenv("PYTEST_XDIST_WORKER_COUNT") is None or "1":
        # Single worker, start Ray locally
        if os.getenv("START_RAY_TPU_CLUSTER") == "true":
            ray.init(
                namespace="marin",
                resources={"TPU": 8, "TPU-v6e-8-head": 1, "head_node": 1},
                num_cpus=120,
                ignore_reinit_error=True,
            )
        elif os.getenv("START_RAY_CPU_CLUSTER") == "true":
            ray.init(namespace="marin", num_cpus=8, resources={"head_node": 1}, ignore_reinit_error=True)
        else:
            ray.init(namespace="marin", num_cpus=8, resources={"head_node": 1}, ignore_reinit_error=True)
        yield
        ray.shutdown()
        return

    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    # best effort isolation in case we fail to clean up between test runs
    # pytest doesn't provide a unique temp path for a session when running
    # in xdist. we round to 100 seconds to reduce the risk of workers
    # getting different timestamps
    tmp_timestamp = time.time() // 100000
    root_tmp_dir = root_tmp_dir / f"ray_cluster_{tmp_timestamp}"
    root_tmp_dir.mkdir(parents=True, exist_ok=True)
    config_file = Path(root_tmp_dir) / "ray_cluster_config.json"
    lock_file = Path(root_tmp_dir) / "ray_cluster_config.lock"
    rw_lock = fasteners.InterProcessReaderWriterLock(str(lock_file))

    def _start_cluster():
        print(f"Worker {worker_id} starting Ray cluster...", file=sys.stderr)

        if os.getenv("START_RAY_TPU_CLUSTER") == "true":
            ray.init(
                namespace="marin",
                resources={"TPU": 8, "TPU-v6e-8-head": 1, "head_node": 1},
                num_cpus=120,
                ignore_reinit_error=True,
            )
        elif os.getenv("START_RAY_CPU_CLUSTER") == "true":
            ray.init(namespace="marin", num_cpus=8, resources={"head_node": 1}, ignore_reinit_error=True)
        else:
            ray.init(namespace="marin", num_cpus=8, resources={"head_node": 1}, ignore_reinit_error=True)
        return ray.worker._global_node.address

    def _init_worker():
        with rw_lock.write_lock():
            if not config_file.exists():
                # First worker to acquire lock - initialize cluster
                config = WorkerConfig(cluster_address=_start_cluster(), worker_count=1)
                with open(config_file, "w") as f:
                    f.write(config.model_dump_json())
                return config

            # Config file exists, increment worker count and connect to existing cluster
            with open(config_file, "r") as f:
                content = f.read().strip()
            current_config = WorkerConfig.model_validate_json(content)
            current_config.worker_count += 1

            # Connect to the existing cluster
            ray.init(
                address=current_config.cluster_address,
                namespace="marin",
                ignore_reinit_error=True,
            )

            print(
                f"Worker {worker_id} connected to cluster at {current_config.cluster_address}, "
                f"worker count {current_config.worker_count}",
                file=sys.stderr,
            )
            with open(config_file, "w") as f:
                f.write(current_config.model_dump_json())

            return current_config

    def _shutdown_worker():
        with rw_lock.write_lock():
            with open(config_file, "r") as f:
                content = f.read().strip()
            current_config = WorkerConfig.model_validate_json(content)
            current_config.worker_count -= 1
            print(
                f"Worker {worker_id} shutting down... worker count {current_config.worker_count}",
                file=sys.stderr,
            )
            if current_config.worker_count == 0:
                # Delete file under lock
                config_file.unlink()
                print("Last worker shutting down Ray cluster...", file=sys.stderr)
                ray.shutdown()
            else:
                with open(config_file, "w") as f:
                    f.write(current_config.model_dump_json())

    config = _init_worker()
    os.environ["RAY_ADDRESS"] = config.cluster_address
    yield
    _shutdown_worker()
