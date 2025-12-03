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

import pytest
from typing import Any
from huggingface_hub import hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
FILENAME = "sample/10BT/000_00000.parquet"
REVISION = "3c452cb"


@pytest.fixture(scope="session")
def parquet_file() -> str:
    print(f"\n[Setup] Ensuring {FILENAME} is available...")
    file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", revision=REVISION)

    # Warm-up OS page cache to prevent disk I/O jitter from affecting the results.
    print(f"[Setup] Warming up OS cache for {file_path}...")
    with open(file_path, "rb") as f:
        while f.read(1024**2):
            pass

    return file_path


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--run-benchmark", action="store_true", default=False, help="run benchmark tests")


def pytest_collection_modifyitems(config: Any, items: list[pytest.Item]) -> None:
    # If the --run-benchmark flag is set, do not skip anything.
    if config.getoption("--run-benchmark"):
        return
    # If the flag is not set, we check every test item
    skip_benchmark = pytest.mark.skip(reason="need --run-benchmark option to run")
    for item in items:
        if "benchmark" in item.fixturenames:
            item.add_marker(skip_benchmark)
