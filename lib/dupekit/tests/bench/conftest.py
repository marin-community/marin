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
import pyarrow as pa
import pyarrow.parquet as pq
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


@pytest.fixture(scope="session")
def small_parquet_path(tmp_path_factory: pytest.TempPathFactory, parquet_file: str) -> str:
    """
    Creates a smaller slice (250k rows) of the main parquet file for faster benchmarking
    and I/O tests.
    """
    fn = tmp_path_factory.mktemp("data_io") / "subset.parquet"
    pf = pq.ParquetFile(parquet_file)
    # 250k rows is substantial enough for I/O throughput tests
    first_batch = next(pf.iter_batches(batch_size=250_000))
    table = pa.Table.from_batches([first_batch])
    pq.write_table(table, fn)
    path_str = str(fn)

    # Warm up OS cache for this new file
    with open(path_str, "rb") as f:
        while f.read(1024**2):
            pass

    return path_str


@pytest.fixture(scope="session")
def in_memory_table(small_parquet_path: str) -> pa.Table:
    """
    Loads 250k rows into memory once. Used for marshaling and batch size tuning benchmarks.
    """
    return pq.read_table(small_parquet_path)


@pytest.fixture(scope="session")
def sample_batch(parquet_file: str) -> pa.RecordBatch:
    """
    Loads a single batch (10k rows) for algorithm benchmarks (hashing, dedupe logic).
    Columns are restricted to ensure we have 'text' and 'id'.
    """
    pf = pq.ParquetFile(parquet_file)
    # Ensure we get necessary columns if they exist, though 'iter_batches' defaults to all.
    return next(pf.iter_batches(batch_size=10_000))


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
