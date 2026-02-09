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

from pathlib import Path

import pytest
from fray.v2.local_backend import LocalClient
from zephyr import ZephyrContext
from zephyr.execution import default_zephyr_context
from zephyr.readers import load_jsonl


@pytest.fixture(scope="module")
def sync_backend():
    """Sets up a local ZephyrContext as the default for tests."""
    client = LocalClient()
    ctx = ZephyrContext(client=client, name="test-dedup")
    with default_zephyr_context(ctx):
        yield
    ctx.shutdown()
    client.shutdown()


@pytest.fixture(scope="module")
def docs():
    test_resources = Path(__file__).parent.joinpath("resources", "docs")
    docs = {}
    for doc_file in test_resources.glob("*.txt"):
        docs[doc_file.stem] = doc_file.read_text()
    return docs


def load_dedup_outputs(output_dir: str) -> dict[str, dict]:
    """Load all dedupe output files and return as id->doc mapping.

    Args:
        output_dir: Directory containing .jsonl.gz output files

    Returns:
        Dictionary mapping document IDs to document records
    """
    output_files = list(Path(output_dir).glob("**/*.jsonl.gz"))
    results = []
    for output_file in output_files:
        results.extend(load_jsonl(str(output_file)))
    return {r["id"]: r for r in results}
