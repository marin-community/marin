# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the auxiliary Parquet split through local and URL storage boundaries."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.datasets.evalchemy.reasoning import MMLU_VALIDATION_FILE, mmlu_support


@pytest.mark.parametrize("as_url", [False, True])
def test_mmlu_support_reads_staged_parquet_from_local_or_url_prefix(tmp_path, as_url):
    inputs = json.loads((Path(__file__).parent / "fixtures/evalchemy_inputs.json").read_text())["choices"][0]
    rows = inputs["fewshot_rows"]
    path = tmp_path / MMLU_VALIDATION_FILE
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), path)
    root = tmp_path.as_uri() + "/" if as_url else str(tmp_path)
    assert mmlu_support(root) == tuple(rows)
