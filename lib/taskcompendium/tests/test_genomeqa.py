# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic questions and completions exercise GenomeQA's native task boundary."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.importers import genomeqa as source
from taskcompendium.importers.evalchemy.scorers.genomeqa import grade_genomeqa
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow


@pytest.mark.parametrize(
    "response,answer",
    [
        ("### Analysis\nAnswer: A\n### Answer\nB", "B"),
        ("### Answer\nAnswer: A", "A"),
        ("Answer: B", "B"),
        ("I considered A or B.", ""),
        ("### Answer\nC", ""),
    ],
)
def test_final_answer_grading(response, answer):
    grade = grade_genomeqa("B", "AB", 1, response)
    assert grade.detail["metrics"]["accuracy"] == (answer == "B")
    assert grade.detail["metrics"]["unanswered"] == (answer == "")
    assert grade.detail["genomeqa_parsing"]["answer"] == answer


def test_source_overrides_preserve_format_and_checksum(tmp_path, monkeypatch):
    path = tmp_path / "override.parquet"
    pq.write_table(pa.table({"id": [1]}), path)
    context = ConversionContext({}, None)
    storage = StoragePath(str(path))
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(source.read_genomeqa(storage, context))

    monkeypatch.setitem(source.FORMAT_BY_SHA256, hashlib.sha256(path.read_bytes()).hexdigest(), "binary")
    [row] = source.read_genomeqa(storage, context)
    locator = Source(dataset="GenomeQA", revision="fixture", row="mcq/data.parquet:0", importer_revision="fixture")
    with pytest.raises(ValueError, match="mcq input contains binary data"):
        source.convert_genomeqa(RawRow("fixture", locator, row), context)
