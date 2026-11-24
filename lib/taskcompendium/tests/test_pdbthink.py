# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic PDBThink inputs protect prompt preservation and grading semantics."""

import gzip
import hashlib
import json

import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.evaluation.response import completion_text
from taskcompendium.importers import pdbthink
from taskcompendium.importers.evalchemy.scorers.pdbthink import grade_pdbthink
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import conversation_messages


@pytest.fixture
def source_row():
    return {
        "render_id": "synthetic",
        "messages": [{"role": "system", "content": "Use FINAL."}, {"role": "user", "content": "Two points."}],
        "answer_schema": "integer",
        "gold_answer": {"value": 2},
        "parameters": {},
        "report_metadata": {
            "render_id": "synthetic",
            "question_family": "P01",
            "representation": "minimal_pdb",
            "is_rotation_variant": False,
        },
        "family_weight": 0.5,
    }


def test_conversion_preserves_messages_and_keeps_grading_private(source_row):
    row = RawRow("synthetic", Source(dataset="fixture", revision="1", row="data:0", importer_revision="1"), source_row)
    context = ConversionContext(
        {}, EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="/tmp/fixture.lock")
    )
    task = pdbthink.convert_pdbthink(row, context)
    assert conversation_messages(task.context.events) == source_row["messages"]
    config = json.loads(
        resource_bytes(next(resource for resource in task.resources.verifier if resource.path == "config.json"))
    )
    assert config["gold_answer"] == {"value": 2}
    assert not task.resources.worker
    assert grade_pdbthink(config, completion_text({"content": "FINAL: 2"}, "stop", "provider_response")).reward == 1


@pytest.mark.parametrize(
    "schema,gold,text,termination,message_extra,expected",
    [
        ("residue_set", {"value": ["A:A1", "A:G2"]}, "FINAL: A:A1", "stop", {}, 2 / 3),
        ("residue_pair", {"value": "A:A1--B:G2", "alternatives": ["A:A1--B:S3"]}, "FINAL: B:S3--A:A1", "stop", {}, 1.0),
        ("distance", {"value": 2.0}, "FINAL: 2.02", "stop", {}, 1.0),
        ("distance", {"value": 2.0}, "FINAL: 2.021", "stop", {}, 0.0),
        ("integer", {"value": 2}, "2", "stop", {}, 0.0),
        ("integer", {"value": 2}, "FINAL: 2", "length", {}, 0.0),
        ("integer", {"value": 2}, "", "stop", {"reasoning_content": "FINAL: 2"}, 0.0),
        ("integer", {"value": 2}, "<|start_think|>FINAL: 2", "stop", {}, 0.0),
        ("integer", {"value": 2}, "<|start_think|>guess<|end_think|>FINAL: 2", "stop", {}, 1.0),
        ("integer", {"value": 2}, "<tool_call>call</tool_call>\nFINAL: 2", "stop", {}, 0.0),
        ("integer", {"value": 2}, "FINAL: 2", "stop", {"refusal": "refused"}, 0.0),
        ("integer", {"value": 2}, "FINAL: 2", "content_filter", {}, 0.0),
        ("integer", {"value": 2}, "FINAL: 2", "tool_calls", {"tool_calls": [{"id": "unexpected"}]}, 0.0),
    ],
)
def test_source_parser_and_provider_signals_determine_score(
    source_row, schema, gold, text, termination, message_extra, expected
):
    config = {**source_row, "answer_schema": schema, "gold_answer": gold}
    response = completion_text({"content": text, **message_extra}, termination, "provider_response")
    result = grade_pdbthink(config, response)
    assert result.reward == float(expected == 1)
    assert result.detail["metrics"]["score"] == pytest.approx(expected)
    assert result.detail["metrics"]["macro_accuracy"] == float(expected == 1)
    assert result.detail["pdbthink_row"]["truncated"] == (termination == "length")


def test_bundle_family_weights_and_checksum(source_row, tmp_path, monkeypatch):
    rows = [
        source_row,
        source_row,
        {**source_row, "report_metadata": {**source_row["report_metadata"], "question_family": "G01"}},
    ]
    path = tmp_path / "evaluation.json.gz"
    payload = gzip.compress(json.dumps({"renders": rows}).encode())
    path.write_bytes(payload)
    monkeypatch.setitem(pdbthink.manifest(), "bundle_sha256", hashlib.sha256(payload).hexdigest())
    loaded = list(pdbthink.read_pdbthink(StoragePath(str(path)), ConversionContext({}, None)))
    assert [row["family_weight"] for row in loaded] == [0.5, 0.5, 1]
    path.write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        list(pdbthink.read_pdbthink(StoragePath(str(path)), ConversionContext({}, None)))
