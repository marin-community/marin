# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taskcompendium.datasets.kto_components import (
    COMPONENTS,
    KTO_REVISION,
    PARENT_FILE,
    PARENT_REVISION,
    KtoComponentError,
    component_inputs,
    normalize_component_binary,
)
from taskcompendium.datasets.preference_tasks import normalize_binary
from taskcompendium.grader import grader_config
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import ImportFailureKind, RawRow
from taskcompendium.pipeline.sources import staged_raw_file_rows


def pair(component, request, system="Keep full context"):
    history = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Earlier request"},
        {"role": "assistant", "content": "Earlier response"},
        {"role": "user", "content": request},
    ]
    return {
        "dataset": component,
        "chosen": [*history, {"role": "assistant", "content": "Chosen"}],
        "rejected": [*history, {"role": "assistant", "content": "Rejected"}],
    }


def observations(parent):
    rows = []
    for row in parent:
        for field, label in (("chosen", True), ("rejected", False)):
            messages = deepcopy(row[field])
            rows.append({"prompt": messages[:-1], "completion": messages[-1:], "label": label})
    return rows


def stage(tmp_path, parent, kto):
    (tmp_path / "component-parent/data").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    pq.write_table(pa.Table.from_pylist(parent), tmp_path / PARENT_FILE)
    pq.write_table(pa.Table.from_pylist(kto), tmp_path / "data/train-00000-of-00001.parquet")
    return str(tmp_path)


def test_components_preserve_full_history_labels_original_order_and_duplicate_observations(tmp_path):
    parent = [
        pair(COMPONENTS[0], "Shared final request", "First system"),
        pair(COMPONENTS[1], "Shared final request", "Other system"),
    ]
    parent.append(deepcopy(parent[0]))
    kto = observations(parent)[::-1]
    root = stage(tmp_path, parent, kto)
    selected = list(
        staged_raw_file_rows(
            root,
            "data/train-00000-of-00001.parquet",
            component_inputs(component=COMPONENTS[0], kto_revision=KTO_REVISION, parent_revision=PARENT_REVISION).files,
        )
    )
    assert [row["locator"] for row in selected] == [
        "data/train-00000-of-00001.parquet:0",
        "data/train-00000-of-00001.parquet:1",
        "data/train-00000-of-00001.parquet:4",
        "data/train-00000-of-00001.parquet:5",
    ]
    for row in selected:
        original = kto[row["index"]]
        assert {key: row["data"][key] for key in original} == original
        evidence = row["data"]["kto_component_provenance"]
        assert evidence["parent_rows"] == [0, 2]
        assert evidence["parent_revision"] == PARENT_REVISION
        task = normalize_binary(
            RawRow(
                str(row["index"]),
                Source(dataset="trl-lib/kto-mix-14k", revision="pin", row=str(row["index"]), importer_revision="test"),
                row["data"],
            )
        )
        assert isinstance(task, TaskSpec)
        assert len(task.context.events) == 4
        assert task.context.events[0].content == "First system"
    assert [row["data"]["label"] for row in selected] == [False, True, False, True]


@pytest.mark.parametrize(
    "failure",
    [
        "unmatched",
        "wrong_label",
        "missing_observation",
        "cross_component_ambiguity",
        "malformed_label",
        "changed_public_boundary",
    ],
)
def test_complete_join_failure_prevents_even_first_component_row_admission(tmp_path, failure):
    parent = [pair(COMPONENTS[0], "Valid request"), pair(COMPONENTS[1], "Other request")]
    kto = observations(parent)
    if failure == "unmatched":
        kto[-1]["prompt"][-1]["content"] = "Not in parent"
    elif failure == "wrong_label":
        kto[-1]["label"] = True
    elif failure == "missing_observation":
        kto.pop()
    elif failure == "cross_component_ambiguity":
        parent[1] = {**deepcopy(parent[0]), "dataset": COMPONENTS[1]}
        kto = observations(parent)
    elif failure == "changed_public_boundary":
        kto[-1]["completion"] = [kto[-1]["prompt"].pop(), *kto[-1]["completion"]]
    else:
        for row in kto:
            row["label"] = str(row["label"])
    root = stage(tmp_path, parent, kto)
    rows = staged_raw_file_rows(
        root,
        "data/train-00000-of-00001.parquet",
        component_inputs(component=COMPONENTS[0], kto_revision=KTO_REVISION, parent_revision=PARENT_REVISION).files,
    )
    with pytest.raises(KtoComponentError) as failure_info:
        next(rows)
    assert failure_info.value.kind == (
        ImportFailureKind.SOURCE_DEFECT if failure == "malformed_label" else ImportFailureKind.UNSUPPORTED
    )
    assert "Valid request" not in str(failure_info.value)


def test_component_provenance_survives_normalized_json_privately_without_changing_grader(tmp_path):
    parent = [pair(COMPONENTS[0], "Preserve context")]
    root = stage(tmp_path, parent, observations(parent))
    selected = next(
        staged_raw_file_rows(
            root,
            "data/train-00000-of-00001.parquet",
            component_inputs(component=COMPONENTS[0], kto_revision=KTO_REVISION, parent_revision=PARENT_REVISION).files,
        )
    )
    row = RawRow(
        "original-task-id",
        Source(dataset="trl-lib/kto-mix-14k", revision=KTO_REVISION, row=selected["locator"], importer_revision="test"),
        selected["data"],
    )
    original = normalize_binary(row)
    normalized = normalize_component_binary(row)
    assert isinstance(original, TaskSpec) and isinstance(normalized, TaskSpec)
    restored = TaskSpec.model_validate_json(normalized.model_dump_json())
    assert restored.context == original.context
    assert restored.source == original.source and restored.id == original.id
    assert restored.grader == original.grader
    assert grader_config(restored) == grader_config(original)
    evidence_resource = next(
        resource
        for resource in restored.resources.verifier
        if resource.path == "acquisition/kto_component_provenance.json"
    )
    evidence = json.loads(base64.b64decode(evidence_resource.source.content_base64))
    assert evidence == {
        "component": COMPONENTS[0],
        "parent_dataset": "argilla/dpo-mix-7k",
        "parent_revision": PARENT_REVISION,
        "parent_file": PARENT_FILE,
        "parent_rows": [0],
    }
    assert evidence_resource not in restored.resources.worker + restored.resources.all
    assert restored.resources.verifier[:-1] == original.resources.verifier
