# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converted code snapshots keep reference cases out of the public task."""

import base64

import pytest
from verifyit.spec import StdioSpec, spec_to_table

from taskcompendium.models import ResourceVisibility, Source, TaskSpec
from taskcompendium.pipeline.datasets import atlas_code
from taskcompendium.pipeline.models import RawRow
from taskcompendium.verifiers.executable import TaskTroveExecutableVerifier


@pytest.mark.parametrize("name", ["code_contests", "codenet"])
def test_code_snapshot_roundtrip_preserves_private_cases_and_oracle(name):
    payloads = {
        "tests/cases/input_0.txt": b"3 4\r\n",
        "tests/cases/output_0.txt": b"7\n",
        "tests/cases/input_1.txt": b"-5 3\n",
        "tests/cases/output_1.txt": b"-2\n",
    }
    oracle = b"printf 'print(sum(map(int,input().split())))' > /app/solution.py\n"
    row = RawRow(
        id="sum",
        source=Source(dataset="open-thoughts/TaskTrove", revision=atlas_code.REVISION, row="1", importer_revision="1"),
        data={
            "converted": {
                "instruction": "Read two integers and print their sum. Write /app/solution.py.",
                "grader_spec": spec_to_table(StdioSpec(command="python3 /app/solution.py")),
                "data_files": {path: base64.b64encode(content).decode() for path, content in payloads.items()},
                "control_files": {"solution/solve.sh": base64.b64encode(oracle).decode()},
            }
        },
    )
    normalized = atlas_code.normalize(row, "sha256:local", 30.0, 256)
    assert isinstance(normalized, TaskSpec)
    task = TaskSpec.model_validate_json(normalized.model_dump_json())
    public = [resource for resource in task.resources if resource.visibility == ResourceVisibility.AGENT]
    assert public == []
    assert {resource.path: resource.data() for resource in task.resources} == {
        **{"/" + path: content for path, content in payloads.items()},
        "/solution/solve.sh": oracle,
    }
    verifier = TaskTroveExecutableVerifier.model_validate_json(task.verifier.parameters_json)
    assert {resource.path: resource.data() for resource in verifier.resources} == {
        "/" + path: content for path, content in payloads.items()
    }
