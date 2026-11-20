# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tomllib

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.long_loop.pi_rl_data import prepare_file


def test_pi_nonroot_parquet_preserves_tasks_and_private_verifier(tmp_path):
    original = TaskSpec(
        id="fixture:1",
        context=ConversationInput(events=(TextMessage(role="user", content="Clone and fix the repo"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="Harbor grades the trial"),
        source=Source(dataset="long-loop-smartie-swesmith", revision="pinned", row="1", importer_revision="pinned"),
        resources=ResourceGroups(
            verifier=(
                inline_resource(
                    "task.toml", b'[agent]\nuser="root"\n[verifier]\nuser="root"\n[environment]\nworkdir="/testbed"\n'
                ),
                inline_resource("environment/Dockerfile", b"FROM python:3.10-bookworm\nWORKDIR /testbed\n"),
                inline_resource("tests/test.sh", b"private grader bytes"),
                inline_resource("instruction.md", b"Clone and fix the repo"),
            )
        ),
        tags=("harbor",),
    )
    unrelated = original.model_copy(
        update={"id": "other:2", "source": original.source.model_copy(update={"dataset": "other"})}
    )
    raw = [original.model_dump_json(), unrelated.model_dump_json()]
    source = tmp_path / "input.parquet"
    pq.write_table(pa.table({"task_spec": raw}), source)
    identities = io.StringIO()
    result = prepare_file(source, tmp_path / "output.parquet", identities)
    output = pq.read_table(tmp_path / "output.parquet").to_pylist()
    prepared = TaskSpec.model_validate_json(output[0]["task_spec"])
    assert (prepared.id, prepared.context, prepared.source, prepared.grader) == (
        original.id,
        original.context,
        original.source,
        original.grader,
    )
    files = {r.path: resource_bytes(r) for r in prepared.resources.verifier}
    assert files["tests/test.sh"] == b"private grader bytes"
    assert files["instruction.md"] == b"Clone and fix the repo"
    config = tomllib.loads(files["task.toml"].decode())
    assert config["agent"]["user"] != "root"
    assert config["verifier"]["user"] == "root"
    assert config["environment"]["workdir"] == "/testbed"
    assert config["agent"]["env"]["HOME"] != "/root"
    assert config["environment"] == tomllib.loads(resource_bytes(original.resources.verifier[0]).decode())["environment"]
    package = json.loads(files["environment/long-loop-pi/package.json"])
    assert "@earendil-works/pi-coding-agent" in package["dependencies"]
    assert output[1]["task_spec"] == raw[1]
    assert (result["rows"], result["changed"]) == (2, 1)
    rows = [json.loads(line) for line in identities.getvalue().splitlines()]
    assert [r["changed"] for r in rows] == [True, False]
    assert rows[0]["input_sha256"] != rows[0]["output_sha256"]
    assert rows[1]["input_sha256"] == rows[1]["output_sha256"]
