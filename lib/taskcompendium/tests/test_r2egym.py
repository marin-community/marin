# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R2E-Gym importer and original status-map contract tests."""

import gzip
import hashlib
import json
from pathlib import Path

import msgspec
import pytest

from taskcompendium.importers.r2e_assets.source_parser import decolor_dict_keys, parse_log_pytest
from taskcompendium.importers.r2egym import import_row
from taskcompendium.models import (
    ContainerRuntime,
    DockerEnvironment,
    EmptyWorkspace,
    ImageOverlay,
    Rejected,
    ResourceRole,
)

FIXTURE = Path(__file__).parent / "fixtures/r2egym/rows.json.gz"


def _row() -> dict:
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))[0]


_VERIFIER_RUNTIME = ContainerRuntime(
    "docker.io/taskcompendium/r2e-verifier@sha256:" + "a" * 64,
    workspace=ImageOverlay((".venv",)),
    supervisor_python="/usr/local/bin/python3",
)


def test_r2egym_import_preserves_source_state_and_hides_verifier_material():
    specification = import_row(_row(), verifier_runtime=_VERIFIER_RUNTIME)

    assert not isinstance(specification, Rejected)
    assert isinstance(specification.environment, DockerEnvironment)
    assert specification.environment.image.endswith(
        "@sha256:d546d21f56bb6b9c99045bd8fea196c04e43c9fbd6d9bc36fba4a90f88a45276"
    )
    assert all(ResourceRole.AGENT not in resource.roles for resource in specification.resources)
    assert any(resource.path == "oracle/execution_result.json" for resource in specification.resources)


def test_r2egym_rejects_implicit_verifier_runtime():
    result = import_row(_row())

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_environment"


def test_r2egym_rejects_supervisor_resolved_through_source_python_path():
    runtime = msgspec.structs.replace(_VERIFIER_RUNTIME, supervisor_python="python3")
    result = import_row(_row(), verifier_runtime=runtime)

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_environment"


@pytest.mark.parametrize("workspace", [EmptyWorkspace(), ImageOverlay(("unrelated",))])
def test_r2egym_rejects_runtime_that_discards_source_python(workspace):
    result = import_row(_row(), verifier_runtime=ContainerRuntime(_VERIFIER_RUNTIME.image, workspace=workspace))

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_environment"


def test_r2egym_rejects_unknown_or_malformed_image():
    row = _row()
    row["docker_image"] = "namanjain12/orange3_final:mutable-tag"

    result = import_row(row, verifier_runtime=_VERIFIER_RUNTIME)

    assert isinstance(result, Rejected)
    assert result.reason.value == "unrecoverable_source"


def test_r2egym_rejects_execution_result_from_different_commit():
    row = _row()
    execution = json.loads(row["execution_result_content"])
    execution["new_commit_hash"] = "different-commit"
    row["execution_result_content"] = json.dumps(execution)

    result = import_row(row, verifier_runtime=_VERIFIER_RUNTIME)

    assert isinstance(result, Rejected)
    assert result.reason.value == "unrecoverable_source"


@pytest.mark.parametrize("index", range(20))
def test_real_r2egym_observed_logs_preserve_original_status_map(index):
    row = json.loads(gzip.decompress(FIXTURE.read_bytes()))[index]
    execution = json.loads(row["execution_result_content"])
    expected = decolor_dict_keys(json.loads(row["expected_output_json"]))
    for prefix, expected_match in (("new", True), ("old", False)):
        actual = decolor_dict_keys(parse_log_pytest(execution[f"{prefix}_commit_res_stdout"]))
        actual = {key.split(" - ")[0]: actual[key] for key in sorted(actual)}
        reference = {key.split(" - ")[0]: expected[key] for key in sorted(expected)}
        match = len(actual) == len(reference) and all(
            not key or (key in reference and actual[key] == reference[key]) for key in actual
        )
        assert match is expected_match


def test_real_r2egym_fixture_matches_recorded_source_provenance():
    provenance = json.loads(FIXTURE.with_name("provenance.json").read_text())
    assert hashlib.sha256(gzip.decompress(FIXTURE.read_bytes())).hexdigest() == provenance["sha256"]
    for row in json.loads(gzip.decompress(FIXTURE.read_bytes())):
        result = import_row(row, verifier_runtime=_VERIFIER_RUNTIME)
        assert not isinstance(result, Rejected)
        assert result.instructions == row["problem_statement"].strip()
