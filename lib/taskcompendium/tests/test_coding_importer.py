# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove coding-family importer contracts."""

from pathlib import Path

import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_coding import import_task
from taskcompendium.models import ContainerRuntime, DockerEnvironment, Rejected, ResourceRole

FIXTURES = Path(__file__).parent / "fixtures/coding"
IMAGE = "python@sha256:" + "a" * 64


def _archive(name: str, family: str):
    row = name.rsplit("-", 1)[1].removesuffix(".tar.gz")
    return read_archive((FIXTURES / name).read_bytes(), row, family)


@pytest.mark.parametrize("row", [1487, 1488, 1489])
def test_python_unit_test_import_preserves_absolute_paths_and_private_boundary(row):
    result = import_task(_archive(f"pytest-row-{row}.tar.gz", "unit-test-gen"), python_image=IMAGE)

    assert not isinstance(result, Rejected)
    assert isinstance(result.environment, DockerEnvironment)
    assert result.environment.image == IMAGE
    assert isinstance(result.verifier_runtime, ContainerRuntime)
    assert result.verifier.mode is Mode.PYTEST
    assert result.verifier.parameters["paths"] == ("/tests/test_curriculum.py",)
    assert result.verifier.parameters["python"] == "/opt/tasktrove-pytest/bin/python"
    assert {resource.path for resource in result.resources if ResourceRole.VERIFIER in resource.roles} == {
        "test_curriculum.py"
    }
    assert all(ResourceRole.AGENT not in resource.roles for resource in result.resources)


def test_python_unit_test_import_rewrites_evaluation_boilerplate():
    forbidden_phrases = (
        "tests-focused",
        "tests will",
        "what tests check",
        "the tests",
        "pytest test",
        "test-friendly",
        "acceptance criteria",
    )

    for row in (1487, 1488, 1489):
        result = import_task(_archive(f"pytest-row-{row}.tar.gz", "unit-test-gen"), python_image=IMAGE)

        assert not isinstance(result, Rejected)
        prompt = result.instructions.lower()
        assert not any(phrase in prompt for phrase in forbidden_phrases)

    qr = import_task(_archive("pytest-row-1488.tar.gz", "unit-test-gen"), python_image=IMAGE)
    assert not isinstance(qr, Rejected)
    assert "border == 0 and data == 'test'" in qr.instructions
    assert "\\033[49m" in qr.instructions

    imx = import_task(_archive("pytest-row-1487.tar.gz", "unit-test-gen"), python_image=IMAGE)
    assert not isinstance(imx, Rejected)
    assert "4 + 8 * number_of_entries" in imx.instructions
    assert "parse(export(obj)) yields an equal object" in imx.instructions


@pytest.mark.parametrize("row", [0, 1, 2])
def test_stdio_import_preserves_cases_build_and_command(row):
    result = import_task(_archive(f"stdio-row-{row}.tar.gz", "competitive-programming"), native_image=IMAGE)

    assert not isinstance(result, Rejected)
    assert "verifier" not in result.instructions.lower()
    assert "special judge" not in result.instructions.lower()
    assert "read from standard input and write to standard output" in result.instructions
    assert result.verifier.mode is Mode.STDIO
    assert result.verifier.parameters["cases"] == "cases"
    assert "solution.py" in result.verifier.parameters["command"]
    assert "solution_bin" in result.verifier.parameters["command"]
    assert "-std=c++17" in result.verifier.parameters["build"]
    paths = {resource.path for resource in result.resources if ResourceRole.VERIFIER in resource.roles}
    assert "cases/input_0.txt" in paths
    assert "cases/output_0.txt" in paths


def test_coding_import_requires_caller_resolved_immutable_image():
    result = import_task(_archive("stdio-row-0.tar.gz", "competitive-programming"))

    assert isinstance(result, Rejected)
    assert "immutable toolchain image" in result.detail


def test_coding_import_rejects_mutable_image_tag():
    result = import_task(_archive("pytest-row-1487.tar.gz", "unit-test-gen"), python_image="python:3.10-slim")

    assert isinstance(result, Rejected)
    assert "digest" in result.detail
