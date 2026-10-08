# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the checked-in ``swe_trusted_paths`` exemplar."""

import json
import subprocess
import uuid
from pathlib import Path

import pytest
from verifyit.spec import PytestSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    TEST_SH,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"
FAMILY = "swe-repo"
SOURCE = "laion__swesmith-oracle-filtered-v2"

OLD_GRADER_FILES = ("tests/config.json", "tests/test_state.py", "tests/install_trusted_test_paths.sh")


def _fixture() -> bytes:
    return (FIXTURES / "swe_trusted_paths.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo(SOURCE, SourceVerdict.KEEP, FAMILY, "")


def _edited(**config_overrides: object) -> bytes:
    task = read_task_binary(_fixture())
    config = json.loads(task.text("tests/config.json"))
    config.update(config_overrides)
    task.files["tests/config.json"] = json.dumps(config).encode()
    return write_task_binary(task)


def test_exemplar_converts_to_pytest_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "pytest"
    assert record.tags == ["code", "swe", "swe-repo", "trusted-test-paths"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, PytestSpec)
    assert spec.workspace == "/testbed"
    assert len(spec.must_pass) == 18
    assert len(spec.must_not_break) == 655
    assert all(path.startswith("tests/") for path in spec.paths)
    assert "3b1e6ec37ffeacc5bed0e55e287f86166b4db21c" in spec.setup
    assert "614b1348ef893b4fa90e76f56df44ee64f5d0222" in spec.setup

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    for old_grader_file in OLD_GRADER_FILES:
        assert old_grader_file not in task.files, "old grader code must not ship"
    assert "tests/trusted_test_paths.txt" in task.files

    dockerfile = task.text(DOCKERFILE)
    assert INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert "pytest-json-report" in dockerfile

    assert record.has_solution and record.solution_binary is not None
    assert "solution/solve.sh" in read_task_binary(record.solution_binary).files


def test_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert verify_task(record.task_binary) is None


def test_missing_config_json_is_rejected_as_unsupported_variant():
    task = read_task_binary(_fixture())
    del task.files["tests/config.json"]
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_empty_fail_and_pass_to_pass_is_rejected_as_null_grader():
    record = convert_one(_info(), "t.tar.gz", _edited(FAIL_TO_PASS=[], PASS_TO_PASS=[]), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_test_sh_without_install_invocation_is_rejected_as_unsupported_variant():
    task = read_task_binary(_fixture())
    task.files["tests/test.sh"] = b"#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n"
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_graded_file_missing_from_manifest_is_rejected_as_unsupported_variant():
    record = convert_one(
        _info(),
        "t.tar.gz",
        _edited(FAIL_TO_PASS=["tests/not_in_manifest.py::TestX::test_y"]),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_fail_to_pass_outside_a_python_file_is_rejected():
    record = convert_one(
        _info(), "t.tar.gz", _edited(FAIL_TO_PASS=["tests/tests.md::tests.md"]), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT
    assert "tests/tests.md::tests.md" in record.error


def test_pass_to_pass_outside_a_python_file_is_dropped_from_the_spec():
    record = convert_one(
        _info(),
        "t.tar.gz",
        _edited(PASS_TO_PASS=["tests/tests.md::tests.md", "tests/test_common.py::test_other"]),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert spec.must_not_break == ("tests/test_common.py::test_other",)
    assert all(path.endswith(".py") for path in spec.paths)


def test_fail_to_pass_doctest_item_is_rejected():
    record = convert_one(
        _info(), "t.tar.gz", _edited(FAIL_TO_PASS=["parso/__init__.py::parso"]), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT
    assert "parso/__init__.py::parso" in record.error


def test_pass_to_pass_doctest_items_are_dropped_but_parametrized_ids_with_colons_stay():
    record = convert_one(
        _info(),
        "t.tar.gz",
        _edited(
            PASS_TO_PASS=[
                "parso/tree.py::parso.tree.NodeOrLeaf.dump",
                "tests/test_common.py::test_parse_address[[::1]:8000-expected5]",
                "tests/test_common.py::TestAddress::test_ipv6[ff::aa:1::2]",
            ]
        ),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert spec.must_not_break == (
        "tests/test_common.py::test_parse_address[[::1]:8000-expected5]",
        "tests/test_common.py::TestAddress::test_ipv6[ff::aa:1::2]",
    )


def test_json_encoded_string_fail_to_pass_is_accepted():
    record = convert_one(
        _info(),
        "t.tar.gz",
        _edited(FAIL_TO_PASS=json.dumps(["tests/test_common.py::test_something"])),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert spec.must_pass == ("tests/test_common.py::test_something",)


def test_dockerfile_reuses_existing_pip_install_line_instead_of_adding_a_new_run():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    dockerfile = read_task_binary(record.task_binary).text(DOCKERFILE)
    body = dockerfile.split(INSTALL_MARKER)[0]
    assert body.count("pytest-json-report") == 1
    assert "RUN pip install --upgrade pip uv pytest pytest-json-report" in body


@pytest.mark.parametrize("repository", ["john-kurkowski__tldextract.3d1bf184", "marshmallow-code__marshmallow.9716fc62"])
def test_swesmith_preserves_selected_cases(repository):
    task = read_task_binary(_fixture())
    task.files[INSTRUCTION] = f"git clone https://github.com/swesmith/{repository} .\n".encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    converted = read_task_binary(record.task_binary)
    assert parse_spec(converted.text(VERIFIER_TOML)).must_pass == tuple(
        json.loads(task.text("tests/config.json"))["FAIL_TO_PASS"]
    )


@pytest.mark.docker
@pytest.mark.timeout(300)
@pytest.mark.parametrize("repository", ["john-kurkowski__tldextract.3d1bf184", "marshmallow-code__marshmallow.9716fc62"])
def test_swesmith_built_image_collects_with_legacy_plugin_and_test_dependencies(repository, tmp_path):
    task = read_task_binary(_fixture())
    task.files[INSTRUCTION] = f"git clone https://github.com/swesmith/{repository} .\n".encode()
    # Reproduce pytest 9 and the legacy plugin being installed before the repair.
    task.files[DOCKERFILE] = (
        b"FROM python:3.10-bookworm\n" b"RUN pip install pytest pytest-gitignore pytest-json-report\n"
    )
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    dockerfile = read_task_binary(record.task_binary).text(DOCKERFILE).split(INSTALL_MARKER)[0]
    probe = "import pytest\n\ndef test_compatible_collection():\n    assert pytest.version_tuple[0] < 9\n"
    if repository.startswith("marshmallow-code__"):
        probe += (
            "\ndef test_declared_dependency():\n    import simplejson\n"
            "    assert simplejson.loads(simplejson.dumps({'ok': True})) == {'ok': True}\n"
        )
    result = docker_pytest_result(dockerfile, probe, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def docker_pytest_result(dockerfile: str, probe: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    (tmp_path / "Dockerfile").write_text(dockerfile + "\nCOPY test_probe.py /probe/test_probe.py\nWORKDIR /probe\n")
    (tmp_path / "test_probe.py").write_text(probe)
    image = f"atlas-swesmith-regression:{uuid.uuid4().hex}"
    try:
        subprocess.run(["docker", "build", "-t", image, str(tmp_path)], check=True, capture_output=True, text=True)
        return subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                image,
                "python",
                "-m",
                "pytest",
                "--json-report",
                "--json-report-file=/probe/report.json",
                "test_probe.py",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        subprocess.run(["docker", "image", "rm", image], check=True, capture_output=True)


@pytest.mark.docker
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "repository", ["seperman__deepdiff.ed252022", "conan-io__conan.86f29e13", "oauthlib__oauthlib.1fd52536"]
)
def test_swesmith_preserves_project_pytest_and_installs_test_dependencies(repository, tmp_path):
    task = read_task_binary(_fixture())
    task.files[INSTRUCTION] = f"git clone https://github.com/swesmith/{repository} .\n".encode()
    task.files[DOCKERFILE] = b"FROM python:3.10-bookworm\nRUN pip install pytest pytest-json-report\n"
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    dockerfile = read_task_binary(record.task_binary).text(DOCKERFILE).split(INSTALL_MARKER)[0]
    probe = "import pytest\n\ndef test_compatible_collection():\n    assert pytest.version_tuple[0] < 9\n"
    if repository.startswith("seperman__"):
        # The actual DeepDiff development requirements pin this version. The
        # shared upper bound must allow that subsequent project installation.
        dockerfile += "\nRUN python -m pip install pytest==8.3.4\n"
        probe += "    assert pytest.__version__ == '8.3.4'\n"
    elif repository.startswith("conan-io__"):
        probe += (
            "\ndef test_project_dependencies():\n"
            "    import mock, webtest, jwt, bottle, parameterized\n"
            "    assert jwt.decode(jwt.encode({'ok': True}, 'key', algorithm='HS256'), "
            "'key', algorithms=['HS256']) == {'ok': True}\n"
        )
    else:
        probe += (
            "\ndef test_project_dependencies():\n"
            "    import blinker, jwt\n"
            "    from cryptography.hazmat.primitives.asymmetric import rsa\n"
            "    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)\n"
            "    token = jwt.encode({'ok': True}, key, algorithm='RS256')\n"
            "    assert jwt.decode(token, key.public_key(), algorithms=['RS256']) == {'ok': True}\n"
            "    seen = []\n"
            "    def receiver(sender):\n"
            "        seen.append(sender)\n"
            "    signal = blinker.Signal()\n"
            "    signal.connect(receiver)\n"
            "    signal.send('scope')\n"
            "    assert seen == ['scope']\n"
        )
    result = docker_pytest_result(dockerfile, probe, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
