# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private script resources remain pinned and confined to their staging root."""

import hashlib

import pytest

from taskcompendium.models import VerifierKind
from taskcompendium.verifiers.script import (
    PrivateResource,
    ScriptVerifier,
    embedded_resource,
    materialize_private_resources,
    script_verifier,
)


def test_script_contract_round_trip_stages_private_files(tmp_path):
    script = embedded_resource("grader/check.py", b"#!/usr/bin/env python3\nprint('ok')\n", executable=True)
    answer = embedded_resource("fixtures/answer.bin", b"private\x00answer")
    config = ScriptVerifier(
        entrypoint=script.path,
        args=("--fixture", "/tests/fixtures/answer.bin"),
        timeout_seconds=30,
        runtime_image=f"example.test/grader@sha256:{'a' * 64}",
        resources=(script, answer),
    )

    specification = script_verifier(config)
    restored = ScriptVerifier.model_validate_json(specification.parameters_json)
    staged = materialize_private_resources(restored.resources, tmp_path / "tests")

    assert specification.kind == VerifierKind.SCRIPT
    assert {path: file.read_bytes() for path, file in staged.items()} == {
        "grader/check.py": b"#!/usr/bin/env python3\nprint('ok')\n",
        "fixtures/answer.bin": b"private\x00answer",
    }
    assert staged["grader/check.py"].stat().st_mode & 0o777 == 0o700
    assert staged["fixtures/answer.bin"].stat().st_mode & 0o777 == 0o600


def test_uri_digest_mismatch_leaves_no_staged_files(tmp_path):
    script = embedded_resource("grader.py", b"#!/usr/bin/env python3\n")
    reference = PrivateResource(
        path="answer.txt", sha256=hashlib.sha256(b"expected").hexdigest(), uri="gs://test/answer"
    )
    destination = tmp_path / "tests"

    with pytest.raises(ValueError, match="digest mismatch"):
        materialize_private_resources((script, reference), destination, lambda _uri: b"tampered")

    assert not destination.exists()


def test_private_resource_path_cannot_escape_staging_root(tmp_path):
    with pytest.raises(ValueError, match="relative to /tests"):
        embedded_resource("../agent/answer.txt", b"secret")

    script = embedded_resource("grader/check.py", b"private")
    destination = tmp_path / "tests"
    destination.mkdir()
    (destination / "grader").symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(ValueError, match="Unsafe private resource parent"):
        materialize_private_resources((script,), destination)

    assert not (tmp_path / "check.py").exists()
