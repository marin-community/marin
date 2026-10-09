# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.harbor.export import VerifierPayloadIdentity, archive_bytes


@pytest.mark.parametrize(
    "path,content,mode",
    [
        ("tests/test.sh", b"new wrapper", "755"),
        ("tests/Dockerfile", b"FROM different:tag", "644"),
        ("tests/helper.py", b"private checker", "755"),
        ("task.toml", b"new dispatch", "644"),
    ],
)
def test_verifier_identity_tracks_payload_recipe_and_modes(path, content, mode):
    files = {
        "tests/test.sh": b"original wrapper",
        "tests/Dockerfile": b"FROM source:tag",
        "tests/helper.py": b"private checker",
        "task.toml": b"dispatch",
        "instruction.md": b"public prompt",
    }
    original = VerifierPayloadIdentity()
    original.add(archive_bytes(files, {}))
    changed = VerifierPayloadIdentity()
    changed.add(archive_bytes({**files, path: content}, {path: mode}))
    assert changed.ref != original.ref
    equivalent = VerifierPayloadIdentity()
    equivalent.add(archive_bytes({**files, "instruction.md": b"different public prompt"}, {}))
    equivalent.add(archive_bytes(files, {}))
    assert equivalent.ref == original.ref
