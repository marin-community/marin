# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import stat

from infra.auth.staging import prepare, private_directory


def test_prepare_preserves_deployment_credentials_and_limits_file_access(tmp_path):
    directory = private_directory(tmp_path / "credentials")
    prepare(directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert len({original[f"{service}.key"] for service in ("openbao", "openfga", "broker")}) == 3
    prepare(directory)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.iterdir())
