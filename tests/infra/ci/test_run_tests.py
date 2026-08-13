# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from infra.ci.run_tests import PackageSelection, PytestInvocation, omit_locally_excluded_modules


def test_safe_invocation_omits_wholly_excluded_modules(tmp_path: Path) -> None:
    cluster_test = tmp_path / "test_cluster.py"
    cluster_test.write_text(
        "import pytest\n\npytestmark = pytest.mark.cluster\n\ndef test_cluster():\n    assert True\n"
    )
    unit_test = tmp_path / "test_unit.py"
    unit_test.write_text("def test_unit():\n    assert True\n")
    invocation = PytestInvocation(
        python="3.12",
        extras=(),
        pytest_args=(),
        packages=(
            PackageSelection(
                label="marin",
                test_paths=(cluster_test.name, unit_test.name),
                source_build=False,
            ),
        ),
    )

    filtered = omit_locally_excluded_modules(invocation, tmp_path)

    assert filtered is not None
    assert filtered.test_paths == (unit_test.name,)
