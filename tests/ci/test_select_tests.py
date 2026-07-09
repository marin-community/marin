# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the import-driven test selector (infra/ci/select_tests.py)."""

from infra.ci.select_tests import all_test_files, classify, is_test_module


def test_all_test_files_selects_only_pytest_collectable_modules(tmp_path):
    """Helper modules under tests/ (workload scripts, stubs, generators) must never be
    handed to pytest explicitly -- an explicit path is imported even when it does not match
    the collection convention, crashing the lane when the helper's deps are absent."""
    test_dir = tmp_path / "lib" / "iris" / "tests"
    test_dir.mkdir(parents=True)
    (test_dir / "test_client.py").touch()
    (test_dir / "actor_test.py").touch()
    (test_dir / "gang_jax_smoke_workload.py").touch()
    (test_dir / "conftest.py").touch()

    names = [p.name for p in all_test_files("iris", tmp_path)]
    assert names == ["actor_test.py", "test_client.py"]


def test_changed_helper_module_forces_full_scope(tmp_path):
    """A changed non-collectable .py under tests/ runs the full scope (its dependents),
    not the file itself; a changed test module still runs directly."""
    result = classify(
        ["lib/iris/tests/e2e/gang_jax_smoke_workload.py", "lib/iris/tests/cluster/test_types.py"],
        tmp_path,
    )
    assert result.direct_tests == {"iris": ["lib/iris/tests/cluster/test_types.py"]}
    assert result.forced == {"iris"}


def test_is_test_module_matches_pytest_defaults():
    assert is_test_module("test_client.py")
    assert is_test_module("gpt2_test.py")
    assert not is_test_module("conftest.py")
    assert not is_test_module("openai_stub.py")
    assert not is_test_module("test_data.json")
