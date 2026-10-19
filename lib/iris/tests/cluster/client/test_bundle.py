# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for workspace bundling."""

import io
import os
import re
from unittest.mock import patch

import pytest
from iris.cluster.backends.k8s import bundle_fetch
from iris.cluster.bundle import BundleStore
from iris.cluster.client.bundle import MAX_BUNDLE_SIZE_BYTES, collect_workspace_files, create_workspace_zip


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"cached")
    return tmp_path


def _mock_git_none():
    return patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=None)


def _mock_git_files(workspace, *rel_paths):
    files = {workspace / p for p in rel_paths}
    return patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=files)


def test_collect_falls_back_when_git_unavailable(workspace):
    with _mock_git_none():
        files = collect_workspace_files(workspace)
    rel = {str(f.relative_to(workspace)) for f in files}
    assert "pyproject.toml" in rel
    assert "src/main.py" in rel
    assert not any("__pycache__" in r for r in rel)


def test_collect_includes_generated_proto_files(workspace):
    rpc_dir = workspace / "src" / "iris" / "rpc"
    rpc_dir.mkdir(parents=True)
    (rpc_dir / "job_pb2.py").write_text("# generated")
    (rpc_dir / "job_pb2.pyi").write_text("# generated")
    (rpc_dir / "controller_connect.py").write_text("# generated")

    with _mock_git_files(workspace, "pyproject.toml", "src/main.py"):
        files = collect_workspace_files(workspace)
    rel = {str(f.relative_to(workspace)) for f in files}
    assert "src/iris/rpc/job_pb2.py" in rel
    assert "src/iris/rpc/job_pb2.pyi" in rel
    assert "src/iris/rpc/controller_connect.py" in rel


def test_collect_adds_caller_extra_includes(workspace):
    # a gitignored build dir the caller needs at runtime (e.g. a bundled frontend dist)
    build = workspace / "build"
    build.mkdir()
    (build / "app.js").write_text("// built")

    with _mock_git_files(workspace, "pyproject.toml"):
        without = {str(f.relative_to(workspace)) for f in collect_workspace_files(workspace)}
        with_hook = {
            str(f.relative_to(workspace)) for f in collect_workspace_files(workspace, extra_includes=["build/**/*"])
        }
    assert "build/app.js" not in without  # gitignored → not bundled by default
    assert "build/app.js" in with_hook  # caller opts it back in


def test_collect_respects_extra_exclude(workspace):
    (workspace / "data").mkdir()
    (workspace / "data" / "big.csv").write_text("1,2,3")

    with _mock_git_none():
        files = collect_workspace_files(workspace, exclude=re.compile(r"^data(/|$)"))
    rel = {str(f.relative_to(workspace)) for f in files}
    assert "data/big.csv" not in rel
    assert "pyproject.toml" in rel


def test_extra_exclude_is_matched_literally_not_verbose(workspace):
    # DEFAULT_EXCLUDE is verbose; the merge must not extend re.VERBOSE onto the
    # caller's pattern, or literal whitespace in it would be dropped and match
    # the wrong path.
    (workspace / "my dir").mkdir()
    (workspace / "my dir" / "keep.txt").write_text("hi")
    (workspace / "mydir").mkdir()
    (workspace / "mydir" / "drop.txt").write_text("hi")

    with _mock_git_none():
        files = collect_workspace_files(workspace, exclude=re.compile(r"^mydir/"))
    rel = {str(f.relative_to(workspace)) for f in files}
    assert "mydir/drop.txt" not in rel
    assert "my dir/keep.txt" in rel


@pytest.mark.timeout(60)
def test_zip_rejects_oversized_bundles(workspace):
    large_file = workspace / "large_file.bin"
    large_file.write_bytes(os.urandom(MAX_BUNDLE_SIZE_BYTES + 1024 * 1024))

    with _mock_git_none():
        with pytest.raises(ValueError, match=r"Bundle size .* exceeds maximum"):
            create_workspace_zip(workspace)


@pytest.fixture(scope="module")
def vendor_bundle(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("vendor-workspace")
    resources = {
        "vendor/evalchemy/eval/chat_benchmarks/fixture/data/questions.bin": os.urandom(26 * 1024 * 1024),
        "vendor/evalchemy/eval/chat_benchmarks/fixture/eval_instruct.py": b"def iter_specs(): return ()\n",
        "vendor/evalchemy/eval/chat_benchmarks/fixture/vendor/scorer/tests/cases.json": b"[1, 2, 3]",
        "vendor/other_package/src/library/__init__.py": b"VALUE = 1\n",
        "vendor/other_package/src/library/data/questions.json": b"[4, 5, 6]",
    }
    for name, contents in resources.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    excluded = ("vendor/evalchemy/tests/fixture.bin", "vendor/other_package/tests/fixture.bin")
    for name in excluded:
        fixture = workspace / name
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"not needed at runtime")
    return create_workspace_zip(workspace), resources, excluded


@pytest.mark.parametrize("consumer", ["worker", "kubernetes"])
@pytest.mark.timeout(60)
def test_large_vendor_bundle_round_trips_runtime_resources(vendor_bundle, tmp_path, monkeypatch, consumer):
    bundle, resources, excluded = vendor_bundle
    store = BundleStore(str(tmp_path / "bundles"))
    bundle_id = store.write(bundle)
    target = tmp_path / "workdir"
    target.mkdir()
    if consumer == "worker":
        store.extract_bundle_to(bundle_id, target)
    else:

        class BundleResponse(io.BytesIO):
            def getheader(self, _name):
                return bundle_id

        monkeypatch.setattr(
            bundle_fetch.urllib.request,
            "urlopen",
            lambda _request, timeout: BundleResponse(bundle),
        )
        bundle_fetch.fetch_bundle("http://controller", bundle_id, str(target))

    assert {name: (target / name).read_bytes() for name in resources} == resources
    assert all(not (target / name).exists() for name in excluded)
