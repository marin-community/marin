# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What `uv build` produces for iris: a revision stamp, and an sdist that rebuilds it.

`hatch_build.py` stamps the revision date into every artifact, and an artifact
that reports no date makes the controller measure client freshness from
wall-clock time instead — so a lost stamp loosens the gate silently. Both
properties only exist once a build runs, so these tests run real builds.
"""

import ast
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
from iris.version import iris_tree_date

# Each test runs at least one real build, which needs longer than the 10s default.
pytestmark = pytest.mark.timeout(120)

IRIS_ROOT = Path(__file__).resolve().parents[1]
# The wheel maps src/iris onto iris/; the sdist ships the checkout layout as is.
WHEEL_BUILD_INFO = "iris/_build_info.py"
SDIST_BUILD_INFO = "src/iris/_build_info.py"


def uv_build(source: Path, out_dir: Path, *kinds: str) -> Path:
    """Build `source` into `out_dir` and return the single artifact produced."""
    subprocess.run(["uv", "build", *kinds, "--out-dir", str(out_dir), str(source)], check=True)
    # uv drops a .gitignore beside whatever it built.
    (artifact,) = out_dir.glob("marin_iris-*")
    return artifact


def build_date(source: str) -> str:
    """The date a generated `_build_info.py` assigns, given its source text."""
    (assignment,) = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)]
    return ast.literal_eval(assignment.value)


def wheel_build_date(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        return build_date(archive.read(WHEEL_BUILD_INFO).decode())


def sdist_build_date(sdist: Path) -> str:
    with tarfile.open(sdist) as archive:
        member = archive.extractfile(f"{sdist.name.removesuffix('.tar.gz')}/{SDIST_BUILD_INFO}")
        assert member is not None
        return build_date(member.read().decode())


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return uv_build(IRIS_ROOT, tmp_path_factory.mktemp("wheel"), "--wheel")


@pytest.fixture(scope="module")
def sdist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return uv_build(IRIS_ROOT, tmp_path_factory.mktemp("sdist"), "--sdist")


def test_wheel_and_sdist_report_the_date_of_the_iris_tree(wheel: Path, sdist: Path) -> None:
    expected = iris_tree_date(IRIS_ROOT)
    assert (wheel_build_date(wheel), sdist_build_date(sdist)) == (expected, expected)


def test_a_wheel_built_from_the_sdist_matches_the_one_built_from_the_checkout(
    wheel: Path,
    sdist: Path,
    tmp_path: Path,
) -> None:
    """A source install builds from the sdist, outside any checkout, and must not lose anything.

    An sdist laid out so the wheel target matches nothing still builds — it just
    yields a wheel with no Python modules in it — so compare the contents.
    """
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path / "unpacked", filter="data")
    (unpacked,) = (tmp_path / "unpacked").iterdir()
    rebuilt = uv_build(unpacked, tmp_path / "out", "--wheel")

    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(rebuilt) as from_sdist:
        assert sorted(from_sdist.namelist()) == sorted(original.namelist())
    # The rebuild has no repository to read, so the sdist's own stamp is the only date left.
    assert wheel_build_date(rebuilt) == iris_tree_date(IRIS_ROOT)
