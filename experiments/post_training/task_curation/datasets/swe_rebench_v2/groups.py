# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ten consolidated SWE-rebench V2.1 environment groups.

Each group is one shared base image (one Daytona snapshot identity): the upstream
per-language recipes merged with per-version toolchains installed side by side, the
declared interpreter-provisioning step, and build-time guards. A task's group comes
from its declared ``install_config.base_image_name`` (joined from nebius/SWE-rebench-V2
for the V2-origin rows; the V1-python rows have no declared base).

Grouping ruling (coordinator, 2026-11-17T16:00Z): option (b), ~10 shared bases with
per-version toolchains, fidelity-gated per group against the upstream digest-pinned
images before the full run.
"""

import json
from functools import cache

from rigging.filesystem.storage_path import StoragePath

BASE_MAP_INPUT = "base_map"
BASE_MAP_FILENAME = "swe-rebench-v21-base-map.json"
IMAGE_BINDINGS_INPUT = "base_images"
IMAGE_BINDINGS_FILENAME = "base-image-bindings.json"

UPSTREAM_DATASET = "ibragim-bad/SWE-rebench-V2.1"
UPSTREAM_REVISION = "26c0145ded93168b0d5d095420155901ad2d5a2f"
UPSTREAM_BASE_MAP_SOURCE = ("nebius/SWE-rebench-V2", "d7608f017a29ab256400ad4b73d5248d97204966")


class Group:
    """One consolidated base environment: its declared bases and their toolchain selection."""

    def __init__(self, group_id: str, bases: tuple[str, ...], toolchain: str | None = None):
        self.group_id = group_id
        self.bases = frozenset(bases)
        # How the per-task setup selects the row's declared toolchain version inside the
        # shared base; None when the base carries a single environment.
        self.toolchain = toolchain


GROUPS = {
    "python": Group(
        "python",
        ("v1-python", "python_base_310", "python_base_37"),
        # Conda env selection: the base pre-creates py310/py37; the setup clones the row's
        # declared env into `testbed` before running the install commands.
        toolchain="conda",
    ),
    "go": Group(
        "go",
        ("go_1.18.10", "go_1.19.13", "go_1.20.14", "go_1.21.13", "go_1.22.12", "go_1.23.8"),
        # Side-by-side Go toolchains under /opt/toolchains/go/<version>; the setup symlinks
        # the row's declared version into /usr/local/go.
        toolchain="go",
    ),
    "node": Group("node", ("node_16", "node_20"), toolchain="node"),
    "rust": Group(
        "rust",
        ("rust_1.77", "rust_1.78", "rust_1.79", "rust_1.81", "rust_1.83", "rust_1.84"),
        # rustup toolchains preinstalled; the setup pins the row's declared default.
        toolchain="rustup",
    ),
    "jvm": Group(
        "jvm",
        (
            "java_11",
            "java_17",
            "java_21",
            "kotlin-jdk-11",
            "kotlin-jdk-21",
            "scala_base",
            "clojure_base",
        ),
        # Side-by-side JDKs under /opt/toolchains/jdk/<version>; the setup selects JAVA_HOME.
        toolchain="jdk",
    ),
    "php": Group("php", ("php_8.3.16", "php:8.3.16")),
    "julia": Group("julia", ("julia_base",)),
    "elixir": Group("elixir", ("elixir_base",)),
    "swift": Group("swift", ("swift_base",)),
    "minor": Group(
        "minor",
        ("c", "csharp_base", "dart_base", "r_base", "ocaml_base", "lua_base"),
    ),
}

GROUP_BY_BASE = {base: group for group in GROUPS.values() for base in group.bases}

# The V1-python rows declare no base; they group under python with the base python the
# fidelity gate measures against their digest-pinned images.
V1_PYTHON_BASE = "v1-python"


@cache
def base_map(path: StoragePath, dataset: str, revision: str) -> dict[str, str]:
    """Read the immutable instance_id -> declared base map once per converter process."""
    manifest = json.loads(path.read_text())
    if manifest["dataset"] != dataset or manifest["revision"] != revision:
        raise ValueError("Base map does not cover this exact source pin")
    entries = manifest["entries"]
    if len(entries) != manifest["rows"]:
        raise ValueError("Base map row count differs from its manifest")
    return {instance_id: entry["base"] for instance_id, entry in entries.items()}


def row_base(row_data: dict, base_map: dict[str, str]) -> str:
    """The declared base for one row: the join for V2-origin rows, V1-python otherwise."""
    instance_id = row_data["instance_id"]
    declared = base_map.get(instance_id)
    if declared is not None:
        return declared
    if row_data.get("language") == "python":
        return V1_PYTHON_BASE
    raise ValueError(f"Row {instance_id} has no declared base and is not a V1-python row")


def row_group(row_data: dict, base_map: dict[str, str]) -> str:
    base = row_base(row_data, base_map)
    group = GROUP_BY_BASE.get(base)
    if group is None:
        raise ValueError(f"Declared base {base!r} is not covered by a consolidated group")
    return group.group_id
