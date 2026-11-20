# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench V2.1 repair conversion: declared bases, task-start setup, family config."""

import json

import pytest
import tomlkit

from experiments.post_training.task_curation.datasets.swe_rebench_v2 import groups
from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    _flatten,
    admission_receipt_labels,
    dockerfile,
    family_config,
    row_base,
    setup_command,
    task_toml,
)

PYTHON_ROW = {
    "instance_id": "0b01001001__spectree-64",
    "language": "python",
    "repo": "0b01001001/spectree",
    "base_commit": "a091fab020ac26548250c907bae0855273a98778",
    "workdir": "/spectree",
    "install_config": {
        "base_image_name": None,
        "install": ["pip install -e .[flask,falcon,starlette]"],
        "log_parser": "parse_log_pytest",
        "test_cmd": "cd /spectree && conda activate testbed || conda activate base; pytest",
    },
}


def test_group_covers_every_declared_base():
    """Every declared V2.1 base resolves to exactly one consolidated group."""
    declared = {
        None,
        "python_base_310",
        "python_base_37",
        "go_1.18.10", "go_1.19.13", "go_1.20.14", "go_1.21.13", "go_1.22.12", "go_1.23.8",
        "node_16", "node_20",
        "rust_1.77", "rust_1.78", "rust_1.79", "rust_1.81", "rust_1.83", "rust_1.84",
        "java_11", "java_17", "java_21", "kotlin-jdk-11", "kotlin-jdk-21", "scala_base", "clojure_base",
        "php_8.3.16", "php:8.3.16",
        "julia_base", "elixir_base", "swift_base",
        "c", "csharp_base", "dart_base", "r_base", "ocaml_base", "lua_base",
    }
    assert groups.GROUP_BY_BASE.keys() >= {base for base in declared if base is not None}
    assert "v1-python" in groups.GROUP_BY_BASE
    # The snapshot budget is ten shared bases.
    assert len(groups.GROUPS) == 10


def test_row_base_declared_and_v1_fallback():
    assert row_base(PYTHON_ROW) == "v1-python"
    declared = {**PYTHON_ROW, "install_config": {**PYTHON_ROW["install_config"], "base_image_name": "go_1.19.13"}}
    assert row_base(declared) == "go_1.19.13"
    non_python = {**PYTHON_ROW, "language": "rust", "install_config": {"base_image_name": None}}
    with pytest.raises(ValueError):
        row_base(non_python)


def test_python_setup_clones_declared_env_and_runs_installs_in_checkout():
    flat = _flatten(PYTHON_ROW)
    group = groups.GROUP_BY_BASE["v1-python"]
    setup = setup_command(flat, "v1-python", group)
    # Environment preparation precedes the clone; the clone block ends inside the workdir;
    # the install commands run inside the activated testbed env with their upstream wrappers.
    create, clone, activate, install = setup.splitlines()[0], None, None, None
    assert create.startswith("conda env remove -n testbed")
    # Unlabeled (v1-python) rows build on python 3.9 upstream (spectree 3.9.21, clique 3.9.20).
    assert "conda create -n testbed --clone py39" in setup.splitlines()[1]
    assert "cd / && rm -rf /spectree && mkdir -p /spectree" in setup
    assert "git clone -o origin https://github.com/0b01001001/spectree /spectree" in setup
    assert "git reset --hard a091fab020ac26548250c907bae0855273a98778" in setup
    assert "git remote remove origin" in setup
    assert "conda activate testbed" in setup
    assert "( pip install -e .[flask,falcon,starlette] ) || true" in setup
    assert setup.index("conda create") < setup.index("git clone") < setup.index("pip install")


def test_python_setup_clones_labeled_env():
    """python_base_310/37 rows clone their declared interpreter (3.10/3.7 upstream)."""
    flat = _flatten(PYTHON_ROW)
    group = groups.GROUP_BY_BASE["v1-python"]
    for declared, env_name in (("python_base_310", "py310"), ("python_base_37", "py37")):
        setup = setup_command(flat, declared, group)
        assert f"conda create -n testbed --clone {env_name}" in setup.splitlines()[1]


def test_go_setup_selects_declared_toolchain():
    flat = _flatten(
        {**PYTHON_ROW, "language": "go", "repo": "0xPolygonHermez/zkevm-node", "workdir": "/zkevm-node",
         "install_config": {"base_image_name": "go_1.19.13", "install": ["go mod download"],
                            "log_parser": "parse_log_go", "test_cmd": "go test ./..."}}
    )
    setup = setup_command(flat, "go_1.19.13", groups.GROUP_BY_BASE["go_1.19.13"])
    assert setup.startswith("ln -sfn /opt/toolchains/go/1.19.13 /usr/local/go\n")
    assert setup.index("ln -sfn") < setup.index("git clone") < setup.index("go mod download")


def test_rust_setup_pins_declared_toolchain_offline():
    flat = _flatten(
        {**PYTHON_ROW, "language": "rust",
         "install_config": {"base_image_name": "rust_1.77", "install": ["cargo test"], "log_parser": "parse_log_rust",
                            "test_cmd": "cargo test"}}
    )
    setup = setup_command(flat, "rust_1.77", groups.GROUP_BY_BASE["rust_1.77"])
    assert setup.startswith("rustup default 1.77\n")


def test_family_config_keeps_upstream_grader_keys_verbatim():
    flat = _flatten({**PYTHON_ROW, "FAIL_TO_PASS": ["tests/test_utils.py::test_a"], "PASS_TO_PASS": ["tests/test_utils.py::test_b"],
                     "test_files": ["tests/common.py"], "test_patch": "diff --git a/tests/common.py"})
    config = family_config(flat)
    # The shared-source machinery reads test_ids; the upstream grade.py reads the verbatim split.
    assert config["test_ids"] == config["FAIL_TO_PASS"] + config["PASS_TO_PASS"]
    assert config["FAIL_TO_PASS"] == ["tests/test_utils.py::test_a"]
    assert config["workdir"] == "/spectree" and config["base_commit"].startswith("a091fab")
    assert config["log_parser"] == "parse_log_pytest" and config["language"] == "python"
    assert set(config) == {
        "workdir", "base_commit", "test_cmd", "test_files", "log_parser", "language",
        "test_ids", "FAIL_TO_PASS", "PASS_TO_PASS", "test_patch",
    }


def test_task_toml_carries_upstream_declarations_and_condensed_environment():
    flat = _flatten({**PYTHON_ROW, "tags": ["swe-rebench", "swe-rebench-v2.1"], "quality_flags": [], "env": []})
    setup = setup_command(flat, "v1-python", groups.GROUPS["python"])
    document = tomlkit.parse(task_toml(flat, "v1-python", groups.GROUPS["python"], setup))
    assert document["verifier"]["timeout_sec"] == 2400.0
    assert document["verifier"]["environment_mode"] == "shared"
    assert document["agent"]["timeout_sec"] == 3000.0
    assert document["environment"]["cpus"] == 6
    assert document["environment"]["memory_mb"] == 24576
    assert document["environment"]["storage_mb"] == 32768
    assert document["environment"]["allow_internet"] is True
    assert document["environment"]["workdir"] == "/spectree"
    assert "conda create -n testbed" in document["environment"]["setup"]["command"]
    assert "docker_image" not in document["environment"]


def test_dockerfile_is_group_shared():
    """The per-task Dockerfile differs only by the pinned group image, keeping one snapshot identity."""
    group = groups.GROUPS["python"]
    assert dockerfile(group, "/spectree", "ghcr.io/marin-community/rl-data-swe-rebench-v2@sha256:" + "0" * 64) == (
        "FROM ghcr.io/marin-community/rl-data-swe-rebench-v2@sha256:" + "0" * 64 + "\nWORKDIR /spectree\n"
    )


def test_admission_receipt_labels_carry_upstream_image_defect(tmp_path):
    """A flip row (NoP 0/0 both sides; oracle passes on ours, fails on the upstream image)
    admits WITH the typed upstream_image_defect receipt + evidence; unknown receipts fail."""
    from rigging.filesystem.storage_path import StoragePath

    receipts = {
        "alecstrong__sql-psi-365": {
            "receipt": "upstream_image_defect",
            "evidence": {
                "upstream_nop_reward": 0.0,
                "condensed_nop_reward": 0.0,
                "upstream_oracle_reward": 0.0,
                "condensed_oracle_reward": 1.0,
                "upstream_oracle_evidence": "gradle wrapper distribution download failed",
            },
        },
    }
    staged = tmp_path / "admission-receipts.json"
    staged.write_text(json.dumps(receipts))

    labels = admission_receipt_labels("alecstrong__sql-psi-365", StoragePath(str(staged)))
    assert labels["swe_rebench_admission_receipt"] == "upstream_image_defect"
    evidence = json.loads(labels["swe_rebench_admission_receipt_evidence"])
    assert evidence["upstream_oracle_reward"] == 0.0
    assert evidence["condensed_oracle_reward"] == 1.0
    # No staged input, or an unlisted row: no receipt labels at all.
    assert admission_receipt_labels("alecstrong__sql-psi-365", None) == {}
    assert admission_receipt_labels("somebody__else-1", StoragePath(str(staged))) == {}
    # A malformed receipt value must fail fast, not silently admit.
    bad = tmp_path / "bad-receipts.json"
    bad.write_text(json.dumps({"x": {"receipt": "looks_fine_to_me"}}))
    with pytest.raises(ValueError):
        admission_receipt_labels("x", StoragePath(str(bad)))


def test_grader_scores_empty_ids_with_empty_log_as_unscorable():
    """Rows with NO test ids and an empty parsed log must not resolve (vacuous pass bug:
    a row whose grader scored 1.0 upstream vs 0.0 condensed from identical empty logs)."""
    import types

    from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream.task_files import (
        grader_source,
    )

    module = types.ModuleType("grade")
    exec(compile(grader_source("parse_log_pytest"), "grade.py", "exec"), module.__dict__)

    empty = {"FAIL_TO_PASS": [], "PASS_TO_PASS": []}
    report = module.grade(empty, "")
    assert report["unscorable"] is True
    assert report["resolved"] is False  # NOT a vacuous pass

    # A real run with no ids parses tests: still not resolved, but scorable evidence exists.
    parsed_log = "PASSED tests/test_a.py::test_one\nPASSED tests/test_a.py::test_two\n"
    report = module.grade(empty, parsed_log)
    assert report["unscorable"] is False
    assert report["resolved"] is False

    # Rows WITH ids keep the pre-existing verdict logic.
    with_ids = {"FAIL_TO_PASS": ["tests/test_a.py::test_one"], "PASS_TO_PASS": ["tests/test_a.py::test_two"]}
    report = module.grade(with_ids, parsed_log)
    assert report["unscorable"] is False
    assert report["resolved"] is True
