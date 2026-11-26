# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench V2.1 repair conversion and the two assumed-admission declarations.

The repair converter tests mirror the Atlas repair lane's behavior tests at the
declared converter version; the declaration tests pin the swesmith retained cohort
and the repaired SWE-rebench source to their Atlas pins with their stage-0 license
receipts.
"""

import json
import re

import pytest
import tomlkit
from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.atlas_publications import (
    bind_publications,
)
from experiments.post_training.task_curation.datasets.swe_rebench_v2 import groups
from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    DECLARED_LIMITS_DEVIATION,
    UPSTREAM_CPUS,
    UPSTREAM_MEMORY_MB,
    UPSTREAM_STORAGE_MB,
    _flatten,
    admission_receipt_labels,
    dockerfile,
    family_config,
    row_base,
    setup_command,
    task_toml,
)
from experiments.post_training.task_curation.datasets.swe_rebench_v2.sources import sources as repair_sources
from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream.task_files import solve_sh
from experiments.post_training.task_curation.datasets.swe_rebench_v2.workspace import to_workspace
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
)
from experiments.post_training.task_curation.sources import all_sources

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
        "python_base_310",
        "python_base_37",
        "go_1.18.10",
        "go_1.19.13",
        "go_1.20.14",
        "go_1.21.13",
        "go_1.22.12",
        "go_1.23.8",
        "node_16",
        "node_20",
        "rust_1.77",
        "rust_1.78",
        "rust_1.79",
        "rust_1.81",
        "rust_1.83",
        "rust_1.84",
        "java_11",
        "java_17",
        "java_21",
        "kotlin-jdk-11",
        "kotlin-jdk-21",
        "scala_base",
        "clojure_base",
        "php_8.3.16",
        "php:8.3.16",
        "julia_base",
        "elixir_base",
        "swift_base",
        "c",
        "csharp_base",
        "dart_base",
        "r_base",
        "ocaml_base",
        "lua_base",
    }
    assert groups.GROUP_BY_BASE.keys() >= declared
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
    create = setup.splitlines()[0]
    assert create.startswith("conda env remove -n testbed")
    # Unlabeled (v1-python) rows build on python 3.9 upstream (spectree 3.9.21, clique 3.9.20).
    assert "conda create -n testbed --clone py39" in setup.splitlines()[1]
    assert "cd / && rm -rf /testbed && mkdir -p /testbed" in setup
    assert "git clone -o origin https://github.com/0b01001001/spectree /testbed" in setup
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
        {
            **PYTHON_ROW,
            "language": "go",
            "repo": "0xPolygonHermez/zkevm-node",
            "workdir": "/zkevm-node",
            "install_config": {
                "base_image_name": "go_1.19.13",
                "install": ["go mod download"],
                "log_parser": "parse_log_go",
                "test_cmd": "go test ./...",
            },
        }
    )
    setup = setup_command(flat, "go_1.19.13", groups.GROUP_BY_BASE["go_1.19.13"])
    assert setup.startswith("ln -sfn /opt/toolchains/go/1.19.13 /usr/local/go\n")
    assert setup.index("ln -sfn") < setup.index("git clone") < setup.index("go mod download")


def test_rust_setup_pins_declared_toolchain_offline():
    flat = _flatten(
        {
            **PYTHON_ROW,
            "language": "rust",
            "install_config": {
                "base_image_name": "rust_1.77",
                "install": ["cargo test"],
                "log_parser": "parse_log_rust",
                "test_cmd": "cargo test",
            },
        }
    )
    setup = setup_command(flat, "rust_1.77", groups.GROUP_BY_BASE["rust_1.77"])
    assert setup.startswith("rustup default 1.77\n")


def test_family_config_keeps_upstream_grader_keys_verbatim():
    flat = _flatten(
        {
            **PYTHON_ROW,
            "FAIL_TO_PASS": ["tests/test_utils.py::test_a"],
            "PASS_TO_PASS": ["tests/test_utils.py::test_b"],
            "test_files": ["tests/common.py"],
            "test_patch": "diff --git a/tests/common.py",
        }
    )
    config = family_config(flat)
    # The shared-source machinery reads test_ids; the upstream grade.py reads the verbatim split.
    assert config["test_ids"] == config["FAIL_TO_PASS"] + config["PASS_TO_PASS"]
    assert config["FAIL_TO_PASS"] == ["tests/test_utils.py::test_a"]
    assert config["workdir"] == "/testbed" and config["base_commit"].startswith("a091fab")
    # The row's per-repo path is rewritten inside the test command too.
    assert config["test_cmd"] == "cd /testbed && conda activate testbed || conda activate base; pytest"
    assert config["log_parser"] == "parse_log_pytest" and config["language"] == "python"
    assert set(config) == {
        "workdir",
        "base_commit",
        "test_cmd",
        "test_files",
        "log_parser",
        "language",
        "test_ids",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "test_patch",
    }


def test_task_toml_carries_approved_reduced_limits_and_upstream_source_declaration():
    """The user decision (2026-11-25) reduces the emitted execution limits to Daytona's
    per-sandbox cap; the upstream 6/24/32 declarations stay recorded as the source
    declaration with the typed deviation record."""
    flat = _flatten({**PYTHON_ROW, "tags": ["swe-rebench", "swe-rebench-v2.1"], "quality_flags": [], "env": []})
    setup = setup_command(flat, "v1-python", groups.GROUPS["python"])
    document = tomlkit.parse(task_toml(flat, "v1-python", groups.GROUPS["python"], setup))
    assert document["verifier"]["timeout_sec"] == 2400.0
    assert document["verifier"]["environment_mode"] == "shared"
    assert document["agent"]["timeout_sec"] == 3000.0
    # The emitted limits: Daytona's per-sandbox cap (the approved user decision).
    assert document["environment"]["cpus"] == 4
    assert document["environment"]["memory_mb"] == 8192
    assert document["environment"]["storage_mb"] == 10240
    assert document["environment"]["setup"]["timeout_sec"] == 3600.0
    assert document["environment"]["allow_internet"] is True
    assert document["environment"]["workdir"] == "/testbed"
    assert "conda create -n testbed" in document["environment"]["setup"]["command"]
    assert "docker_image" not in document["environment"]
    # The upstream values stay recorded as the source declaration, never emitted.
    assert (UPSTREAM_CPUS, UPSTREAM_MEMORY_MB, UPSTREAM_STORAGE_MB) == (6, 24576, 32768)
    assert DECLARED_LIMITS_DEVIATION["type"] == "declared_limits_reduction"
    assert DECLARED_LIMITS_DEVIATION["reason"] == "upstream spec exceeds the Daytona 4/8/10 per-sandbox cap"
    assert "fidelity gate" in DECLARED_LIMITS_DEVIATION["evidence"]
    assert DECLARED_LIMITS_DEVIATION["upstream_declaration"]["cpus"] == 6
    assert DECLARED_LIMITS_DEVIATION["declared_limits"]["cpus"] == 4


def test_dockerfile_is_group_shared():
    """The per-task Dockerfile differs only by the pinned group image, keeping one snapshot identity."""
    group = groups.GROUPS["python"]
    assert dockerfile(group, "ghcr.io/marin-community/rl-data-swe-rebench-v2@sha256:" + "0" * 64) == (
        "FROM ghcr.io/marin-community/rl-data-swe-rebench-v2@sha256:" + "0" * 64 + "\n"
    )


def test_verifier_and_setup_do_not_rely_on_image_default_cwd():
    """Dropping the Dockerfile WORKDIR cannot change grading: the verifier and the setup must
    run from task.toml environment.workdir by their own explicit cds, never the image's
    default cwd."""
    flat = _flatten(PYTHON_ROW)
    setup = setup_command(flat, "v1-python", groups.GROUP_BY_BASE["v1-python"])
    # the setup's workspace steps are explicit: the clone guard cds to /, then into the workspace
    assert "cd / && rm -rf /testbed && mkdir -p /testbed" in setup
    assert "cd /testbed" in setup
    # the Dockerfile carries no WORKDIR: the group's one snapshot identity
    group = groups.GROUPS["python"]
    image = "ghcr.io/marin-community/rl-data-swe-rebench-v2@sha256:" + "0" * 64
    assert "WORKDIR" not in dockerfile(group, image)


def test_admission_receipt_labels_carry_upstream_image_defect(tmp_path):
    """A flip row (NoP 0/0 both sides; oracle passes on ours, fails on the upstream image)
    admits WITH the typed upstream_image_defect receipt + evidence; unknown receipts fail."""
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
    staged = tmp_path / "staged" / "admission_receipts"
    staged.mkdir(parents=True)
    (staged / "admission-receipts.json").write_text(json.dumps(receipts))

    labels = admission_receipt_labels("alecstrong__sql-psi-365", StoragePath(str(staged)))
    assert labels["swe_rebench_admission_receipt"] == "upstream_image_defect"
    evidence = json.loads(labels["swe_rebench_admission_receipt_evidence"])
    assert evidence["upstream_oracle_reward"] == 0.0
    assert evidence["condensed_oracle_reward"] == 1.0
    # No staged input, or an unlisted row: no receipt labels at all.
    assert admission_receipt_labels("alecstrong__sql-psi-365", None) == {}
    assert admission_receipt_labels("somebody__else-1", StoragePath(str(staged))) == {}
    # A malformed receipt value must fail fast, not silently admit.
    bad = tmp_path / "staged" / "bad_receipts"
    bad.mkdir(parents=True)
    (bad / "admission-receipts.json").write_text(json.dumps({"x": {"receipt": "looks_fine_to_me"}}))
    with pytest.raises(ValueError):
        admission_receipt_labels("x", StoragePath(str(bad)))


def test_every_task_uses_the_uniform_testbed_workspace():
    """Ruling 2026-11-20 (the 10-snapshot consolidation): every converted task's workdir is the
    uniform /testbed and no spec wiring references the row's per-repo absolute path — the
    shared-source image layer's workspace line is then constant and the declared Daytona
    snapshot identities collapse to the ten base images."""
    grader_keys = {
        "FAIL_TO_PASS": ["tests/test_utils.py::test_a"],
        "PASS_TO_PASS": ["tests/test_utils.py::test_b"],
        "test_files": ["tests/common.py"],
        "test_patch": "diff --git a/tests/common.py",
    }
    rows = [
        PYTHON_ROW,
        {
            **PYTHON_ROW,
            "language": "go",
            "repo": "0xPolygonHermez/zkevm-node",
            "workdir": "/zkevm-node",
            "install_config": {
                "base_image_name": "go_1.19.13",
                "install": ["go build ./..."],
                "log_parser": "parse_log_go",
                "test_cmd": "cd /zkevm-node && go test ./...",
            },
        },
        {
            **PYTHON_ROW,
            "repo": "tobymao/sqlglot",
            "workdir": "/sqlglot",
            "install_config": {
                "base_image_name": "python_base_310",
                "install": ["pip install -e /sqlglot"],
                "log_parser": "parse_log_pytest",
                "test_cmd": "pytest /sqlglot/tests",
            },
        },
    ]
    for row in (
        {**candidate, **grader_keys, "tags": ["swe-rebench", "swe-rebench-v2.1"], "quality_flags": [], "env": []}
        for candidate in rows
    ):
        flat = _flatten(row)
        old = flat["workdir"]
        base = row_base(row)
        group = groups.GROUP_BY_BASE[base]
        setup = setup_command(flat, base, group)
        config = family_config(flat)
        document = tomlkit.parse(task_toml(flat, base, group, setup))
        assert document["environment"]["workdir"] == "/testbed"
        assert config["workdir"] == "/testbed"
        assert f"git clone -o origin https://github.com/{row['repo']} /testbed" in setup
        # no spec wiring references the row's per-repo absolute path (path-token check: the
        # clone URL legitimately carries the repo name, which may equal the old workdir name)
        pattern = rf"(?<![0-9A-Za-z_/-]){re.escape(old)}(?![0-9A-Za-z_-])"
        assert re.search(pattern, setup) is None
        assert re.search(pattern, json.dumps(config)) is None
        assert re.search(pattern, tomlkit.dumps(document)) is None


def test_workspace_rewrites_respect_path_boundaries():
    """Sibling names and nested paths survive; exact occurrences rewrite to /testbed."""
    assert to_workspace("cd /sympy && pytest /sympy/tests", "/sympy") == "cd /testbed && pytest /testbed/tests"
    assert to_workspace("ls /sympy2 /sympy-bench", "/sympy") == "ls /sympy2 /sympy-bench"
    assert to_workspace("ls /x/sympy", "/sympy") == "ls /x/sympy"
    assert to_workspace("cd /testbed", "/testbed") == "cd /testbed"


def test_the_condensed_solve_script_moves_to_the_uniform_workspace():
    """Ruling 2026-11-20: the vendored solve script hard-codes @@WORKDIR@@ to the row's
    per-repo path; the condensed flow rewrites it to /testbed so the oracle's gold-patch
    delivery runs in the consolidated workspace. The upstream variant keeps the row's
    original paths (its digest image bakes the repo there)."""
    row = dict(PYTHON_ROW)
    original = solve_sh(row)
    assert "WORKDIR=/spectree" in original
    rewritten = to_workspace(original, row["workdir"])
    assert "WORKDIR=/testbed" in rewritten
    assert "/spectree" not in rewritten.replace("/x/spectree", "")
    # the upstream variant's own script is not rewritten
    assert "WORKDIR=/spectree" in solve_sh(row)


def test_repaired_source_declares_the_atlas_pin():
    """The repaired source's identity, pin, version, files and use match the Atlas
    declaration; its stage-0 license receipt resolves at that exact pin."""
    (source,) = repair_sources()
    assert source.info.id == "SWE-rebench:ibragim-bad/SWE-rebench-V2.1@repaired-20261119"
    assert source.name == "swe-rebench-v2-1-repaired"
    assert source.version == "2026.11.20.2"
    assert source.intended_use.value == "train"
    dataset = source.dataset
    assert dataset is not None
    assert dataset.name == "ibragim-bad/SWE-rebench-V2.1"
    assert dataset.revision == "26c0145ded93168b0d5d095420155901ad2d5a2f"
    assert source.files == tuple(f"data/train-{i:05d}-of-00006.parquet" for i in range(6))
    source.require_license()
    assert source.info.license_receipt is not None
    assert source.info.license_receipt.evidence[0].license_ids == ["cc-by-4.0"]


def test_swesmith_retained_cohort_declares_the_atlas_pin():
    """The remediated swesmith cohort binds the retained import at its exact Atlas pin,
    and its user-approved license exception passes the stage-0 gate."""
    source = all_sources()["TaskTrove:swesmith-verifier-repaired-v5"]
    assert source.name == "swesmith-verifier-repaired-v5-retained"
    assert source.version == "2026.11.13.1"
    dataset = source.dataset
    assert dataset is not None
    assert dataset.name == "open-athena/campaign-release-swesmith-verifier-repaired-v5-0bfcd894-retained-20261116-v1"
    assert dataset.revision == "758b28bf4cc0d0e770329f2944c9451fd578566a"
    assert source.files == ("tasks.parquet",)
    assert source.config is not None
    assert source.config.expected_rows == 7016
    assert source.intended_use.value == "train"
    source.require_license()
    assert source.info.license_receipt is not None
    assert any(
        decision.kind.value == "exception" and decision.approved for decision in source.info.license_receipt.decisions
    )


def test_binding_preserves_the_declaration_over_retirement():
    """An assumed-admission source survives bind_publications while other retired
    declarations stay dropped."""

    def _source(source_id: str) -> RlDataSource:
        origin, _, dataset = source_id.partition(":")
        return RlDataSource(
            info=SourceInfo(
                origin=SourceOrigin(origin),
                upstream_dataset=dataset,
                upstream_name=dataset,
                build_path="test",
                data_restriction=DataRestriction.OPEN,
            )
        )

    kept = bind_publications([_source("TaskTrove:swesmith-verifier-repaired-v5")])
    kept_ids = {source.info.id for source in kept}
    assert "TaskTrove:swesmith-verifier-repaired-v5" in kept_ids
    dropped = bind_publications([_source("ASDiv:asdiv")])
    dropped_ids = {source.info.id for source in dropped}
    assert "ASDiv:asdiv" not in dropped_ids
    assert "TaskTrove:swesmith-verifier-repaired-v5" not in dropped_ids
