# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench V2.1 repair conversion: declared bases, task-start setup, family config."""

import json
import re
import types
from dataclasses import replace

import pytest
import tomlkit
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.swe_rebench_v2 import groups
from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    DECLARED_LIMITS_DEVIATION,
    SETUP_TIMEOUT_SEC,
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
    to_workspace,
)
from experiments.post_training.task_curation.datasets.swe_rebench_v2.sources import _declare
from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream import task_files
from experiments.post_training.task_curation.datasets.swe_rebench_v2.upstream.task_files import (
    grader_source,
    solve_sh,
)
from experiments.post_training.task_curation.derived_contract_census import DerivedContractReceipt
from experiments.post_training.task_curation.derived_contract_tags import bind_derived_contract_tags
from experiments.post_training.task_curation.swe_rebench_fidelity_driver import (
    CHURN_THRESHOLD,
    CHURN_WINDOW,
    TRANSIENT_RETRY_DELAYS_SEC,
    ChurnMonitor,
    OutcomeLedger,
    fail_closed_reward,
    row_variant_files,
    transient_daytona_error,
    transient_daytona_text,
    verifier_timeout_text,
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


def test_verifier_and_setup_do_not_rely_on_image_default_cwd(tmp_path):
    """Dropping the Dockerfile WORKDIR cannot change grading: the verifier and the setup must
    run from task.toml environment.workdir by their own explicit cds, never the image's
    default cwd."""
    template = task_files._shell_template("test.sh")
    assert 'cd "$WORKDIR"' in template  # the grading cds into the workdir explicitly

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


def test_grader_scores_empty_ids_with_empty_log_as_unscorable():
    """Rows with NO test ids and an empty parsed log must not resolve (vacuous pass bug:
    a row whose grader scored 1.0 upstream vs 0.0 condensed from identical empty logs)."""
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


def _grade(log_parser: str, log: str, fail_to_pass: list[str], pass_to_pass: list[str]) -> dict:
    module = types.ModuleType("grade")
    exec(compile(grader_source(log_parser), "grade.py", "exec"), module.__dict__)
    return module.grade({"FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": pass_to_pass}, log)


def test_grader_repairs_the_interleaved_harness_echo():
    """The sandbox console capture interleaves the harness's own "test_cmd exit code" echo into
    per-test lines at chunk boundaries (DELTA5 zero-capture fixtures). The repair removes the
    marker and recovers a cut name only when exactly one configured id is consistent with the
    surviving prefix; any ambiguity fails closed."""

    # Duration cut mid-stream and the echo glued onto the same line: the real fixture is
    # swe-rebench-v2-1-repaired-04fbc0a6… ("creates a json file", jest).
    log = (
        "PASS src/generateGraphqlCodegen.test.ts\n"
        "  generateGraphqlCodegen\n"
        "    ✓ creates a json file (18test_cmd exit code: 0\n"
        "PASS src/connection.test.ts\n"
        "  connection\n"
        "    ✓ should parse single DATABASE_URL (2 ms)\n"
        "test_cmd exit code: 0\n"
    )
    report = _grade(
        "parse_log_js_4",
        log,
        fail_to_pass=[],
        pass_to_pass=["creates a json file", "should parse single DATABASE_URL"],
    )
    assert report["pass_to_pass"] == {"creates a json file": True, "should parse single DATABASE_URL": True}
    assert report["resolved"] is True

    # The echo glued straight after a complete name: real fixture a28b9b0a… ("is a singleton.").
    log = "    ✓ is a singleton.test_cmd exit code: 0\n"
    report = _grade("parse_log_js_4", log, fail_to_pass=[], pass_to_pass=["is a singleton."])
    assert report["pass_to_pass"] == {"is a singleton.": True}

    # A name cut mid-word by the glue recovers when exactly one configured id extends the
    # surviving prefix: real fixture 6823445d… (the callbacks.spec.ts suite).
    callbacks_id = (
        "src/lib/internal/helpers/tests/callbacks.spec.ts"
        " > executeCallbacks > groups callbacks and executes them together"
    )
    log = " ✓ src/lib/internal/helpers/tests/callbacks.spec.ts > executeCallbatest_cmd exit code: 0\n"
    report = _grade(
        "parse_log_js_4",
        log,
        fail_to_pass=[],
        pass_to_pass=[callbacks_id, "src/lib/internal/helpers/tests/other.spec.ts > unrelated test"],
    )
    assert report["pass_to_pass"][callbacks_id] is True

    # The same cut with MANY configured ids sharing the prefix stays fail-closed: real
    # fixture dd598fa4… (20 ids extend "src/rules/no-nth-methods.test.ts > no").
    log = " ✓ src/rules/no-nth-methods.test.ts > notest_cmd exit code: 0\n"
    ids = [
        "src/rules/no-nth-methods.test.ts > no-nth-methods > valid > page",
        "src/rules/no-nth-methods.test.ts > no-nth-methods > invalid > foo.first()",
        "src/rules/no-nth-methods.test.ts > no-nth-methods > invalid > foo.last()",
    ]
    report = _grade("parse_log_js_4", log, fail_to_pass=[], pass_to_pass=ids)
    assert all(value is False for value in report["pass_to_pass"].values())
    assert report["resolved"] is False

    # A shorter parsed name that is a natural prefix of an id never matches without the
    # truncation mark: real fixture 234f51e… ("should render" vs "should render correct
    # number of pages" are different tests).
    log = "    ✓ should render correct number of pages (5 ms)\n"
    report = _grade(
        "parse_log_js_4",
        log,
        fail_to_pass=[],
        pass_to_pass=["should render", "should render correct number of pages"],
    )
    assert report["pass_to_pass"] == {"should render": False, "should render correct number of pages": True}

    # A marker-only parsed payload (the name was cut entirely) never satisfies an id.
    log = " ✓ test_cmd exit code: 0\n"
    report = _grade("parse_log_js_4", log, fail_to_pass=[], pass_to_pass=["anything"])
    assert report["pass_to_pass"] == {"anything": False}


def test_grader_echo_repair_never_breaks_clean_matches():
    """The repair only removes the harness's own echo marker: clean logs grade exactly as
    before, and the truncation path never fires without the cut evidence."""
    log = "PASS src/a.test.ts\n" "  suite\n" "    ✓ handles (special) cases (5 ms)\n" "    ✓ fast test\n"
    report = _grade(
        "parse_log_js_4",
        log,
        fail_to_pass=["handles (special) cases"],
        pass_to_pass=["fast test"],
    )
    assert report["resolved"] is True
    assert report["fail_to_pass"] == {"handles (special) cases": True}
    assert report["pass_to_pass"] == {"fast test": True}

    # A complete duration on a marked line is still stripped, and the name stays recoverable:
    # the capture can cut the newline after "(5 ms)" and glue the echo to it.
    log = "    ✓ creates a json file (18 ms)test_cmd exit code: 0\n"
    report = _grade("parse_log_js_4", log, fail_to_pass=[], pass_to_pass=["creates a json file"])
    assert report["pass_to_pass"] == {"creates a json file": True}


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
            **grader_keys,
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
            **grader_keys,
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


def test_transient_daytona_classification_retries_infra_and_never_grading():
    """The wave resume survives the org CPU-limit saturation and reaped sandboxes: the
    infrastructure failures classify as retryable; grading verdicts and config errors do
    not (ruling 2026-11-20: the wave1-full post-mortem)."""
    assert TRANSIENT_RETRY_DELAYS_SEC == (60.0, 300.0)
    assert transient_daytona_error(RuntimeError("Daytona environment is not running"))
    assert transient_daytona_error(
        RuntimeError("Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: 2500.")
    )
    assert transient_daytona_error(RuntimeError("DaytonaConnectionError: connection reset"))
    assert transient_daytona_error(RuntimeError("Machine is closed"))
    assert transient_daytona_error(RuntimeError("DaytonaTimeoutError: sandbox start timed out"))
    # grading verdicts and config errors are never retried
    assert not transient_daytona_error(AssertionError("assert 0.0 == 1.0"))
    assert not transient_daytona_error(ValueError("Unexpected hardened setup command shape"))
    assert not transient_daytona_error(RuntimeError("UnsupportedMachineSpec: missing environment/Dockerfile"))
    # The machinery embeds provider failures in the summary instead of raising: the embedded
    # form classifies the same way (the wave-v9 churn recorded 8,000 reward=None summaries).
    assert transient_daytona_text(
        "DaytonaValidationError: Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: 2500."
    )
    assert transient_daytona_text("RuntimeError: Daytona environment is not running")
    assert not transient_daytona_text("ValueError: Unexpected hardened setup command shape")


def test_churn_monitor_trips_at_twenty_percent_and_stays_quiet_below():
    """Ruling 2026-11-20 (the wave-v9 churn): more than about 20% of the trials in the
    rolling window ending reward=None/infra trips the typed QUOTA/INFRA stop."""
    assert CHURN_WINDOW == 50 and CHURN_THRESHOLD == 0.2
    monitor = ChurnMonitor()
    for _ in range(CHURN_WINDOW):
        monitor.record(False)
    assert not monitor.tripped()
    # 10 failures in 50 (exactly 20%) does not trip; the 11th (22%) does, and the window
    # sliding back under 20% clears it
    for _ in range(10):
        monitor.record(True)
    assert not monitor.tripped()
    monitor.record(True)
    assert monitor.tripped()
    assert "11/50" in monitor.describe()
    # the 11 failures sit at the window's back: 39 clean records only push them forward;
    # the 40th slides the first one out and the monitor clears
    for _ in range(39):
        monitor.record(False)
    assert monitor.tripped()
    monitor.record(False)
    assert not monitor.tripped()


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
    assert 'cd "$WORKDIR"' in rewritten
    assert "/spectree" not in rewritten.replace("/x/spectree", "")
    # the upstream variant's own script is not rewritten
    assert "WORKDIR=/spectree" in solve_sh(row)


def test_declared_shape_tag_matches_the_measured_derived_contract():
    """The source shape must accept the measured contract; the old tag must fail."""
    source = next(s for s in _declare() if s.info.id == "SWE-rebench:ibragim-bad/SWE-rebench-V2.1@repaired-20261119")
    assert "single-turn" in source.info.tags and "multi-turn" not in source.info.tags
    receipt = DerivedContractReceipt.model_validate(
        {
            "source_id": source.info.id,
            "input": {
                "dataset_id": source.dataset.name,
                "dataset_revision": source.dataset.revision,
                "converter_version": source.version,
                "converter_revision": "0" * 40,
                "measured_tasks": 21272,
                "task_specs_sha256": "0" * 64,
                "evidence_url": "s3://proof/derived-contract",
            },
            "population_path": "s3://proof/population",
            "scanned_task_specs_sha256": "0" * 64,
            "evidence_url": "s3://proof/derived-contract",
            "summary": {
                "task_count": 21272,
                "grader_type_counts": {"none": 21272},
                "agentic_count": 21272,
                "single_turn_count": 21272,
                "reply_only_count": 0,
                "workspace_graded_count": 0,
                "judge_rubric_counts": {},
                "judge_reference_count": 0,
                "judge_no_reference_count": 0,
            },
        }
    )
    bound = bind_derived_contract_tags(source, receipt)
    assert bound.info.derived_contract is not None
    stale = replace(
        source,
        info=replace(
            source.info,
            tags=frozenset(
                {"agentic", "multi-turn", *(t for t in source.info.tags if t not in {"agentic", "single-turn"})}
            ),
        ),
    )
    with pytest.raises(ValueError, match="multi-turn"):
        bind_derived_contract_tags(stale, receipt)


def test_outcome_ledger_checkpoint_preserves_cross_pod_history(tmp_path):
    """The wave ledger's checkpoint must write the MERGED canonical: a fresh pod starts
    with an empty local sidecar, so checkpointing pod-local content alone erases every
    prior pod's durable rewards on each preemption (the wave-v11 churn stop lost
    ~3,700 graded rows this way)."""
    uri = tmp_path / "wave-outcomes.json"
    first = OutcomeLedger(str(uri), local_dir=tmp_path / "pod1")
    first.record(("task-a", "condensed", "nop"), {"rewards": {"reward": 0.0}})
    first.record(("task-a", "condensed", "oracle"), {"rewards": {"reward": 1.0}})
    first.close()
    # the second pod: the fresh local sidecar, the durable rewards replayed from s3
    second = OutcomeLedger(str(uri), local_dir=tmp_path / "pod2")
    assert second.completed(("task-a", "condensed", "nop")) and second.completed(("task-a", "condensed", "oracle"))
    second.record(("task-b", "condensed", "nop"), {"rewards": {"reward": 0.0}})
    second._checkpoint_locked()
    # the canonical now carries BOTH pods' rewards; a third pod resumes everything
    third = OutcomeLedger(str(uri), local_dir=tmp_path / "pod3")
    assert third.completed(("task-a", "condensed", "nop"))
    assert third.completed(("task-a", "condensed", "oracle"))
    assert third.completed(("task-b", "condensed", "nop"))
    assert third.reward(("task-a", "condensed", "oracle")) == {"reward": 1.0}


def test_verifier_timeout_fails_closed_and_never_feeds_the_churn_window():
    """Ruling 2026-11-20 (the wave-v11 churn refinement): graders fail closed. A verifier
    timeout is a FAILED grading outcome for that phase - an oracle timeout means the oracle
    did not pass (the row is not admitted), a NoP timeout counts as NoP not passing - never
    reward=None, never a pass, and never a churn signal. EnvironmentStartTimeoutError stays
    in the churn window (that is infra)."""
    assert verifier_timeout_text("Verifier execution timed out after 2400.0 seconds")
    assert verifier_timeout_text("harbor.trial.errors.VerifierTimeoutError: deadline")
    assert not verifier_timeout_text("EnvironmentStartTimeoutError: Public fixture initialization timed out")
    # fail-closed values: NoP-not-passing records 1.0; oracle-not-passing records 0.0
    assert fail_closed_reward("nop") == 1.0
    assert fail_closed_reward("oracle") == 0.0
    # a verifier timeout is not a transient infra error and is not retried
    assert not transient_daytona_error(RuntimeError("Verifier execution timed out after 2400.0 seconds"))
    # the recorded float keeps the timeout out of the churn window
    monitor = ChurnMonitor()
    for _ in range(60):
        monitor.record(False)
    assert not monitor.tripped()


def test_gate_doubles_the_setup_deadline_for_the_slow_tail():
    """Ruling 2026-11-21 (the EnvironmentStartTimeout diagnosis): the measured trial
    population finishes its setup in ~126-185 s, but a long tail of heavy installs exceeds
    the declared 1800 s and dies as "Public fixture initialization timed out". The gate's
    trial package doubles the setup deadline (the 4cpu/8GB profile stretches the tail)."""
    row = {
        **PYTHON_ROW,
        "tags": ["swe-rebench"],
        "quality_flags": [],
        "env": [],
        "image_name": "ghcr.io/marin-community/rl-data-swe-rebench-v2@sha256:" + "0" * 64,
        "problem_statement": "Fix the bug.",
        "patch": "diff --git a/x",
        "FAIL_TO_PASS": ["tests/test_x.py::test_a"],
        "PASS_TO_PASS": ["tests/test_x.py::test_b"],
        "test_files": ["tests/test_x.py"],
        "test_patch": "diff --git a/tests/test_x.py",
    }
    files = row_variant_files(row, "condensed", "ghcr.io/marin-community/rl-data-swe-rebench-v2@sha256:" + "0" * 64)
    document = tomlkit.parse(files["task.toml"])
    assert document["environment"]["setup"]["timeout_sec"] == SETUP_TIMEOUT_SEC * 2
