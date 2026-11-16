from pathlib import Path

import pytest

from harbor.tasks.packager import Packager


MULTI_STEP_TOML = """\
[task]
name = "test-org/multi-task"
description = "A multi-step test task"

[environment]
build_timeout_sec = 300

[environment.healthcheck]
command = "test -f /ready"
interval_sec = 1
timeout_sec = 5
retries = 3

[[steps]]
name = "scaffold"
min_reward = 1.0

[steps.agent]
timeout_sec = 60.0

[steps.verifier]
timeout_sec = 30.0

[[steps]]
name = "implement"
min_reward = 0.5

[steps.agent]
timeout_sec = 90.0

[steps.verifier]
timeout_sec = 30.0

[steps.verifier.env]
EXPECTED = "Hello"

[steps.healthcheck]
command = "test -x /app/script.sh"
interval_sec = 1
timeout_sec = 5
retries = 3
"""


@pytest.fixture
def multi_step_task_dir(tmp_path: Path) -> Path:
    d = tmp_path / "multi-task"
    d.mkdir()
    (d / "task.toml").write_text(MULTI_STEP_TOML)

    env_dir = d / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    # Shared top-level tests directory
    (d / "tests").mkdir()
    (d / "tests" / "helpers.sh").write_text("# shared helpers\n")

    steps_dir = d / "steps"
    steps_dir.mkdir()

    for step_name in ("scaffold", "implement"):
        step_dir = steps_dir / step_name
        step_dir.mkdir()
        (step_dir / "instruction.md").write_text(f"# {step_name}\nDo {step_name}.\n")
        sol_dir = step_dir / "solution"
        sol_dir.mkdir()
        (sol_dir / "solve.sh").write_text("#!/bin/bash\nexit 0\n")
        tests_dir = step_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n")

    # Step-scoped workdir for the implement step only
    (steps_dir / "implement" / "workdir").mkdir()
    (steps_dir / "implement" / "workdir" / "config.txt").write_text("name=Hello\n")

    return d


class TestCollectFilesMultiStep:
    def test_includes_every_step_file(self, multi_step_task_dir: Path) -> None:
        files = Packager.collect_files(multi_step_task_dir)
        rel_paths = {f.relative_to(multi_step_task_dir).as_posix() for f in files}
        assert {
            "task.toml",
            "environment/Dockerfile",
            "tests/helpers.sh",
            "steps/scaffold/instruction.md",
            "steps/scaffold/solution/solve.sh",
            "steps/scaffold/tests/test.sh",
            "steps/implement/instruction.md",
            "steps/implement/solution/solve.sh",
            "steps/implement/tests/test.sh",
            "steps/implement/workdir/config.txt",
        } == rel_paths

    def test_no_root_instruction_for_multi_step(
        self, multi_step_task_dir: Path
    ) -> None:
        files = Packager.collect_files(multi_step_task_dir)
        rel_paths = {f.relative_to(multi_step_task_dir).as_posix() for f in files}
        assert "instruction.md" not in rel_paths
