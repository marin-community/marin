# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest

from experiments.post_training.tmax_reserve import shared_image
from experiments.post_training.tmax_reserve.shared_image import (
    CONTEXT_FILE_MODE,
    FAILED_STEP_MARK,
    IMAGE_B_ENV,
    Template,
    package_shared,
    parse_dockerfile,
    setup_script,
    task_digest,
    write_context,
)

COMMON_HEAD = "FROM ubuntu:22.04\n\nENV DEBIAN_FRONTEND=noninteractive\n"
POST_INSTALL = "COPY post_install.sh /tmp/post_install.sh\nRUN bash /tmp/post_install.sh && rm /tmp/post_install.sh\n"
TMUX = "RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*\n"
TEMPLATE_A = COMMON_HEAD + "\n" + POST_INSTALL + TMUX
TEMPLATE_B = (
    COMMON_HEAD
    + "ENV LC_ALL=C.UTF-8\nENV LANG=C.UTF-8\nENV GOPATH=/home/user/go\n"
    + "ENV PATH=/home/user/go/bin:/home/user/.cargo/bin:$PATH\n\n"
    + "COPY base_install.sh /tmp/base_install.sh\nRUN bash /tmp/base_install.sh && rm /tmp/base_install.sh\n\n"
    + "COPY _fixtures/app/fixtures/image.png /app/fixtures/image.png\n\n"
    + POST_INSTALL
    + TMUX
)


def test_dockerfiles_classify_into_the_two_templates():
    assert parse_dockerfile(TEMPLATE_A).template is Template.A
    b = parse_dockerfile(TEMPLATE_B)
    assert b.template is Template.B
    assert b.fixtures == [("app/fixtures/image.png", "app/fixtures/image.png")]
    assert b.env["PATH"].startswith("/home/user/go/bin:/home/user/.cargo/bin:/usr/local/sbin")


@pytest.mark.parametrize(
    "dockerfile",
    [
        TEMPLATE_A + "RUN curl https://example.com | sh\n",
        TEMPLATE_A.replace("ENV DEBIAN_FRONTEND=noninteractive\n", ""),
        TEMPLATE_B.replace("ENV LANG=C.UTF-8\n", ""),
        TEMPLATE_A.replace(POST_INSTALL, POST_INSTALL + "COPY _fixtures/a /a\n"),
        TEMPLATE_A.replace("ubuntu:22.04", "ubuntu:24.04"),
    ],
)
def test_dockerfiles_outside_the_templates_are_rejected(dockerfile):
    with pytest.raises(ValueError):
        parse_dockerfile(dockerfile)


def _shimmed_env(tmp_path, env):
    """The template env with apt-get and rm replaced by shims that log their arguments to ``<tmp>/calls``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("apt-get", "rm"):
        (bin_dir / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> {tmp_path}/calls\n')
        (bin_dir / tool).chmod(0o755)
    return {**env, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def test_setup_runs_post_install_then_the_tmux_step_with_template_env_and_retries_once(tmp_path):
    counter = tmp_path / "attempts"
    (tmp_path / "post_install.sh").write_text(
        f'echo x >> {counter}\necho "post_install" >> {tmp_path}/calls\npwd > {tmp_path}/cwd\n'
        f'echo "$GOPATH $HOME" > {tmp_path}/env\n[ $(wc -l < {counter}) -ge 2 ]\n'
    )
    (tmp_path / "setup.sh").write_text(setup_script(_shimmed_env(tmp_path, parse_dockerfile(TEMPLATE_B).env)))

    result = subprocess.run(["bash", str(tmp_path / "setup.sh")], capture_output=True, text=True)

    assert result.returncode == 0
    assert "attempt 1 exited 1" in result.stderr
    assert counter.read_text().count("x") == 2
    # The image build ran the tmux step after post_install.sh; it also configures packages left unpacked.
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls[:4] == [
        "post_install",
        "post_install",
        "apt-get update",
        "apt-get install -y --no-install-recommends tmux",
    ]
    # The shell expands the lists glob, so the shim sees this host's file names.
    assert len(calls) == 5 and calls[4].startswith("rm -rf /var/lib/apt/lists/")
    assert (tmp_path / "cwd").read_text().strip() == "/"
    assert (tmp_path / "env").read_text().split() == [IMAGE_B_ENV["GOPATH"], "/root"]
    assert (
        result.stderr.splitlines()[-1] == f"{FAILED_STEP_MARK} attempt 1 line 5 exit 1: [ $(wc -l < {counter}) -ge 2 ]"
    )


def test_setup_logs_unhandled_failing_steps_without_changing_the_script(tmp_path):
    (tmp_path / "post_install.sh").write_text(
        "false || echo handled\n"
        "if false; then :; fi\n"
        "install_it() { missing-tool --flag; }\n"
        "install_it\n"
        f'echo "$0 ${{BASH_ENV:-unset}} $SETUP_ATTEMPT" > {tmp_path}/seen\n'
    )
    (tmp_path / "setup.sh").write_text(setup_script(_shimmed_env(tmp_path, parse_dockerfile(TEMPLATE_A).env)))

    result = subprocess.run(["bash", str(tmp_path / "setup.sh")], capture_output=True, text=True)

    assert result.returncode == 0
    # A failing function body is reported in the function and again at its call; handled steps never.
    assert [line for line in result.stderr.splitlines() if line.startswith(FAILED_STEP_MARK)] == [
        f"{FAILED_STEP_MARK} attempt 1 line 3 exit 127: missing-tool --flag",
        f"{FAILED_STEP_MARK} attempt 1 line 4 exit 127: missing-tool --flag",
    ]
    # The script sees its own path as $0, and the tracing variables do not reach it or its children.
    assert (tmp_path / "seen").read_text().split() == [str(tmp_path / "post_install.sh"), "unset"]


def test_setup_fails_after_two_failed_attempts(tmp_path):
    (tmp_path / "post_install.sh").write_text("exit 7\n")
    (tmp_path / "setup.sh").write_text(setup_script(_shimmed_env(tmp_path, parse_dockerfile(TEMPLATE_A).env)))

    result = subprocess.run(["bash", str(tmp_path / "setup.sh")], capture_output=True, text=True)

    assert result.returncode == 7
    assert result.stderr.count("exited 7") == 2
    assert not (tmp_path / "calls").exists()


def _source_task(root, name, dockerfile, fixtures=False):
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Do it.")
    (task / "task.toml").write_text('[task]\nname = "x"\n\n[environment]\ncpus = 1\n')
    environment = task / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(dockerfile)
    (environment / "post_install.sh").write_text("echo hi\n")
    if fixtures:
        (environment / "base_install.sh").write_text("echo base\n")
        (environment / "_fixtures" / "app" / "fixtures").mkdir(parents=True)
        (environment / "_fixtures" / "app" / "fixtures" / "image.png").write_bytes(b"png")


def _contexts(root):
    a, b = root / "context-a", root / "context-b"
    a.mkdir()
    b.mkdir()
    (a / "Dockerfile").write_text("FROM ubuntu\n")
    (b / "Dockerfile").write_text("FROM ubuntu\nCOPY base_install.sh /tmp/\n")
    (b / "base_install.sh").write_text("echo base\n")
    return {Template.A: a, Template.B: b}


def _fake_names(monkeypatch, keys):
    def names(task_dirs):
        return {Path(task).name: keys[Path(task, "environment", "Dockerfile").read_text()] for task in task_dirs}

    monkeypatch.setattr(shared_image, "snapshot_names", names)


def _pilot(root, b_base_install="echo base\n"):
    source = root / "pilot"
    _source_task(source, "task_a", TEMPLATE_A)
    _source_task(source, "task_b", TEMPLATE_B, fixtures=True)
    (source / "task_b" / "environment" / "base_install.sh").write_text(b_base_install)
    digests = {name: task_digest(source / name) for name in ("task_a", "task_b")}
    (root / "pilot.manifest.json").write_text(json.dumps({"tasks": digests, "sha256": "src"}))
    return source


def test_package_shared_writes_template_contexts_and_hides_build_inputs(tmp_path, monkeypatch):
    contexts = _contexts(tmp_path)
    _fake_names(monkeypatch, {"FROM ubuntu\n": "snap-a", "FROM ubuntu\nCOPY base_install.sh /tmp/\n": "snap-b"})

    manifest = package_shared(_pilot(tmp_path), contexts, tmp_path / "shared")

    assert manifest["snapshot_keys"] == {"A": "snap-a", "B": "snap-b"}
    assert {task: row["template"] for task, row in manifest["tasks"].items()} == {"task_a": "A", "task_b": "B"}
    b = tmp_path / "shared" / "task_b"
    assert sorted(path.name for path in (b / "environment").iterdir()) == ["Dockerfile", "base_install.sh"]
    assert (b / "task_setup" / "files" / "app" / "fixtures" / "image.png").read_bytes() == b"png"
    assert (b / "task_setup" / "post_install.sh").read_text() == "echo hi\n"
    assert (b / "task.toml").read_text() == (tmp_path / "pilot" / "task_b" / "task.toml").read_text()


def test_package_shared_requires_one_snapshot_key_per_template(tmp_path, monkeypatch):
    contexts = _contexts(tmp_path)
    _fake_names(monkeypatch, {"FROM ubuntu\n": "snap-same", "FROM ubuntu\nCOPY base_install.sh /tmp/\n": "snap-same"})

    with pytest.raises(ValueError, match="one distinct snapshot key per template"):
        package_shared(_pilot(tmp_path), contexts, tmp_path / "shared")


def test_package_shared_refuses_a_task_whose_base_install_differs_from_template_b(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"base_install\.sh differs"):
        package_shared(_pilot(tmp_path, b_base_install="echo other\n"), _contexts(tmp_path), tmp_path / "shared")


def test_context_modes_do_not_depend_on_the_umask(tmp_path):
    previous = os.umask(0o002)
    try:
        write_context(_contexts(tmp_path)[Template.B], tmp_path / "environment")
    finally:
        os.umask(previous)
    assert {path.stat().st_mode & 0o777 for path in (tmp_path / "environment").iterdir()} == {CONTEXT_FILE_MODE}


def test_package_shared_refuses_a_source_that_drifted_from_its_manifest(tmp_path):
    source = tmp_path / "pilot"
    _source_task(source, "task_a", TEMPLATE_A)
    (tmp_path / "pilot.manifest.json").write_text(json.dumps({"tasks": {"task_a": "0" * 64}, "sha256": "src"}))

    with pytest.raises(ValueError, match="does not match"):
        package_shared(source, _contexts(tmp_path), tmp_path / "shared")
