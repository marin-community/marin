# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from experiments.post_training.tmax_reserve.shared_image import (
    IMAGE_B_ENV,
    Template,
    package_shared,
    parse_dockerfile,
    setup_script,
    task_digest,
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
IMAGES = {Template.A: "r.example/a@sha256:" + "a" * 64, Template.B: "r.example/b@sha256:" + "b" * 64}


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


def test_setup_runs_post_install_from_root_with_template_env_and_retries_once(tmp_path):
    counter = tmp_path / "attempts"
    (tmp_path / "post_install.sh").write_text(
        f'echo x >> {counter}\npwd > {tmp_path}/cwd\necho "$GOPATH $HOME" > {tmp_path}/env\n'
        f"[ $(wc -l < {counter}) -ge 2 ]\n"
    )
    (tmp_path / "setup.sh").write_text(setup_script(parse_dockerfile(TEMPLATE_B).env))

    result = subprocess.run(["bash", str(tmp_path / "setup.sh")], capture_output=True, text=True)

    assert result.returncode == 0
    assert "attempt 1 exited 1" in result.stderr
    assert counter.read_text().count("x") == 2
    assert (tmp_path / "cwd").read_text().strip() == "/"
    assert (tmp_path / "env").read_text().split() == [IMAGE_B_ENV["GOPATH"], "/root"]


def test_setup_fails_after_two_failed_attempts(tmp_path):
    (tmp_path / "post_install.sh").write_text("exit 7\n")
    (tmp_path / "setup.sh").write_text(setup_script(parse_dockerfile(TEMPLATE_A).env))

    result = subprocess.run(["bash", str(tmp_path / "setup.sh")], capture_output=True, text=True)

    assert result.returncode == 7
    assert result.stderr.count("exited 7") == 2


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


def test_package_shared_converts_verified_tasks_and_hides_build_inputs(tmp_path):
    source = tmp_path / "pilot"
    _source_task(source, "task_a", TEMPLATE_A)
    _source_task(source, "task_b", TEMPLATE_B, fixtures=True)
    digests = {name: task_digest(source / name) for name in ("task_a", "task_b")}
    (tmp_path / "pilot.manifest.json").write_text(json.dumps({"tasks": digests, "sha256": "src"}))

    manifest = package_shared(source, IMAGES, tmp_path / "shared")

    assert {task: row["template"] for task, row in manifest["tasks"].items()} == {"task_a": "A", "task_b": "B"}
    b = tmp_path / "shared" / "task_b"
    assert (b / "task_setup" / "files" / "app" / "fixtures" / "image.png").read_bytes() == b"png"
    assert (b / "task_setup" / "post_install.sh").read_text() == "echo hi\n"
    assert sorted(path.name for path in (b / "environment").iterdir()) == ["Dockerfile"]
    assert f'docker_image = "{IMAGES[Template.B]}"' in (b / "task.toml").read_text()


def test_package_shared_refuses_a_source_that_drifted_from_its_manifest(tmp_path):
    source = tmp_path / "pilot"
    _source_task(source, "task_a", TEMPLATE_A)
    (tmp_path / "pilot.manifest.json").write_text(json.dumps({"tasks": {"task_a": "0" * 64}, "sha256": "src"}))

    with pytest.raises(ValueError, match="does not match"):
        package_shared(source, IMAGES, tmp_path / "shared")
