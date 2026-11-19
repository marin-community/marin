"""Supplementary task inputs stay separate from image build definitions."""

from harbor.environments.definition import should_upload_environment_dir


def _empty_env_dir(root):
    directory = root / "environment"
    directory.mkdir()
    return directory


def test_should_upload_environment_dir(temp_dir):
    env_dir = _empty_env_dir(temp_dir)

    assert not should_upload_environment_dir(env_dir, docker_image=None)
    assert not should_upload_environment_dir(env_dir, docker_image="ubuntu:22.04")

    (env_dir / "data.txt").write_text("hello\n")
    assert should_upload_environment_dir(env_dir, docker_image="ubuntu:22.04")

    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    assert not should_upload_environment_dir(env_dir, docker_image="ubuntu:22.04")

    (env_dir / "Dockerfile").unlink()
    (env_dir / "docker-compose.yaml").write_text("services: {}\n")
    assert not should_upload_environment_dir(env_dir, docker_image="ubuntu:22.04")
