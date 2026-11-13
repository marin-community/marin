from pathlib import Path


def should_upload_environment_dir(
    environment_dir: Path,
    *,
    docker_image: str | None,
) -> bool:
    """True when a prebuilt image has supplementary files rather than build inputs."""
    if not docker_image:
        return False
    if (environment_dir / "Dockerfile").exists():
        return False
    if (environment_dir / "docker-compose.yaml").exists():
        return False
    return environment_dir.is_dir() and any(environment_dir.iterdir())
