# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish checked guest ports with shared immutable assets and versioned catalogs.

Marin caches stage records; Buildomatic owns recipe scheduling and BATCH worker
placement. A source checkout is needed only while an uncached stage executes.
"""

import logging
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urljoin

import click
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.execution.step_status import step_lock
from packaging.version import Version
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

CACHE_PREFIX = "s3://marin-us-east-02a/marin/shellsim/buildomatic/v2"
PUBLICATION_PREFIX = "s3://marin-us-east-02a/marin/shellsim/ports"
ASSET_URL_PREFIX = "https://ports.oa.dev"
SOURCE_REPOSITORY = "https://github.com/rjpower/shellsim.git"
STATE_FILES = ("graph.json", "results.json")
RELEASE_DIRECTORY = "releases"
CATALOG_FILENAME = "catalog.json"
RUNTIME_FIELDS = ("work_directory", "max_workers", "worker_cpus", "worker_memory_gb", "worker_disk_gb")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReleaseConfig:
    """Source and worker image, with driver storage and worker resources explicit."""

    source_revision: str
    version: str
    work_directory: str
    worker_image: str
    cache_prefix: str = CACHE_PREFIX
    publication_prefix: str = PUBLICATION_PREFIX
    asset_url_prefix: str = ASSET_URL_PREFIX
    source_repository: str = SOURCE_REPOSITORY
    max_workers: int = 8
    worker_cpus: int = 64
    worker_memory_gb: int = 32
    worker_disk_gb: int = 100

    def __post_init__(self) -> None:
        Version(self.version)
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_revision):
            raise ValueError("source_revision must be a full Git commit SHA")


@dataclass(frozen=True)
class StageConfig:
    release: ReleaseConfig
    output_path: str
    input_path: str | None = None


def _checkout(release: ReleaseConfig) -> Path:
    """Reuse a pinned checkout without installing or building the source SDK."""
    root = Path(release.work_directory).resolve() / "source" / release.source_revision
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").is_dir():
        subprocess.run(["git", "init", str(root)], check=True)
    current = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    if current.returncode == 0 and current.stdout.strip() == release.source_revision:
        return root
    subprocess.run(
        ["git", "-C", str(root), "fetch", "--depth=1", release.source_repository, release.source_revision], check=True
    )
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", release.source_revision], check=True)
    return root


def _state(config: StageConfig, stage: str) -> Path:
    state = Path(config.release.work_directory).resolve() / "state" / config.release.version / stage
    state.mkdir(parents=True, exist_ok=True)
    if config.input_path is not None:
        for name in STATE_FILES:
            (StoragePath(config.input_path) / name).download_to(str(state / name))
    return state


def _save_state(state: Path, output_path: str) -> None:
    StoragePath(output_path).mkdirs(exist_ok=True)
    for name in STATE_FILES:
        (StoragePath(output_path) / name).upload_from(str(state / name))


def _run_ports(config: StageConfig, state: Path, operation: str, *arguments: str) -> None:
    release = config.release
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ports",
            operation,
            "--state",
            str(state),
            "--store",
            release.cache_prefix,
            "--backend",
            "iris",
            "--iris-image",
            release.worker_image,
            "--max-workers",
            str(release.max_workers),
            "--worker-cpus",
            str(release.worker_cpus),
            "--worker-memory-gb",
            str(release.worker_memory_gb),
            "--worker-disk-gb",
            str(release.worker_disk_gb),
            *arguments,
        ],
        cwd=_checkout(release),
        check=True,
    )


def build_ports(config: StageConfig) -> None:
    """Build every guest recipe with BATCH workers and the existing sccache store."""
    state = _state(config, "build")
    _run_ports(config, state, "build", "--all")
    _save_state(state, config.output_path)


def check_ports(config: StageConfig) -> None:
    """Run the recipe graph's guest checks before any catalog becomes public."""
    state = _state(config, "check")
    _run_ports(config, state, "check")
    _save_state(state, config.output_path)


def publish_ports(config: StageConfig) -> None:
    """Publish snapshots before promoting the newest version to the stable catalog."""
    state = _state(config, "export")
    _run_ports(
        config,
        state,
        "export",
        "--url-prefix",
        config.release.asset_url_prefix,
        "--output",
        str(StoragePath(config.release.publication_prefix) / RELEASE_DIRECTORY / config.release.version),
        "--copy-assets",
        "--asset-output",
        config.release.publication_prefix,
        "--simple",
    )
    publication = StoragePath(config.release.publication_prefix)
    with step_lock(str(StoragePath(config.output_path).parent / ".promotion"), "ports stable catalog", force_rerun=True):
        current = publication / "release-version"
        if current.exists():
            current_version = Version(current.read_text())
            if current_version > Version(config.release.version):
                logger.info(
                    "Published snapshot %s; skipping stable promotion because %s is newer",
                    config.release.version,
                    current_version,
                )
                return
        current.write_text(config.release.version)
        _run_ports(
            config,
            state,
            "export",
            "--url-prefix",
            config.release.asset_url_prefix,
            "--output",
            config.release.publication_prefix,
            "--simple",
        )


def _release_config(config: ReleaseConfig, ctx: StepContext) -> ReleaseConfig:
    return replace(config, **{name: ctx.runtime_arg(name) for name in RUNTIME_FIELDS})


def ports_release(config: ReleaseConfig) -> ArtifactStep[Artifact]:
    """Declare build, check, and publication dependencies without executing."""
    runtime = {name: getattr(config, name) for name in RUNTIME_FIELDS}
    build = ArtifactStep(
        name="shellsim/buildomatic/build",
        version=config.version,
        artifact_type=Artifact,
        run=build_ports,
        build_config=lambda ctx: StageConfig(_release_config(config, ctx), ctx.output_path),
        runtime_args=runtime,
    )
    checked = ArtifactStep(
        name="shellsim/buildomatic/checked",
        version=config.version,
        artifact_type=Artifact,
        run=check_ports,
        build_config=lambda ctx: StageConfig(_release_config(config, ctx), ctx.output_path, ctx.artifact_path(build)),
        runtime_args=runtime,
        deps=(build,),
    )
    return ArtifactStep(
        name="shellsim/buildomatic/published",
        version=config.version,
        artifact_type=Artifact,
        run=publish_ports,
        build_config=lambda ctx: StageConfig(_release_config(config, ctx), ctx.output_path, ctx.artifact_path(checked)),
        runtime_args=runtime,
        deps=(checked,),
    )


@click.command()
@click.option("--source-revision", required=True, help="Full Shellsim source commit SHA, including ports recipes.")
@click.option("--version", required=True, help="Immutable release version; bump when source or runtime changes.")
@click.option("--worker-image", help="Override the worker image published by the Pulumi stack.")
@click.option("--stack", default="cw-us-east-02a", show_default=True, help="Shellsim ports Pulumi stack.")
@click.option("--url-prefix", default=ASSET_URL_PREFIX, show_default=True)
@click.option("--work-directory", default="build/ports-release", show_default=True)
@click.option("--max-workers", default=ReleaseConfig.max_workers, type=click.IntRange(1, 32), show_default=True)
@click.option("--worker-cpus", default=ReleaseConfig.worker_cpus, type=click.IntRange(min=1), show_default=True)
@click.option(
    "--worker-memory-gb", default=ReleaseConfig.worker_memory_gb, type=click.IntRange(min=1), show_default=True
)
@click.option("--worker-disk-gb", default=ReleaseConfig.worker_disk_gb, type=click.IntRange(min=1), show_default=True)
@click.option("--run", "do_run", is_flag=True, help="Execute the graph; default prints a plan without submitting jobs.")
def main(
    source_revision: str,
    version: str,
    worker_image: str | None,
    stack: str,
    work_directory: str,
    url_prefix: str,
    max_workers: int,
    worker_cpus: int,
    worker_memory_gb: int,
    worker_disk_gb: int,
    do_run: bool,
) -> None:
    configure_logging()
    if worker_image is None:
        worker_image = subprocess.run(
            ["pulumi", "stack", "output", "workerImage", "--stack", stack],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    logger.info("Using ports worker image %s", worker_image)
    handle = ports_release(
        ReleaseConfig(
            source_revision,
            version,
            work_directory,
            worker_image=worker_image,
            asset_url_prefix=url_prefix,
            max_workers=max_workers,
            worker_cpus=worker_cpus,
            worker_memory_gb=worker_memory_gb,
            worker_disk_gb=worker_disk_gb,
        )
    )
    if do_run:
        configure_coreweave_s3()
        run(handle)
        base = url_prefix.rstrip("/") + "/"
        click.echo(f"Catalog: {urljoin(base, CATALOG_FILENAME)}")
        click.echo(f"Snapshot: {urljoin(base, f'{RELEASE_DIRECTORY}/{version}/{CATALOG_FILENAME}')}")
    else:
        click.echo(lower(handle))


if __name__ == "__main__":
    main()
