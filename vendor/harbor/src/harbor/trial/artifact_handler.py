import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePath, PurePosixPath

from harbor.environments.base import BaseEnvironment, EnvironmentPath
from harbor.models.trial.artifact_manifest import (
    ArtifactManifest,
    ArtifactManifestEntry,
)
from harbor.models.trial.config import ArtifactConfig
from harbor.utils.artifact_writer import get_artifact_writer
from harbor.utils.path_compat import safe_rmtree


def _remove_capture(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        safe_rmtree(path)
    elif path.exists():
        path.unlink()


def _replace_capture(staging: Path, target: Path) -> None:
    _remove_capture(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)


async def _publish_capture(staging: Path, target: Path) -> None:
    publication = asyncio.create_task(
        asyncio.to_thread(_replace_capture, staging, target)
    )
    try:
        await asyncio.shield(publication)
    except asyncio.CancelledError as cancelled:
        # Filesystem threads cannot be cancelled. Cleanup must wait until the
        # rename finishes so it cannot publish a capture after its cancellation.
        try:
            await publication
        finally:
            raise cancelled


class ArtifactHandler:
    def __init__(
        self,
        *,
        artifacts: Sequence[str | ArtifactConfig],
        logger: logging.Logger,
    ):
        self.artifacts = list(artifacts)
        self.logger = logger

    @staticmethod
    def move_dir_contents(src: Path, dst: Path) -> None:
        """Move all contents from src to dst, leaving src empty."""
        if not src.exists():
            return

        items = list(src.iterdir())
        if not items:
            return

        dst.mkdir(parents=True, exist_ok=True)
        for item in items:
            target = dst / item.name
            _remove_capture(target)
            shutil.move(str(item), target)

    async def download_artifacts(
        self,
        source_env: BaseEnvironment,
        artifacts_dir: Path,
        *,
        source_artifacts_dir: EnvironmentPath,
        artifacts: Sequence[str | ArtifactConfig] | None = None,
    ) -> ArtifactManifest:
        """Best-effort artifact download with a manifest of attempted sources."""
        # On a remote (s3://) trials_dir the mkdir below is SYNCHRONOUS fsspec
        # I/O; offload it off the shared event loop. The manifest write at the
        # end goes through the background artifact writer.
        await asyncio.to_thread(artifacts_dir.mkdir, parents=True, exist_ok=True)
        entries: list[ArtifactManifestEntry] = []
        convention_source = self._environment_path_str(source_artifacts_dir)

        for artifact in self._normalized_artifacts(artifacts, convention_source):
            entries.append(
                await self._download_artifact(
                    source_env=source_env,
                    artifacts_dir=artifacts_dir,
                    artifact=artifact,
                    convention_source=convention_source,
                )
            )

        manifest = ArtifactManifest(entries=entries)
        if manifest.entries:
            # Route the (possibly remote fsspec) manifest write through the
            # bounded artifact writer instead of the shared default executor.
            # A manifest is diagnostic and must not delay result publication.
            get_artifact_writer().submit_optional(
                artifacts_dir / "manifest.json",
                lambda: json.dumps(manifest.to_json_data(), indent=2),
                description="trial artifact manifest.json",
            )
        return manifest

    async def upload_artifacts(
        self,
        target_env: BaseEnvironment,
        artifacts_dir: Path,
        *,
        source_artifacts_dir: EnvironmentPath,
        target_artifacts_dir: EnvironmentPath,
        artifacts: Sequence[str | ArtifactConfig] | None = None,
    ) -> None:
        """Upload host artifacts back to their configured environment sources."""
        source_convention = self._environment_path_str(source_artifacts_dir)
        target_convention = self._environment_path_str(target_artifacts_dir)

        for artifact in self._normalized_artifacts(artifacts, source_convention):
            host_path = self._host_path(
                artifacts_dir,
                artifact,
                convention_source=source_convention,
            )
            if not host_path.exists():
                continue

            target_source = self._upload_target_source(
                artifact.source,
                source_convention=source_convention,
                target_convention=target_convention,
            )
            if host_path.is_dir():
                await target_env.empty_dirs([target_source], chmod=True)
                await target_env.upload_dir(
                    source_dir=host_path,
                    target_dir=target_source,
                )
                continue

            await target_env.upload_file(
                source_path=host_path,
                target_path=target_source,
            )

    def _normalized_artifacts(
        self,
        artifacts: Sequence[str | ArtifactConfig] | None,
        convention_source: str,
    ) -> list[ArtifactConfig]:
        artifact_values: list[str | ArtifactConfig] = [
            *self.artifacts,
            *(artifacts or []),
        ]
        normalized = [
            ArtifactConfig(source=artifact) if isinstance(artifact, str) else artifact
            for artifact in artifact_values
        ]

        if not self._has_artifact_source(normalized, convention_source):
            normalized.insert(
                0,
                ArtifactConfig(
                    source=convention_source,
                    destination=convention_source,
                ),
            )
        # Convention capture may replace the artifact root. Collect it before
        # independent captures regardless of their order in the task config.
        normalized.sort(
            key=lambda artifact: (
                not self._is_environment_artifacts_dir(
                    artifact.source, convention_source
                )
            )
        )
        return normalized

    def _has_artifact_source(
        self,
        artifacts: Sequence[ArtifactConfig],
        source: str,
    ) -> bool:
        normalized_source = source.rstrip("/")
        return any(
            artifact.source.rstrip("/") == normalized_source for artifact in artifacts
        )

    async def _download_artifact(
        self,
        *,
        source_env: BaseEnvironment,
        artifacts_dir: Path,
        artifact: ArtifactConfig,
        convention_source: str,
    ) -> ArtifactManifestEntry:
        source = artifact.source
        target = self._host_path(
            artifacts_dir,
            artifact,
            convention_source=convention_source,
        )
        manifest_destination = self._manifest_destination(artifacts_dir, target)

        if (
            self._is_environment_artifacts_dir(source, convention_source)
            and source_env.capabilities.mounted
            and not artifact.exclude
        ):
            return self._record_mounted_artifacts_dir(
                source=source,
                target=target,
                manifest_destination=manifest_destination,
            )

        staging = artifacts_dir.parent / f".artifact-{uuid.uuid4().hex}"
        completed = False
        is_dir = False
        try:
            try:
                is_dir = await source_env.is_dir(source, user="root")
            except Exception:
                is_dir = not Path(source).suffix
            if is_dir:
                await asyncio.to_thread(staging.mkdir, parents=True, exist_ok=True)
                if artifact.exclude:
                    await source_env.download_dir_with_exclusions(
                        source_dir=source,
                        target_dir=staging,
                        exclude=artifact.exclude,
                    )
                else:
                    await source_env.download_dir(
                        source_dir=source,
                        target_dir=staging,
                    )
                artifact_type = "directory"
            else:
                await asyncio.to_thread(
                    staging.parent.mkdir, parents=True, exist_ok=True
                )
                await source_env.download_file(
                    source_path=source,
                    target_path=staging,
                )
                artifact_type = "file"

            # Copy operations can leave partial trees before raising. Publish
            # only a complete capture, replacing any colliding convention files.
            await _publish_capture(staging, target)
            completed = True
            return ArtifactManifestEntry(
                source=source,
                destination=manifest_destination,
                type=artifact_type,
                status="ok",
            )
        except Exception:
            self.logger.debug(
                f"Failed to download artifact '{source}' (best-effort)",
                exc_info=True,
            )
            return ArtifactManifestEntry(
                source=source,
                destination=manifest_destination,
                type="directory" if is_dir else "file",
                status="failed",
            )
        finally:
            await asyncio.to_thread(_remove_capture, staging)
            if not completed:
                await asyncio.to_thread(_remove_capture, target)

    def _record_mounted_artifacts_dir(
        self,
        *,
        source: str,
        target: Path,
        manifest_destination: str,
    ) -> ArtifactManifestEntry:
        has_contents = target.exists() and any(target.iterdir())
        return ArtifactManifestEntry(
            source=source,
            destination=manifest_destination,
            type="directory",
            status="ok" if has_contents else "empty",
        )

    def _host_path(
        self,
        artifacts_dir: Path,
        artifact: ArtifactConfig,
        *,
        convention_source: str,
    ) -> Path:
        if self._is_environment_artifacts_dir(artifact.source, convention_source):
            destination = artifact.destination or artifact.source
            if (
                self._is_environment_artifacts_dir(destination, convention_source)
                or destination == "."
            ):
                return artifacts_dir

        destination = artifact.destination or PurePosixPath(artifact.source).name
        return artifacts_dir / self._relative_host_destination(destination)

    @staticmethod
    def _relative_host_destination(destination: str) -> Path:
        destination_path = PurePosixPath(destination)
        parts = [part for part in destination_path.parts if part not in ("", "/")]
        return Path(*parts) if parts else Path(".")

    def _is_environment_artifacts_dir(
        self,
        source: str,
        convention_source: str,
    ) -> bool:
        return source.rstrip("/") == convention_source.rstrip("/")

    def _upload_target_source(
        self,
        source: str,
        *,
        source_convention: str,
        target_convention: str,
    ) -> str:
        if self._is_environment_artifacts_dir(source, source_convention):
            return target_convention
        return source

    @staticmethod
    def _environment_path_str(path: EnvironmentPath) -> str:
        if isinstance(path, PurePath):
            return path.as_posix()
        return path

    def _manifest_destination(self, artifacts_dir: Path, target: Path) -> str:
        if target == artifacts_dir:
            return "artifacts"
        return f"artifacts/{target.relative_to(artifacts_dir).as_posix()}"
