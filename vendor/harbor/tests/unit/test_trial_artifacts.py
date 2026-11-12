import asyncio
import fnmatch
import json
import logging
import shutil
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harbor.models.trial.config import ArtifactConfig
from harbor.models.trial.paths import EnvironmentPaths
from harbor.trial.artifact_handler import ArtifactHandler
from harbor.utils.artifact_writer import get_artifact_writer

ENV_ARTIFACTS_DIR = EnvironmentPaths().artifacts_dir
WINDOWS_ARTIFACTS_DIR = EnvironmentPaths.for_windows().artifacts_dir


def _handler(
    artifacts: list[str | ArtifactConfig],
) -> ArtifactHandler:
    return ArtifactHandler(
        artifacts=artifacts,
        logger=logging.getLogger(__name__),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_downloads_configured_file_to_destination(tmp_path: Path) -> None:
    environment = AsyncMock()
    environment.capabilities.mounted = True
    environment.is_dir = AsyncMock(return_value=False)
    environment.download_file = AsyncMock()
    environment.download_file.side_effect = lambda source_path, target_path: (
        target_path.write_text("answer")
    )
    handler = _handler(
        [
            ArtifactConfig(
                source="/tmp/answer.json",
                destination="answers/final.json",
            )
        ],
    )

    artifacts_dir = tmp_path / "artifacts"

    manifest = await handler.download_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    assert (artifacts_dir / "answers" / "final.json").read_text() == "answer"
    assert manifest.entries[1].source == "/tmp/answer.json"
    assert manifest.entries[1].destination == "artifacts/answers/final.json"
    assert manifest.entries[1].type == "file"
    assert manifest.entries[1].status == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_downloads_configured_directory_with_exclude(tmp_path: Path) -> None:
    environment = AsyncMock()
    environment.capabilities.mounted = True
    environment.is_dir = AsyncMock(return_value=True)
    environment.download_dir = AsyncMock()
    environment.download_dir_with_exclusions = AsyncMock()
    handler = _handler(
        [
            ArtifactConfig(
                source="/app/my dir",
                exclude=["*.pyc", "helper files", "$(touch hacked)"],
            )
        ],
    )

    artifacts_dir = tmp_path / "artifacts"

    await handler.download_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    environment.download_dir.assert_not_awaited()
    assert (artifacts_dir / "my dir").is_dir()
    assert environment.download_dir_with_exclusions.call_args.kwargs["exclude"] == [
        "*.pyc",
        "helper files",
        "$(touch hacked)",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_implicit_artifacts_dir_downloads_to_artifacts_root(
    tmp_path: Path,
) -> None:
    environment = AsyncMock()
    environment.capabilities.mounted = False
    environment.is_dir = AsyncMock(return_value=True)
    environment.download_dir = AsyncMock()
    handler = _handler([])

    artifacts_dir = tmp_path / "artifacts"

    manifest = await handler.download_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    assert artifacts_dir.is_dir()
    assert manifest.entries[0].source == "/logs/artifacts"
    assert manifest.entries[0].destination == "artifacts"
    get_artifact_writer().flush(timeout=30)
    disk_manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    assert disk_manifest[0]["source"] == "/logs/artifacts"
    assert disk_manifest[0]["destination"] == "artifacts"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_artifacts_dir_with_exclude_uses_artifacts_root(
    tmp_path: Path,
) -> None:
    environment = AsyncMock()
    environment.capabilities.mounted = False
    environment.is_dir = AsyncMock(return_value=True)
    environment.download_dir = AsyncMock()
    environment.download_dir_with_exclusions = AsyncMock()
    handler = _handler(
        [ArtifactConfig(source="/logs/artifacts", exclude=["*.pt"])],
    )

    artifacts_dir = tmp_path / "artifacts"

    await handler.download_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    environment.download_dir.assert_not_awaited()
    assert artifacts_dir.is_dir()
    assert environment.download_dir_with_exclusions.call_args.kwargs["exclude"] == [
        "*.pt"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uploads_implicit_artifacts_dir_from_artifacts_root(
    tmp_path: Path,
) -> None:
    environment = AsyncMock()
    environment.upload_dir = AsyncMock()
    environment.empty_dirs = AsyncMock()
    environment.reset_dirs = AsyncMock()
    handler = _handler([])
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "result.txt").write_text("ok")

    await handler.upload_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
        target_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    environment.empty_dirs.assert_awaited_once_with(["/logs/artifacts"], chmod=True)
    environment.reset_dirs.assert_not_awaited()
    environment.upload_dir.assert_awaited_once_with(
        source_dir=artifacts_dir,
        target_dir="/logs/artifacts",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uploads_configured_file_from_destination_to_source(
    tmp_path: Path,
) -> None:
    environment = AsyncMock()
    environment.upload_file = AsyncMock()
    environment.upload_dir = AsyncMock()
    environment.empty_dirs = AsyncMock()
    environment.reset_dirs = AsyncMock()
    handler = _handler(
        [
            ArtifactConfig(
                source="/tmp/answer.json",
                destination="answers/final.json",
            )
        ],
    )
    artifacts_dir = tmp_path / "artifacts"
    target = artifacts_dir / "answers" / "final.json"
    target.parent.mkdir(parents=True)
    target.write_text("ok")

    await handler.upload_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
        target_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    environment.upload_file.assert_awaited_once_with(
        source_path=target,
        target_path="/tmp/answer.json",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uploads_configured_directory_from_destination_to_source(
    tmp_path: Path,
) -> None:
    environment = AsyncMock()
    environment.upload_dir = AsyncMock()
    environment.empty_dirs = AsyncMock()
    environment.reset_dirs = AsyncMock()
    handler = _handler(
        [ArtifactConfig(source="/tmp/output", destination="out")],
    )
    artifacts_dir = tmp_path / "artifacts"
    target = artifacts_dir / "out"
    target.mkdir(parents=True)
    (target / "result.txt").write_text("ok")

    await handler.upload_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
        target_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    environment.empty_dirs.assert_any_await(["/tmp/output"], chmod=True)
    environment.reset_dirs.assert_not_awaited()
    environment.upload_dir.assert_any_await(
        source_dir=target,
        target_dir="/tmp/output",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_skips_missing_host_paths(tmp_path: Path) -> None:
    environment = AsyncMock()
    environment.upload_file = AsyncMock()
    environment.upload_dir = AsyncMock()
    handler = _handler(
        [ArtifactConfig(source="/tmp/missing.txt", destination="missing.txt")],
    )

    await handler.upload_artifacts(
        environment,
        tmp_path / "artifacts",
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
        target_artifacts_dir=ENV_ARTIFACTS_DIR,
    )

    environment.upload_file.assert_not_awaited()
    environment.upload_dir.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uploads_implicit_artifacts_dir_to_target_convention(
    tmp_path: Path,
) -> None:
    environment = AsyncMock()
    environment.upload_dir = AsyncMock()
    environment.empty_dirs = AsyncMock()
    environment.reset_dirs = AsyncMock()
    handler = _handler([])
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "result.txt").write_text("ok")

    await handler.upload_artifacts(
        environment,
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
        target_artifacts_dir=WINDOWS_ARTIFACTS_DIR,
    )

    windows_artifacts_dir = WINDOWS_ARTIFACTS_DIR.as_posix()
    environment.empty_dirs.assert_awaited_once_with([windows_artifacts_dir], chmod=True)
    environment.reset_dirs.assert_not_awaited()
    environment.upload_dir.assert_awaited_once_with(
        source_dir=artifacts_dir,
        target_dir=windows_artifacts_dir,
    )


@pytest.mark.unit
def test_move_dir_contents_moves_contents_and_leaves_source_empty(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "file.txt").write_text("ok")
    (src / "nested").mkdir()
    (src / "nested" / "value.txt").write_text("nested")

    ArtifactHandler.move_dir_contents(src, dst)

    assert not any(src.iterdir())
    assert (dst / "file.txt").read_text() == "ok"
    assert (dst / "nested" / "value.txt").read_text() == "nested"


def _copy_environment(root: Path, *, mounted: bool = False) -> AsyncMock:
    environment = AsyncMock()
    environment.capabilities.mounted = mounted
    environment.is_dir.side_effect = lambda source, user: (
        root / source.lstrip("/")
    ).is_dir()
    environment.download_dir.side_effect = lambda source_dir, target_dir: (
        shutil.copytree(
            root / source_dir.lstrip("/"), target_dir, dirs_exist_ok=True, symlinks=True
        )
    )
    environment.download_file.side_effect = lambda source_path, target_path: (
        shutil.copy2(root / source_path.lstrip("/"), target_path)
    )

    def download_with_exclusions(source_dir, target_dir, exclude):
        source = root / source_dir.lstrip("/")
        return shutil.copytree(
            source,
            target_dir,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=lambda directory, names: {
                name
                for name in names
                if any(
                    fnmatch.fnmatch(
                        "./" + str(Path(directory, name).relative_to(source)), pattern
                    )
                    for pattern in exclude
                )
            },
        )

    environment.download_dir_with_exclusions.side_effect = download_with_exclusions

    def empty_dirs(paths, *, chmod):
        for path in paths:
            target = root / path.lstrip("/")
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)

    environment.empty_dirs.side_effect = empty_dirs
    environment.upload_dir.side_effect = lambda source_dir, target_dir: shutil.copytree(
        source_dir, root / target_dir.lstrip("/"), dirs_exist_ok=True, symlinks=True
    )
    environment.upload_file.side_effect = lambda source_path, target_path: shutil.copy2(
        source_path, root / target_path.lstrip("/")
    )
    return environment


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("mounted", [False, True])
async def test_failed_capture_removes_partial_and_previous_data(
    tmp_path: Path, kind: str, mounted: bool
) -> None:
    actor = tmp_path / "actor"
    (actor / "logs/artifacts").mkdir(parents=True)
    source = actor / "answer"
    if kind == "directory":
        (source / ".git").mkdir(parents=True)
        (source / "code.py").write_text("complete")
    else:
        source.write_text("complete answer")
    environment = _copy_environment(actor, mounted=mounted)
    handler = _handler([ArtifactConfig(source="/answer", destination="answer")])
    artifacts_dir = tmp_path / "artifacts"
    await handler.download_artifacts(
        environment, artifacts_dir, source_artifacts_dir=ENV_ARTIFACTS_DIR
    )

    def interrupted_copy(**kwargs):
        if kwargs.get("source_dir") == "/logs/artifacts":
            return
        if kind == "directory":
            (kwargs["target_dir"] / ".git").mkdir()
        else:
            kwargs["target_path"].write_text("partial")
        raise OSError("capture interrupted")

    environment.download_dir.side_effect = interrupted_copy
    environment.download_file.side_effect = interrupted_copy
    manifest = await handler.download_artifacts(
        environment, artifacts_dir, source_artifacts_dir=ENV_ARTIFACTS_DIR
    )
    assert manifest.entries[1].status == "failed"
    assert not (artifacts_dir / "answer").exists()
    verifier = tmp_path / "verifier"
    await handler.upload_artifacts(
        _copy_environment(verifier),
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
        target_artifacts_dir=ENV_ARTIFACTS_DIR,
    )
    assert not (verifier / "answer").exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_capture_does_not_restore_previous_answer(
    tmp_path: Path,
) -> None:
    actor = tmp_path / "actor"
    actor.mkdir()
    (actor / "answer").write_text("previous answer")
    environment = _copy_environment(actor, mounted=True)
    handler = _handler([ArtifactConfig(source="/answer", destination="answer")])
    artifacts_dir = tmp_path / "artifacts"
    await handler.download_artifacts(
        environment, artifacts_dir, source_artifacts_dir=ENV_ARTIFACTS_DIR
    )
    started = asyncio.Event()

    async def blocked_probe(source, user):
        started.set()
        await asyncio.Future()

    environment.is_dir.side_effect = blocked_probe
    capture = asyncio.create_task(
        handler.download_artifacts(
            environment, artifacts_dir, source_artifacts_dir=ENV_ARTIFACTS_DIR
        )
    )
    await started.wait()
    capture.cancel()
    with pytest.raises(asyncio.CancelledError):
        await capture
    assert not (artifacts_dir / "answer").exists()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("convention_destination", [None, "convention"])
@pytest.mark.parametrize("convention_first", [False, True])
async def test_repository_capture_replaces_convention_collision(
    tmp_path: Path,
    mounted: bool,
    convention_destination: str | None,
    convention_first: bool,
) -> None:
    actor = tmp_path / "actor"
    artifacts_dir = tmp_path / "artifacts"
    logs = artifacts_dir if mounted else actor / "logs/artifacts"
    logs.mkdir(parents=True)
    if mounted:
        (actor / "logs").mkdir(parents=True)
        (actor / "logs/artifacts").symlink_to(logs, target_is_directory=True)
    (logs / "testbed").mkdir()
    (logs / "testbed/extra.py").write_text("not in repository")
    (logs / "progress.txt").write_text("actor log")
    (logs / "diagnostics/testbed").mkdir(parents=True)
    (logs / "diagnostics/testbed/log.txt").write_text("nested user log")
    (actor / "testbed/.git").mkdir(parents=True)
    (actor / "testbed/.git/config").write_text("repository")
    executable = actor / "testbed/run.sh"
    executable.write_text("echo candidate")
    executable.chmod(0o755)
    (actor / "testbed/link").symlink_to("run.sh")
    artifacts = [ArtifactConfig(source="/testbed", destination="testbed")]
    artifacts.insert(
        0 if convention_first else 1,
        ArtifactConfig(
            source="/logs/artifacts",
            destination=convention_destination,
            exclude=["./" + convention_destination, "./testbed"]
            if convention_destination
            else [],
        ),
    )
    handler = _handler(artifacts)
    for _ in range(2):
        manifest = await handler.download_artifacts(
            _copy_environment(actor, mounted=mounted),
            artifacts_dir,
            source_artifacts_dir=ENV_ARTIFACTS_DIR,
        )
        assert all(entry.status == "ok" for entry in manifest.entries)
        assert (logs / "progress.txt").read_text() == "actor log"
    verifier = tmp_path / "verifier"
    await handler.upload_artifacts(
        _copy_environment(verifier),
        artifacts_dir,
        source_artifacts_dir=ENV_ARTIFACTS_DIR,
        target_artifacts_dir=ENV_ARTIFACTS_DIR,
    )
    assert not (verifier / "testbed/extra.py").exists()
    assert (verifier / "testbed/.git/config").read_text() == "repository"
    assert (verifier / "testbed/run.sh").stat().st_mode & 0o777 == 0o755
    assert (verifier / "testbed/link").readlink() == Path("run.sh")
    if convention_destination:
        assert (verifier / "logs/artifacts/progress.txt").read_text() == "actor log"
        assert (
            verifier / "logs/artifacts/diagnostics/testbed/log.txt"
        ).read_text() == "nested user log"
        assert not (verifier / "logs/artifacts" / convention_destination).exists()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("rename_fails", [False, True])
async def test_cancellation_waits_for_publication_and_removes_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rename_fails: bool
) -> None:
    actor = tmp_path / "actor"
    actor.mkdir()
    (actor / "answer").write_text("candidate")
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    rename = Path.rename

    def blocked_rename(path, target):
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=10):
            raise TimeoutError("test publication was not released")
        if rename_fails:
            raise OSError("publication failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", blocked_rename)
    handler = _handler([ArtifactConfig(source="/answer", destination="answer")])
    artifacts_dir = tmp_path / "artifacts"
    capture = asyncio.create_task(
        handler.download_artifacts(
            _copy_environment(actor, mounted=True),
            artifacts_dir,
            source_artifacts_dir=ENV_ARTIFACTS_DIR,
        )
    )
    await started.wait()
    capture.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await capture
    assert not (artifacts_dir / "answer").exists()
    assert not list(tmp_path.glob(".artifact-*"))
