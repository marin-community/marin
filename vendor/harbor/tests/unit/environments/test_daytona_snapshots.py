"""Unit tests for DaytonaSnapshotService naming and hashing."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.environments.daytona.snapshots import (
    DaytonaSnapshotService,
    bake_agent_tooling,
)
from harbor.utils.logger import logger


class TestBakeAgentTooling:
    """The tmux layer must not concatenate into a task Dockerfile
    whose file lacks a trailing newline (would produce e.g.
    ``numpy==2.2.0RUN DEBIAN_FRONTEND=...`` → deterministic BUILD_FAILED)."""

    def _dockerfile(self, image) -> str:
        # SDK exposes the accumulated string as the private attr.
        return image._dockerfile

    def test_no_trailing_newline_does_not_concatenate(self, temp_dir: Path) -> None:
        from daytona.common.image import Image

        p = temp_dir / "Dockerfile"
        p.write_text(
            "FROM python:3.12\nRUN uv pip install --system pandas==2.2.3 numpy==2.2.0"
        )  # NO trailing newline (the failing-task shape)
        out = self._dockerfile(bake_agent_tooling(Image.from_dockerfile(str(p))))
        assert "numpy==2.2.0RUN" not in out
        assert "numpy==2.2.0\nRUN DEBIAN_FRONTEND=noninteractive apt-get update" in out

    def test_trailing_newline_base_still_valid(self, temp_dir: Path) -> None:
        from daytona.common.image import Image

        p = temp_dir / "Dockerfile"
        p.write_text("FROM ubuntu:22.04\nRUN echo hi\n")  # well-formed
        out = self._dockerfile(bake_agent_tooling(Image.from_dockerfile(str(p))))
        assert "echo hiRUN" not in out
        assert "apt-get install -y tmux" in out


@pytest.fixture
def snapshot_service(temp_dir: Path) -> DaytonaSnapshotService:
    env_dir = temp_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    return DaytonaSnapshotService(
        logger=logger.getChild("test"),
        environment_dir=env_dir,
        dockerfile_path=env_dir / "Dockerfile",
    )


class TestAutoSnapshotNaming:
    def test_snapshot_name_without_target(
        self, snapshot_service: DaytonaSnapshotService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DAYTONA_TARGET", raising=False)
        env_hash = snapshot_service.environment_hash()
        assert snapshot_service.auto_snapshot_name() == f"harbor__{env_hash}__snapshot"

    def test_snapshot_name_includes_daytona_target(
        self, snapshot_service: DaytonaSnapshotService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAYTONA_TARGET", "RL")
        env_hash = snapshot_service.environment_hash()
        assert (
            snapshot_service.auto_snapshot_name() == f"harbor__{env_hash}__RL__snapshot"
        )


PREWARMED_ID_SNAPSHOT_CASES = (
    (
        "DCAgent--dev_set_v2--acl-permissions-inheritance",
        "harbor__d7fa68a13650__snapshot",
    ),
    (
        "DCAgent2--terminal_bench_2--bn-fit-modify",
        "harbor__175b95a6d2e6__snapshot",
    ),
    (
        "DCAgent2--swebench-verified-random-100-folders--astropy__astropy-13236",
        "harbor__7b813a2b01e5__snapshot",
    ),
)


@pytest.mark.parametrize(("fixture_name", "snapshot_name"), PREWARMED_ID_SNAPSHOT_CASES)
async def test_pre_warmed_id_snapshot_is_reused(
    fixture_name: str, snapshot_name: str
) -> None:
    from daytona import Resources
    from daytona._async.snapshot import SnapshotState

    environment_dir = (
        Path(__file__).parents[2]
        / "golden"
        / "daytona_snapshot_environments"
        / fixture_name
    )
    service = DaytonaSnapshotService(
        logger=logger.getChild("pre-warmed-snapshot-test"),
        environment_dir=environment_dir,
        dockerfile_path=environment_dir / "Dockerfile",
    )

    class SnapshotApi:
        def __init__(self) -> None:
            self.requested_names: list[str] = []

        async def get(self, name: str):
            self.requested_names.append(name)
            if name != snapshot_name:
                raise AssertionError(f"unexpected snapshot lookup: {name}")
            return SimpleNamespace(state=SnapshotState.ACTIVE)

        async def create(self, _params) -> None:
            raise AssertionError("an existing pre-warmed snapshot must not be rebuilt")

    snapshot_api = SnapshotApi()
    daytona = SimpleNamespace(snapshot=snapshot_api)

    resolved = await service.ensure_auto(daytona, Resources())

    assert resolved == snapshot_name
    assert snapshot_api.requested_names == [snapshot_name]
