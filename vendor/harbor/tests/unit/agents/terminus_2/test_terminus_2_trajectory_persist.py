"""Regression tests for trajectory.json persistence to remote (UPath) log dirs.

Commit 3d170c8f coerced the agent ``logs_dir`` from a remote ``UPath``
(``gs://.../agent``) to a stdlib ``Path`` via ``Path(str(agent_dir))``. That
collapses the scheme (``gs://`` -> ``gs:/``) and turns the path into a local
filesystem path, so the trajectory dump silently wrote ``trajectory.json`` to a
bogus local directory instead of the trials_dir. The file never reached object
storage and downstream trace datasets came out empty.

These tests pin the contract that an agent given a remote-scheme ``logs_dir``
persists ``trajectory.json`` *to that scheme* via the background artifact
writer. ``memory://`` is an in-process fsspec backend, so it exercises the real
UPath write path with no network.
"""

import json
import uuid

from upath import UPath

from harbor.utils.artifact_writer import get_artifact_writer
from tests.unit.agents.terminus_2.bare_agent import make_bare_terminus2


def test_persist_trajectory_writes_to_remote_upath():
    """A remote-scheme logs_dir must receive trajectory.json at that scheme."""
    logs_dir = UPath(f"memory://harbor-traj-{uuid.uuid4().hex}/agent")
    logs_dir.mkdir(parents=True, exist_ok=True)

    agent = make_bare_terminus2(logs_dir)
    agent._persist_trajectory(agent._summarization_count)
    get_artifact_writer().flush(prefix=str(logs_dir), timeout=30)

    trajectory_path = logs_dir / "trajectory.json"
    assert trajectory_path.exists(), (
        "trajectory.json was not written to the remote logs_dir"
    )

    data = json.loads(trajectory_path.read_text())
    assert data["session_id"] == agent._session_id
    assert len(data["steps"]) == 1
    assert data["steps"][0]["message"] == "hi"
