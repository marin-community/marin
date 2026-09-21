# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List permanent hero checkpoints from a run's checkpoint ancestry."""

import json
import logging
import re
from dataclasses import dataclass

import httpx
from levanter.checkpoint import CheckpointCandidate, discover_checkpoint_candidates
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_ep.current_run import CURRENT_HERO_RUN_ID

logger = logging.getLogger(__name__)

_GRAPHQL_URL = "https://api.wandb.ai/graphql"
_WANDB_TIMEOUT = 30.0
_ENTITY = "marin-community"
_PROJECT = "marin_moe"
_PARENT_CHECKPOINT_PATTERN = re.compile(r".*/(?P<run>[^/]+)/[^/]+/checkpoints/step-(?P<step>\d+)/?")
_RUN_QUERY = """
query HeroRun($entity: String!, $project: String!, $run: String!) {
  project(entityName: $entity, name: $project) {
    run(name: $run) { config(keys: ["trainer", "hero_handoff_checkpoint"]) }
  }
}
"""


@dataclass(frozen=True)
class _HeroCheckpointRun:
    """A run phase and its permanent checkpoint root, with inclusive step bounds."""

    run_id: str
    checkpoint_root: str
    start_step: int
    end_step: int | None


def _graphql(client: httpx.Client, query: str, variables: dict) -> dict:
    response = client.post(_GRAPHQL_URL, json={"query": query, "variables": variables})
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise ValueError(f"W&B GraphQL errors: {payload['errors']}")
    return payload["data"]


def _checkpoint_root(trainer: dict) -> str:
    checkpointer = trainer["checkpointer"]
    root = checkpointer["base_path"]
    if not root:
        raise ValueError(f"Missing checkpoint root for {trainer['id']}")
    if checkpointer["append_run_id_to_base_path"]:
        root = prefix_join(root, trainer["id"])
    return root


def _parent_run_and_step(config: dict, run_id: str) -> tuple[str, int] | None:
    handoff = config.get("hero_handoff_checkpoint")
    paths = handoff or config["trainer"]["trainer"]["load_checkpoint_path"]
    if paths is None:
        return None
    if isinstance(paths, str):
        paths = [paths]
    parents: set[tuple[str, int]] = set()
    for path in paths:
        if not handoff and not path.rstrip("/").rsplit("/", 1)[-1].startswith("step-"):
            continue
        match = _PARENT_CHECKPOINT_PATTERN.fullmatch(path)
        if match is None:
            raise ValueError(f"Unknown hero checkpoint path: {path}")
        if match["run"] != run_id:
            parents.add((match["run"], int(match["step"])))
        elif handoff:
            raise ValueError(f"Run {run_id} names itself as its handoff source")
    if len(parents) > 1:
        raise ValueError(f"Ambiguous parent checkpoints for {run_id}: {sorted(parents)}")
    return next(iter(parents)) if parents else None


def hero_checkpoint_paths(run_id: str = CURRENT_HERO_RUN_ID) -> list[str]:
    """Return permanent checkpoint paths for a run and its ancestors, in step order.

    The default run ID comes from ``current_run.CURRENT_HERO_RUN_ID``.
    Read parent links from ``hero_handoff_checkpoint`` or explicit foreign
    checkpoints in ``trainer.trainer.load_checkpoint_path``. Paths must use the
    hero layout ``<run>/<version>/checkpoints/step-N``. Directory search roots do
    not identify parents. Parent links supply inclusive phase bounds. Ambiguous
    links and cycles raise an error. Read directory listings and metadata only.
    Keep distinct paths at the same step. Bucket credentials must be available
    through the usual Marin storage configuration.
    """
    runs: list[_HeroCheckpointRun] = []
    seen: set[str] = set()
    end_step: int | None = None
    with httpx.Client(timeout=_WANDB_TIMEOUT) as client:
        while True:
            if run_id in seen:
                raise ValueError(f"Checkpoint ancestry contains a cycle at {run_id}")
            seen.add(run_id)
            variables = {"entity": _ENTITY, "project": _PROJECT, "run": run_id}
            node = _graphql(client, _RUN_QUERY, variables)["project"]["run"]
            if node is None:
                raise ValueError(f"W&B run not found: {_ENTITY}/{_PROJECT}/{run_id}")
            config = {key: value["value"] for key, value in json.loads(node["config"]).items()}
            trainer = config["trainer"]["trainer"]
            parent = _parent_run_and_step(config, run_id)
            start_step = parent[1] if parent else 0
            if end_step is not None and start_step > end_step:
                raise ValueError(f"Run {run_id} starts at {start_step}, after its descendant's handoff at {end_step}")
            runs.append(_HeroCheckpointRun(run_id, _checkpoint_root(trainer), start_step, end_step))
            if parent is None:
                break
            run_id, end_step = parent
    return _checkpoint_paths(runs)


def _checkpoint_paths(runs: list[_HeroCheckpointRun]) -> list[str]:
    checkpoints: dict[str, CheckpointCandidate] = {}
    for run in runs:
        fs, _ = filesystem_for(run.checkpoint_root)
        candidates = discover_checkpoint_candidates(run.checkpoint_root, max_step=run.end_step, fs=fs)
        permanent = [
            candidate
            for candidate in candidates
            if candidate.step >= run.start_step and candidate.metadata.get("is_temporary") is False
        ]
        if not permanent:
            logger.warning("No permanent checkpoints in phase %s at %s", run.run_id, run.checkpoint_root)
        checkpoints.update((candidate.path, candidate) for candidate in permanent)
    return [
        candidate.path
        for candidate in sorted(
            checkpoints.values(), key=lambda candidate: (candidate.step, candidate.timestamp, candidate.path)
        )
    ]
