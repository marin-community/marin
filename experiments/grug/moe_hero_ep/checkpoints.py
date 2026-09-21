# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List permanent hero checkpoints from a W&B report or a run's checkpoint ancestry."""

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

REPORT_URL = (
    "https://wandb.ai/marin-community/marin_moe/reports/"
    "535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng"
)
_GRAPHQL_URL = "https://api.wandb.ai/graphql"
_ENTITY = "marin-community"
_PROJECT = "marin_moe"
_PARENT_CHECKPOINT_PATTERN = re.compile(r".*/(?P<run>[^/]+)/[^/]+/checkpoints/step-(?P<step>\d+)/?")
_REPORT_QUERY = """
query Report($id: ID!) {
  view(id: $id) { spec }
}
"""
_RUNS_QUERY = """
query HeroRuns($entity: String!, $project: String!, $filters: JSONString!, $after: String) {
  project(entityName: $entity, name: $project) {
    runs(filters: $filters, first: 100, after: $after) {
      edges {
        cursor
        node { name config }
      }
      pageInfo { hasNextPage }
    }
  }
}
"""
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


def _hero_checkpoint_runs() -> list[_HeroCheckpointRun]:
    """Read selected hero phases and checkpoint roots from the public W&B report.

    Explicit selections across enabled, ungrouped report run sets supply run IDs.
    Runs with ``report_phase_start`` supply the lineage. Forecast and ladder runs
    without that field are excluded. Missing phase bounds or checkpoint config
    raise an error. The report read does not require a W&B API key.
    """
    with httpx.Client(timeout=30.0) as client:
        view_id = REPORT_URL.rsplit("--", 1)[1]
        view_id += "=" * (-len(view_id) % 4)
        view = _graphql(client, _REPORT_QUERY, {"id": view_id})["view"]
        if view is None:
            raise ValueError(f"W&B report not found: {REPORT_URL}")
        spec = json.loads(view["spec"])
        run_ids: set[str] = set()
        for block in spec["blocks"]:
            if block.get("type") != "panel-grid":
                continue
            for run_set in block["metadata"]["runSets"]:
                if not run_set.get("enabled", True) or run_set.get("grouping"):
                    continue
                selections = run_set["selections"]
                if selections["root"] == 0:
                    run_ids.update(selections["tree"])
        if not run_ids:
            raise ValueError("The W&B report has no explicit run selections")

        filters = {"name": {"$in": sorted(run_ids)}, "config.report_phase_start": {"$exists": True}}
        variables = {"entity": _ENTITY, "project": _PROJECT, "filters": json.dumps(filters), "after": None}
        runs: list[_HeroCheckpointRun] = []
        while True:
            connection = _graphql(client, _RUNS_QUERY, variables)["project"]["runs"]
            for edge in connection["edges"]:
                node = edge["node"]
                config = {key: value["value"] for key, value in json.loads(node["config"]).items()}
                start = config["report_phase_start"]
                end = config["report_phase_end"]
                if type(start) is not int or type(end) is not int or not 0 <= start <= end:
                    raise ValueError(f"Invalid hero phase bounds for {node['name']}: {start}, {end}")
                trainer = config["trainer"]["trainer"]
                runs.append(_HeroCheckpointRun(node["name"], _checkpoint_root(trainer), start, end))
            if not connection["pageInfo"]["hasNextPage"]:
                break
            variables["after"] = connection["edges"][-1]["cursor"]
    if not runs:
        raise ValueError("The selected W&B runs have no hero phase metadata")
    return sorted(runs, key=lambda run: (run.start_step, run.run_id))


def hero_checkpoint_paths() -> list[str]:
    """Return permanent checkpoint paths in the W&B report's hero phases, sorted by step.

    Read directory listings and metadata only. Include each phase's start and end
    checkpoints. Keep distinct paths at the same step. Bucket credentials must be
    available through the usual Marin storage configuration.
    """
    return _checkpoint_paths(_hero_checkpoint_runs())


def _parent_run_and_step(config: dict, run_id: str) -> tuple[str, int] | None:
    handoff = config.get("hero_handoff_checkpoint")
    paths = [handoff] if handoff else config["trainer"]["trainer"]["load_checkpoint_path"]
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


def hero_checkpoint_paths_from_run(run_id: str = CURRENT_HERO_RUN_ID) -> list[str]:
    """Return permanent checkpoint paths for a run and its ancestors, in step order.

    The default run ID comes from ``current_run.CURRENT_HERO_RUN_ID``.
    Read parent links from ``hero_handoff_checkpoint`` or explicit foreign
    checkpoints in ``trainer.trainer.load_checkpoint_path``. Paths must use the
    hero layout ``<run>/<version>/checkpoints/step-N``. Directory search roots do
    not identify parents. Parent links supply inclusive phase bounds. No report
    or ``report_phase_*`` fields are read. Ambiguous links and cycles raise an error.
    """
    runs: list[_HeroCheckpointRun] = []
    seen: set[str] = set()
    end_step: int | None = None
    with httpx.Client(timeout=30.0) as client:
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
