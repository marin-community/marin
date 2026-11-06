# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic first wave: one corrected BioTask, one response from each pinned model.

This is not the official item-response panel. Route mismatches or missing
OpenRouter generation metadata remain visible failures, never scored as zeros.
"""

import argparse
import asyncio
import hashlib
import json
import tempfile
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import fsspec
from fray.types import ResourceConfig
from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.biotasks_library.calibrate_glm_v1 import (
    CalibrationConfig,
    _calibrate,
)
from experiments.post_training.capability_driven_envs.biotasks_library.freeze_combined_v2 import V2_ROOT

MANIFEST = Path(__file__).with_name("openrouter_first12.json")
TASK_ID = "biotasks-read_counting_modes-78a95b29f211f753-v2"
OUTPUT = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v2/"
    "calibration/openrouter-first12-one-item-diagnostic-v1/results.jsonl"
)
CAMPAIGN_ID = "biotasks/combined/v2/calibration/openrouter-first12-one-item-diagnostic-v1"


@dataclass(frozen=True)
class PinnedModel:
    """One exact model and OpenRouter serving route."""

    model: str
    provider_tag: str
    provider_name: str
    context_window: int


def _read_panel() -> tuple[list[PinnedModel], str]:
    data = MANIFEST.read_bytes()
    manifest = json.loads(data)
    if manifest["panel_id"] != "biotasks-v2-openrouter-first12-diagnostic":
        raise ValueError("Unexpected OpenRouter manifest")
    models = manifest["models"]
    if len(models) != 12 or len({row["model"] for row in models}) != 12:
        raise ValueError("Expected 12 different model IDs")
    rows = [PinnedModel(**row) for row in models]
    if any(row.context_window < 32768 or not row.provider_tag or not row.provider_name for row in rows):
        raise ValueError("Invalid pinned provider or context window")
    return rows, hashlib.sha256(data).hexdigest()


def _load_task(uri: str) -> TaskSpec:
    with fsspec.open(uri, "rt") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    selected = [row for row in rows if row["task_id"] == TASK_ID]
    if len(selected) != 1 or selected[0]["audit_disposition"] != "verified_reference_corrected":
        raise ValueError("Diagnostic item must be the independently corrected v2 task")
    task = TaskSpec.model_validate(selected[0]["task"])
    if digest(task) != selected[0]["task_digest"]:
        raise ValueError("Frozen v2 task digest mismatch")
    return task


def _shard(rows: Iterable[dict[str, object]], shard_info: object) -> Iterator[dict[str, object]]:
    del shard_info
    for row in rows:
        task = TaskSpec.model_validate_json(str(row["task_json"]))
        pinned_data = row["pinned"]
        if not isinstance(pinned_data, dict):
            raise ValueError("Expected a pinned model record")
        pinned = PinnedModel(
            model=str(pinned_data["model"]),
            provider_tag=str(pinned_data["provider_tag"]),
            provider_name=str(pinned_data["provider_name"]),
            context_window=int(pinned_data["context_window"]),
        )
        config = CalibrationConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            model=pinned.model,
            provider_tag=pinned.provider_tag,
            provider_name=pinned.provider_name,
            context_window=pinned.context_window,
            campaign_id=CAMPAIGN_ID,
            max_retries=0,
        )
        with tempfile.TemporaryDirectory(prefix="biotasks-openrouter-first12-") as temporary:
            try:
                result = asyncio.run(_calibrate(task, config, Path(temporary)))
                yield {**result, "panel_manifest_sha256": row["panel_manifest_sha256"]}
            except Exception as error:
                yield {
                    "campaign_id": CAMPAIGN_ID,
                    "task_id": task.id,
                    "frozen_task_digest": digest(task),
                    "model": pinned.model,
                    "requested_provider_tag": pinned.provider_tag,
                    "panel_manifest_sha256": row["panel_manifest_sha256"],
                    "infrastructure_error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc()[-4000:],
                    },
                }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=V2_ROOT + "taskspecs.jsonl")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    fs, path = fsspec.core.url_to_fs(args.output)
    if fs.exists(path):
        raise FileExistsError(f"refusing to overwrite diagnostic results: {args.output}")
    task = _load_task(args.tasks)
    models, sha = _read_panel()
    rows = [
        {"task_json": task.model_dump_json(), "pinned": vars(model), "panel_manifest_sha256": sha} for model in models
    ]
    pipeline = Dataset.from_list(rows).reshard(12).map_shard(_shard).reshard(1).write_jsonl(args.output)
    with ZephyrContext(
        max_workers=12,
        resources=ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="2g", disk="8g", preemptible=False),
        name="biotasks-openrouter-first12-one-item",
        heartbeat_timeout=1800,
    ) as context:
        result = context.execute(pipeline)
    print(f"task_id={task.id} models={len(models)} manifest_sha256={sha} output={args.output}")
    print(result)


if __name__ == "__main__":
    main()
