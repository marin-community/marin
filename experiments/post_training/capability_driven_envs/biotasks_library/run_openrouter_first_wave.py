# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run 12 provider-pinned OpenRouter models over 247 provisionally eligible v2 BioTasks.

Model batches are sequential for budget checks and per-model immutable recovery.
The result rows retain trial-level trajectories. GLM-audit confirmations are
provisional; this first wave is diagnostic, not a validated IRT dataset.
"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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

MANIFEST = Path(__file__).with_name("openrouter_first_wave.json")
OUTPUT_ROOT = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/"
    "campaigns/biotasks/combined/v2/calibration/openrouter-first-wave-v1/"
)
CAMPAIGN_ID = "biotasks/combined/v2/calibration/openrouter-first-wave-v1"
ELIGIBLE = frozenset({"verified_reference_corrected", "glm_audit_confirmed_unverified"})
MAX_CAMPAIGN_SPEND_USD = Decimal("200")
CAMPAIGN_START_USAGE_USD = Decimal("0.346742151")
EXPECTED_TASK_COUNT = 247


@dataclass(frozen=True)
class PinnedModel:
    """One OpenRouter model/provider route, sampled once per eligible item."""

    model: str
    provider_tag: str
    provider_name: str
    context_window: int


def load_panel() -> tuple[list[PinnedModel], str]:
    """Load and validate the pinned diagnostic panel manifest."""
    data = MANIFEST.read_bytes()
    manifest = json.loads(data)
    if manifest["panel_id"] != "biotasks-v2-openrouter-first-wave-v1":
        raise ValueError("Unexpected first-wave manifest")
    models = [PinnedModel(**row) for row in manifest["models"]]
    if len(models) != 12 or len({row.model for row in models}) != 12:
        raise ValueError("First wave requires 12 unique models")
    if any(row.context_window < 32768 or not row.provider_tag or not row.provider_name for row in models):
        raise ValueError("Panel includes a missing provider or insufficient context window")
    return models, hashlib.sha256(data).hexdigest()


def load_eligible(uri: str) -> tuple[list[TaskSpec], str]:
    """Require exactly the v2 provisional cohort; never run disputed references."""
    with fsspec.open(uri, "rb") as source:
        data = source.read()
    records = [json.loads(line) for line in data.splitlines() if line]
    counts = Counter(row["audit_disposition"] for row in records)
    if counts != {
        "glm_audit_confirmed_unverified": 245,
        "verified_reference_corrected": 2,
        "audit_disputed_unresolved": 58,
    }:
        raise ValueError(f"Unexpected v2 bank dispositions: {counts}")
    tasks = []
    for row in records:
        if row["audit_disposition"] not in ELIGIBLE:
            continue
        task = TaskSpec.model_validate(row["task"])
        if task.id != row["task_id"] or digest(task) != row["task_digest"]:
            raise ValueError(f"Frozen task identity mismatch: {row['task_id']}")
        tasks.append(task)
    if len(tasks) != EXPECTED_TASK_COUNT or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Expected 247 unique eligible tasks")
    return tasks, hashlib.sha256(data).hexdigest()


def _model_path(model: PinnedModel) -> str:
    return model.model.replace("/", "--").replace(":", "-")


def _read_completed(
    uri: str, tasks: list[TaskSpec], model: PinnedModel, manifest_sha: str
) -> list[dict[str, Any]] | None:
    fs, path = fsspec.core.url_to_fs(uri)
    if not fs.exists(path):
        return None
    with fs.open(path, "rt") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    expected = {task.id: digest(task) for task in tasks}
    if (
        len(rows) != len(expected)
        or len({row["task_id"] for row in rows}) != len(rows)
        or {row["task_id"]: row["frozen_task_digest"] for row in rows} != expected
    ):
        raise ValueError(f"Existing model output is incomplete or mismatched: {uri}")
    if any(
        row["model"] != model.model
        or row["panel_manifest_sha256"] != manifest_sha
        or row["requested_provider_tag"] != model.provider_tag
        for row in rows
    ):
        raise ValueError(f"Existing model output has a different route/manifest: {uri}")
    return rows


def _run_shard(
    rows: Iterable[dict[str, object]], shard_info: object, *, model: PinnedModel, manifest_sha: str
) -> Iterator[dict[str, object]]:
    del shard_info
    config = CalibrationConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model=model.model,
        provider_tag=model.provider_tag,
        provider_name=model.provider_name,
        context_window=model.context_window,
        campaign_id=CAMPAIGN_ID,
        max_retries=0,
    )
    for row in rows:
        task = TaskSpec.model_validate_json(str(row["task_json"]))
        with tempfile.TemporaryDirectory(prefix="biotasks-openrouter-wave-") as temporary:
            try:
                result = asyncio.run(_calibrate(task, config, Path(temporary)))
                yield {**result, "panel_manifest_sha256": manifest_sha}
            except Exception as error:
                yield {
                    "campaign_id": CAMPAIGN_ID,
                    "task_id": task.id,
                    "frozen_task_digest": digest(task),
                    "model": model.model,
                    "requested_provider_tag": model.provider_tag,
                    "panel_manifest_sha256": manifest_sha,
                    "infrastructure_error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc()[-4000:],
                    },
                }


def _verify_routes(rows: list[dict[str, Any]], model: PinnedModel, result_uri: str) -> None:
    """Resolve delayed generation metadata without changing frozen response rows."""
    if any(
        evidence.get("metadata_error") and not isinstance(evidence.get("response_id"), str)
        for row in rows
        for evidence in row.get("provider_evidence", [])
    ):
        raise RuntimeError(f"Completion lacks response ID for {model.model}")
    pending = {
        evidence["response_id"]
        for row in rows
        for evidence in row.get("provider_evidence", [])
        if evidence.get("metadata_error") and isinstance(evidence.get("response_id"), str)
    }
    immediate_mismatch = [
        evidence
        for row in rows
        for evidence in row.get("provider_evidence", [])
        if not evidence.get("route_matches") and not evidence.get("metadata_error")
    ]
    if immediate_mismatch:
        raise RuntimeError(f"Verified provider mismatch or missing response ID for {model.model}")
    if not pending:
        return
    overlay_uri = result_uri.replace("results.jsonl", "route-metadata-backfill.jsonl")
    fs, path = fsspec.core.url_to_fs(overlay_uri)
    if fs.exists(path):
        with fs.open(path, "rt") as source:
            backfill = [json.loads(line) for line in source if line.strip()]
    else:
        key = os.environ["OPENROUTER_API_KEY"]
        backfill = []
        for response_id in sorted(pending):
            for wait in (2, 5, 10, 20, 40):
                time.sleep(wait)
                request = Request(
                    "https://openrouter.ai/api/v1/generation?" + urlencode({"id": response_id}),
                    headers={"Authorization": f"Bearer {key}"},
                )
                try:
                    with urlopen(request, timeout=30) as response:
                        metadata = json.load(response)["data"]
                    break
                except HTTPError as error:
                    if error.code != 404:
                        raise
            else:
                raise RuntimeError(f"Generation metadata still unavailable: {response_id}")
            backfill.append(
                {
                    "response_id": response_id,
                    "provider_name": metadata.get("provider_name"),
                    "served_model": metadata.get("model"),
                    "request_id": metadata.get("request_id"),
                    "total_cost": metadata.get("total_cost"),
                }
            )
        if {row["response_id"] for row in backfill} != pending or any(
            row["provider_name"] != model.provider_name for row in backfill
        ):
            raise RuntimeError(f"OpenRouter generation route mismatch in metadata backfill: {model.model}")
        with fs.open(path, "wb") as destination:
            destination.write(b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in backfill))
    if {row["response_id"] for row in backfill} != pending or any(
        row["provider_name"] != model.provider_name for row in backfill
    ):
        raise RuntimeError(f"Existing OpenRouter metadata backfill disagrees with pinned route: {model.model}")
    print(f"BACKFILL model={model.model} generations={len(backfill)} uri={overlay_uri}", flush=True)


def _key_budget() -> tuple[Decimal, Decimal]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY must be passed only as a secret environment variable")
    request = Request("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"})
    with urlopen(request, timeout=30) as response:
        metadata = json.load(response)["data"]
    return Decimal(str(metadata["usage"])), Decimal(str(metadata["limit_remaining"]))


def _provider_estimate(model: PinnedModel) -> Decimal:
    """Double the p95-v1 token-cost estimate before scheduling a model batch."""
    url = f"https://openrouter.ai/api/v1/models/{model.model}/endpoints"
    with urlopen(url, timeout=30) as response:
        endpoints = json.load(response)["data"]["endpoints"]
    selected = [row for row in endpoints if row["tag"] == model.provider_tag]
    if len(selected) != 1:
        raise RuntimeError(f"Pinned provider endpoint unavailable for {model.model}")
    endpoint = selected[0]
    if not set(("tools", "temperature", "top_p", "max_tokens")).issubset(endpoint["supported_parameters"]):
        raise RuntimeError(f"Pinned provider does not support fixed policy: {model.model}")
    if endpoint["context_length"] < 32768 or endpoint["max_completion_tokens"] < 4096:
        raise RuntimeError(f"Pinned provider lacks sufficient context/output: {model.model}")
    pricing = endpoint["pricing"]
    return (
        2
        * EXPECTED_TASK_COUNT
        * (Decimal("12877") * Decimal(pricing["prompt"]) + Decimal("4280") * Decimal(pricing["completion"]))
    )


def _run_model(tasks: list[TaskSpec], model: PinnedModel, manifest_sha: str, output: str, max_workers: int) -> None:
    rows = [{"task_json": task.model_dump_json()} for task in tasks]
    pipeline = (
        Dataset.from_list(rows)
        .reshard(len(rows))
        .map_shard(partial(_run_shard, model=model, manifest_sha=manifest_sha))
        .reshard(1)
        .write_jsonl(output)
    )
    with ZephyrContext(
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="2g", disk="8g", preemptible=False),
        name=f"biotasks-openrouter-wave-{_model_path(model)}",
        heartbeat_timeout=1800,
    ) as context:
        print(context.execute(pipeline), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=V2_ROOT + "taskspecs.jsonl")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 32:
        parser.error("--max-workers must be between 1 and 32")
    tasks, task_sha = load_eligible(args.tasks)
    models, manifest_sha = load_panel()
    usage, remaining = _key_budget()
    start_usage = CAMPAIGN_START_USAGE_USD
    if usage < start_usage or usage - start_usage >= MAX_CAMPAIGN_SPEND_USD:
        raise RuntimeError("Campaign baseline or 200 USD spend ceiling violated")
    print(f"first_wave tasks={len(tasks)} models={len(models)} task_sha={task_sha} panel_sha={manifest_sha}", flush=True)
    for model in models:
        uri = f"{args.output_root.rstrip('/')}/attempts/{_model_path(model)}/results.jsonl"
        existing = _read_completed(uri, tasks, model, manifest_sha)
        if existing is None:
            usage, remaining = _key_budget()
            spent = usage - start_usage
            reserve = max(Decimal("10"), _provider_estimate(model))
            if spent + reserve > MAX_CAMPAIGN_SPEND_USD or reserve > remaining:
                raise RuntimeError(
                    f"Budget gate: spent={spent}, reserve={reserve}, key_remaining={remaining}; "
                    f"not starting {model.model}"
                )
            print(
                f"START model={model.model} provider={model.provider_tag} reserve_usd={reserve} at={time.time()}",
                flush=True,
            )
            _run_model(tasks, model, manifest_sha, uri, args.max_workers)
            existing = _read_completed(uri, tasks, model, manifest_sha)
            if existing is None:
                raise RuntimeError(f"No immutable model output after successful pipeline: {uri}")
        _verify_routes(existing, model, uri)
        counts = Counter(
            "infrastructure_error" if "infrastructure_error" in row else row["trial"]["status"] for row in existing
        )
        usage, remaining = _key_budget()
        print(
            f"DONE model={model.model} rows={len(existing)} outcomes={dict(counts)} "
            f"spent_usd={usage - start_usage} key_remaining={remaining} uri={uri}",
            flush=True,
        )
    print(f"FIRST_WAVE_COMPLETE task_sha={task_sha} panel_sha={manifest_sha}", flush=True)


if __name__ == "__main__":
    main()
