# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a disjoint 20-model panel against the same frozen v3 BioTasks bank.

Run --pilot-only first. Finished pilot trials are reused on the full run; a
new generation is never submitted for an already complete model/task pair.
"""

import argparse
import asyncio
import hashlib
import json
import tempfile
import traceback
from collections.abc import Iterable, Iterator
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any

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
from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import _write_new
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    PinnedModel,
    _key_budget,
    _model_path,
    _provider_estimate,
    _read_completed,
    _verify_routes,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    load_panel as load_first_panel,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave_v3 import (
    OUTPUT_ROOT as FIRST_ROOT,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave_v3 import (
    load_cohort,
)

HERE = Path(__file__).parent
PANEL = HERE / "openrouter_v3_additional20.json"
ROOT = FIRST_ROOT.replace("openrouter-first-wave-v3/", "openrouter-additional20-v1/")
CAMPAIGN_ID = "biotasks/combined/v3/calibration/openrouter-additional20-v1"
FIRST_RESPONSE_SHA = "acf9b5aff6a47230e2e0d0c751554e03969d4af9fbbb22099154db1aed039cda"
MAX_NEW_SPEND = Decimal("40")
PILOT_TARGETS = (0.2, 0.4, 0.6, 0.8)  # Three distinct items per difficulty stratum.


def load_panel() -> tuple[list[PinnedModel], str]:
    """Pin identities and prohibit any repeated respondent from the first 12."""
    data = PANEL.read_bytes()
    panel = json.loads(data)
    original, _ = load_first_panel()
    models = [PinnedModel(**row) for row in panel["models"]]
    if (
        panel["panel_id"] != "biotasks-v3-openrouter-additional20-v1"
        or panel["task_snapshot_sha256"] != "b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6"
        or panel["prior_response_sha256"] != FIRST_RESPONSE_SHA
        or Decimal(panel["max_new_spend_usd"]) != MAX_NEW_SPEND
        or len(models) != 20
        or len({m.model for m in models}) != 20
        or {m.model for m in models} & {m.model for m in original}
        or any(m.context_window < 65536 or not m.provider_tag or not m.provider_name for m in models)
    ):
        raise ValueError("Additional panel changed, overlaps, or lacks a pinned route")
    return models, hashlib.sha256(data).hexdigest()


def select_pilot(rows: list[dict[str, Any]], available_ids: set[str]) -> list[str]:
    """Select 12 prior-difficulty-spanning items without disclosing task IDs."""
    rates: dict[str, list[int]] = {}
    if len(rows) != 2760 or len({(r["model"], r["task_id"]) for r in rows}) != len(rows):
        raise ValueError("Invalid first-wave response matrix")
    for row in rows:
        if row["task_id"] not in available_ids:
            raise ValueError("First-wave task is not eligible for the additional panel")
        if row["score"] in (0, 1):
            rates.setdefault(row["task_id"], []).append(row["score"])
    if set(rates) != available_ids or any(len(scores) < 4 for scores in rates.values()):
        raise ValueError("Insufficient prior scores to stratify pilot")
    ordered: list[str] = []
    for target in PILOT_TARGETS:
        for _ in range(3):
            choice = min(
                (item for item in rates if item not in ordered),
                key=lambda item: (abs(sum(rates[item]) / len(rates[item]) - target), item),
            )
            ordered.append(choice)
    return sorted(ordered)


def load_pilot_ids(task_ids: set[str]) -> list[str]:
    """Verify frozen first-wave response bytes before deriving pilot items."""
    with fsspec.open(FIRST_ROOT + "snapshots/complete-1/item-responses.jsonl", "rb") as f:
        content = f.read()
    if hashlib.sha256(content).hexdigest() != FIRST_RESPONSE_SHA:
        raise ValueError("Unexpected source response snapshot")
    return select_pilot([json.loads(line) for line in content.splitlines() if line], task_ids)


def _run_shard(
    rows: Iterable[dict[str, object]], shard_info: object, *, model: PinnedModel, panel_sha: str, campaign_id: str
) -> Iterator[dict[str, object]]:
    del shard_info
    config = CalibrationConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model=model.model,
        provider_tag=model.provider_tag,
        provider_name=model.provider_name,
        context_window=model.context_window,
        campaign_id=campaign_id,
        max_retries=0,
    )
    for row in rows:
        task = TaskSpec.model_validate_json(str(row["task_json"]))
        with tempfile.TemporaryDirectory(prefix="biotasks-additional20-") as temporary:
            try:
                result = asyncio.run(_calibrate(task, config, Path(temporary)))
                yield {**result, "panel_manifest_sha256": panel_sha}
            except Exception as error:
                yield {
                    "campaign_id": campaign_id,
                    "task_id": task.id,
                    "frozen_task_digest": digest(task),
                    "model": model.model,
                    "requested_provider_tag": model.provider_tag,
                    "panel_manifest_sha256": panel_sha,
                    "infrastructure_error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc()[-4000:],
                    },
                }


def _run(
    tasks: list[TaskSpec],
    model: PinnedModel,
    panel_sha: str,
    uri: str,
    workers: int,
    *,
    campaign_id: str,
    name_prefix: str,
) -> None:
    pipeline = (
        Dataset.from_list([{"task_json": task.model_dump_json()} for task in tasks])
        .reshard(len(tasks))
        .map_shard(partial(_run_shard, model=model, panel_sha=panel_sha, campaign_id=campaign_id))
        .reshard(1)
        .write_jsonl(uri)
    )
    with ZephyrContext(
        max_workers=min(workers, len(tasks)),
        resources=ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="2g", disk="8g", preemptible=False),
        name=f"{name_prefix}-{_model_path(model)}",
        heartbeat_timeout=1800,
    ) as context:
        print(context.execute(pipeline), flush=True)


def budget_gate(model: PinnedModel, pairs: int, panel_sha: str) -> None:
    """Immutable campaign baseline and conservative pre-batch spend reserve."""
    uri = ROOT + "budget.json"
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        with fs.open(path, "rt") as f:
            budget = json.load(f)
    else:
        usage, remaining = _key_budget()
        if remaining < MAX_NEW_SPEND:
            raise RuntimeError("Insufficient OpenRouter key credit for the new campaign")
        budget = {"panel_sha256": panel_sha, "baseline_usage_usd": str(usage), "max_new_spend_usd": str(MAX_NEW_SPEND)}
        _write_new(uri, json.dumps(budget, sort_keys=True).encode() + b"\n")
    if budget["panel_sha256"] != panel_sha or Decimal(budget["max_new_spend_usd"]) != MAX_NEW_SPEND:
        raise ValueError("Panel or campaign spend ceiling changed mid-run")
    usage, remaining = _key_budget()
    reserve = max(Decimal("1"), _provider_estimate(model) * Decimal(pairs) / Decimal(247))
    spent = usage - Decimal(budget["baseline_usage_usd"])
    if spent < 0 or spent + reserve > MAX_NEW_SPEND or reserve > remaining:
        raise RuntimeError(f"Additional20 budget gate: spent={spent}, reserve={reserve}, remaining={remaining}")
    print(f"BUDGET spent_usd={spent} reserve_usd={reserve} remaining_usd={remaining}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        parser.error("max-workers must be in [1, 8]")
    cohort = load_cohort()
    models, panel_sha = load_panel()
    tasks = {row["task_id"]: TaskSpec.model_validate(row["task"]) for row in cohort.rows}
    pilot_ids = load_pilot_ids(set(tasks))
    for model in models:
        prefix = ROOT + f"attempts/{_model_path(model)}/"
        pilot = [tasks[item] for item in pilot_ids]
        pilot_uri = prefix + "pilot.jsonl"
        completed = _read_completed(pilot_uri, pilot, model, panel_sha)
        if completed is None:
            budget_gate(model, len(pilot), panel_sha)
            _run(
                pilot,
                model,
                panel_sha,
                pilot_uri,
                min(2, args.max_workers),
                campaign_id=CAMPAIGN_ID,
                name_prefix="biotasks-additional20",
            )
            completed = _read_completed(pilot_uri, pilot, model, panel_sha)
            if completed is None:
                raise RuntimeError(f"Missing pilot: {model.model}")
        try:
            _verify_routes(completed, model, pilot_uri)
        except RuntimeError as error:
            if "Generation metadata still unavailable" not in str(error):
                raise
            print(f"PILOT_METADATA_UNAVAILABLE model={model.model}; no full batch", flush=True)
            continue
        if not any(row.get("provider_evidence") for row in completed):
            print(f"PILOT_UNROUTED model={model.model} completed={len(completed)}; no full batch", flush=True)
            continue
        print(f"PILOT model={model.model} completed={len(completed)}", flush=True)
        if args.pilot_only:
            continue
        rest = [tasks[item] for item in sorted(tasks) if item not in pilot_ids]
        uri = prefix + "results.jsonl"
        existing = _read_completed(uri, rest, model, panel_sha)
        if existing is None:
            budget_gate(model, len(rest), panel_sha)
            _run(
                rest,
                model,
                panel_sha,
                uri,
                args.max_workers,
                campaign_id=CAMPAIGN_ID,
                name_prefix="biotasks-additional20",
            )
            existing = _read_completed(uri, rest, model, panel_sha)
            if existing is None:
                raise RuntimeError(f"Missing full batch: {model.model}")
        _verify_routes(existing, model, uri)
        print(f"DONE model={model.model} pairs={len(pilot) + len(rest)}", flush=True)
    print("ADDITIONAL20_PILOT_COMPLETE" if args.pilot_only else "ADDITIONAL20_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
