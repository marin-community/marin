# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only changed or infrastructure-missing BioTasks v3 model/task pairs.

The 60 tasks with byte-equivalent model-visible prompts, tools, resources,
working directories and reference answers reuse v2 generations. The other
170 changed sandboxes require new trials. All new rows are immutable.
"""

import argparse
import asyncio
import hashlib
import json
import tempfile
import traceback
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any

import fsspec
from fray.types import ResourceConfig
from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages, submission_instruction
from taskforge.content_hash import digest
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.biotasks_library.calibrate_glm_v1 import (
    CalibrationConfig,
    _calibrate,
)
from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import _write_new
from experiments.post_training.capability_driven_envs.biotasks_library.freeze_combined_v2 import V2_ROOT
from experiments.post_training.capability_driven_envs.biotasks_library.freeze_combined_v3 import V3_ROOT
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    PinnedModel,
    _key_budget,
    _model_path,
    _provider_estimate,
    _read_completed,
    _verify_routes,
    load_panel,
)

OUTPUT_ROOT = V3_ROOT.replace("snapshots/reference-path-review-1/", "calibration/openrouter-first-wave-v3/")
CAMPAIGN_ID = "biotasks/combined/v3/calibration/openrouter-first-wave-v3"
MAX_NEW_SPEND = Decimal("50")
V3_SHA = "b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6"


@dataclass(frozen=True)
class Cohort:
    """Frozen v3 bank and proven v2 generation-reuse plan."""

    rows: list[dict[str, Any]]
    reusable_ids: frozenset[str]
    changed_ids: frozenset[str]


def load_cohort(uri: str = V3_ROOT + "taskspecs.jsonl", prior_uri: str = V2_ROOT + "taskspecs.jsonl") -> Cohort:
    """Verify that an old model response is reusable only if generation was identical."""
    with fsspec.open(uri, "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != V3_SHA:
        raise ValueError("Unexpected frozen v3 bank")
    with fsspec.open(prior_uri, "rb") as f:
        old = {r["task_id"]: TaskSpec.model_validate(r["task"]) for r in map(json.loads, f)}
    records = [json.loads(line) for line in data.splitlines() if line]
    selected = [r for r in records if r["audit_disposition"] == "provisionally_eligible_v3"]
    if len(selected) != 230 or len({r["task_id"] for r in selected}) != len(selected):
        raise ValueError("Unexpected v3 eligibility")
    reusable = set()
    changed = set()
    for row in selected:
        task = TaskSpec.model_validate(row["task"])
        prior = old[row["v2_task_id"]]
        if task.id != row["task_id"] or digest(task) != row["task_digest"] or digest(prior) != row["v2_task_digest"]:
            raise ValueError("V3 identity mismatch")
        # Changing the private grader reference or any model-visible field
        # disqualifies reuse; merely changing the reference-blind answer
        # extractor does not change this identical generation request.
        same_request = (
            conversation_messages(task.context.events) == conversation_messages(prior.context.events)
            and submission_instruction(task.answer_format) == submission_instruction(prior.answer_format)
            and task.interaction_tools == prior.interaction_tools
            and task.final_tools == prior.final_tools
            and task.tool_bindings == prior.tool_bindings
            and task.resources == prior.resources
            and task.environment_requirements == prior.environment_requirements
            and task.grader == prior.grader
            and task.output_paths == prior.output_paths
        )
        if same_request:
            reusable.add(task.id)
        else:
            changed.add(task.id)
    if len(reusable) != 60 or len(changed) != 170:
        raise ValueError(f"Unexpected dedup split: reuse={len(reusable)} changed={len(changed)}")
    return Cohort(selected, frozenset(reusable), frozenset(changed))


def _source_responses() -> dict[tuple[str, str], dict[str, Any]]:
    uri = V2_ROOT.replace("v2/snapshots/verified-corrections-1/", "v2/calibration/openrouter-first-wave-v1/")
    with fsspec.open(uri + "snapshots/complete-1/item-responses.jsonl", "rt") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) != 2964:
        raise ValueError("Incomplete frozen v2 response matrix")
    return {(r["model"], r["task_id"]): r for r in rows}


def _run_shard(rows, shard_info, *, model: PinnedModel, panel_sha: str):
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
        with tempfile.TemporaryDirectory(prefix="biotasks-v3-wave-") as temporary:
            try:
                result = asyncio.run(_calibrate(task, config, Path(temporary)))
                yield {**result, "panel_manifest_sha256": panel_sha}
            except Exception as error:
                yield {
                    "campaign_id": CAMPAIGN_ID,
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


def _run(tasks: list[TaskSpec], model: PinnedModel, panel_sha: str, output: str, workers: int) -> None:
    rows = [{"task_json": task.model_dump_json()} for task in tasks]
    pipeline = (
        Dataset.from_list(rows)
        .reshard(len(rows))
        .map_shard(partial(_run_shard, model=model, panel_sha=panel_sha))
        .reshard(1)
        .write_jsonl(output)
    )
    with ZephyrContext(
        max_workers=min(workers, len(tasks)),
        resources=ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="2g", disk="8g", preemptible=False),
        name=f"biotasks-v3-{_model_path(model)}",
        heartbeat_timeout=1800,
    ) as context:
        print(context.execute(pipeline), flush=True)


def _budget(model: PinnedModel, pairs: int) -> None:
    uri = OUTPUT_ROOT + "budget.json"
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        with fs.open(path, "rt") as f:
            budget = json.load(f)
    else:
        usage, remaining = _key_budget()
        if remaining < MAX_NEW_SPEND:
            raise RuntimeError("Insufficient key credit for the v3 circuit breaker")
        budget = {"baseline_usage_usd": str(usage), "max_new_spend_usd": str(MAX_NEW_SPEND)}
        _write_new(uri, json.dumps(budget, sort_keys=True).encode() + b"\n")
    if Decimal(budget["max_new_spend_usd"]) != MAX_NEW_SPEND:
        raise ValueError("Campaign spend ceiling changed")
    usage, remaining = _key_budget()
    reserve = max(Decimal("2"), _provider_estimate(model) * Decimal(pairs) / Decimal(247))
    spent = usage - Decimal(budget["baseline_usage_usd"])
    if spent < 0 or spent + reserve > MAX_NEW_SPEND or reserve > remaining:
        raise RuntimeError(f"V3 budget gate: spent={spent}, reserve={reserve}, remaining={remaining}")
    print(f"BUDGET spent_usd={spent} reserve_usd={reserve} remaining_usd={remaining}", flush=True)


def _batches(tasks: list[TaskSpec], model: PinnedModel, prefix: str) -> list[tuple[str, list[TaskSpec]]]:
    """Bound a premium provider's per-batch reserve without altering its trials."""
    if model.model != "anthropic/claude-sonnet-4.6":
        return [(prefix + "results.jsonl", tasks)]
    chunk = (len(tasks) + 2) // 3
    return [
        (prefix + f"results-part{index + 1}.jsonl", tasks[start : start + chunk])
        for index, start in enumerate(range(0, len(tasks), chunk))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("max workers must be between 1 and 8")
    cohort = load_cohort()
    models, panel_sha = load_panel()
    prior = _source_responses()
    for model in models:
        tasks = [TaskSpec.model_validate(r["task"]) for r in cohort.rows]
        by_id = {t.id: t for t in tasks}
        pending = []
        for row in cohort.rows:
            task = by_id[row["task_id"]]
            if task.id in cohort.changed_ids:
                pending.append(task)
                continue
            old_result = prior[(model.model, row["v2_task_id"])]
            if old_result["task_spec_digest"] != row["v2_task_digest"]:
                raise ValueError("Dedup prior task digest mismatch")
            if old_result["provider_tag"] != model.provider_tag or old_result["provider_name"] != model.provider_name:
                raise ValueError("Dedup prior provider mismatch")
            if old_result["missing_infrastructure"] or not old_result["route_verified"]:
                pending.append(task)
        pending.sort(key=lambda task: task.id)
        smoke = [task for task in pending if task.id in cohort.changed_ids][:1]
        prefix = f"{OUTPUT_ROOT}attempts/{_model_path(model)}/"
        smoke_uri = prefix + "smoke.jsonl"
        smoke_rows = _read_completed(smoke_uri, smoke, model, panel_sha)
        if smoke_rows is None:
            _budget(model, len(smoke))
            _run(smoke, model, panel_sha, smoke_uri, 1)
            smoke_rows = _read_completed(smoke_uri, smoke, model, panel_sha)
            if smoke_rows is None:
                raise RuntimeError("Smoke output missing")
        _verify_routes(smoke_rows, model, smoke_uri)
        print(
            f"SMOKE model={model.model} status={smoke_rows[0].get('trial', {}).get('status')} uri={smoke_uri}",
            flush=True,
        )
        if args.smoke_only:
            continue
        rest = [task for task in pending if task.id not in {s.id for s in smoke}]
        for output, batch in _batches(rest, model, prefix):
            existing = _read_completed(output, batch, model, panel_sha)
            if existing is None:
                _budget(model, len(batch))
                workers = (
                    min(args.max_workers, 2) if model.model == "nvidia/nemotron-3-ultra-550b-a55b" else args.max_workers
                )
                _run(batch, model, panel_sha, output, workers)
                existing = _read_completed(output, batch, model, panel_sha)
                if existing is None:
                    raise RuntimeError(f"Model output missing: {model.model}")
            _verify_routes(existing, model, output)
        print(f"DONE model={model.model} new_pairs={len(pending)} reused_pairs={230 - len(pending)}", flush=True)
    print("V3_SMOKE_COMPLETE" if args.smoke_only else "V3_FIRST_WAVE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
