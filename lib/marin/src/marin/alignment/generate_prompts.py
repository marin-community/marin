# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1: Generate diverse evaluation prompts from a behavioral specification.

Implements the 3-stage Bloom pipeline:
  Stage 1: Understanding — analyze each behavior statement, generate variation axes
  Stage 2: Concretization — covering array → concrete scenarios via LLM
  Stage 3: Extraction — extract clean system_prompt + user_message from scenario prose

"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from rigging.filesystem import url_to_fs
from levanter.data.utils import batched
from zephyr import load_jsonl, write_jsonl_file

from marin.alignment.batched_vllm_serve import BatchedVllmServeSession, write_vllm_metrics_artifact
from marin.alignment.coverage import compute_coverage_stats, generate_covering_configs, make_tags
from marin.alignment.inference_config import InferenceConfig, VLLMConfig
from marin.alignment.live_progress import LiveProgressReporter, vllm_stage_metrics_provider
from marin.alignment.llm_client import llm_chat_single
from marin.alignment.prompts.concretize import make_concretize_prompt
from marin.alignment.prompts.extract import make_extraction_prompt
from marin.alignment.prompts.understanding import (
    STANDARD_DEMOGRAPHIC_AXES,
    make_behavior_understanding_prompt,
    make_understanding_system_prompt,
)
from marin.alignment.types import Statement

logger = logging.getLogger(__name__)
_CHECKPOINT_DIRNAME = "artifacts/checkpoints"
_STAGE_STATUS_FILENAME = "stage_status.json"
_STAGE1_CHECKPOINT_FILENAME = "understandings.jsonl.gz"
_STAGE1_ATTEMPTS_DIRNAME = "understanding_attempts"
_STAGE2_CHECKPOINT_FILENAME = "ideations.jsonl.gz"
_STAGE2_INCREMENTAL_DIRNAME = "ideation_by_statement"
_STAGE3_CHECKPOINT_DIRNAME = "extractions"
_CHECKPOINT_SHARD_PREFIX = "shard_"
_STAGE1_NAME = "understanding"
_STAGE2_NAME = "concretize"
_STAGE3_NAME = "extract"


@dataclass(frozen=True)
class PromptGenConfig:
    """Configuration for prompt generation from a specification."""

    spec_path: str
    output_path: str

    # LLM settings — InferenceConfig or string (string → OpenAI model ID)
    ideation_model: InferenceConfig | str = "gpt-4.1"
    extract_model: InferenceConfig | str = "gpt-4.1-mini"

    # Covering array
    covering_strength: int = 3
    covering_seed: int = 42

    # Runtime batching for local vLLM
    local_serve_batch_size: int = 8

    # Parallelism
    ideation_workers: int = 32
    concretize_workers: int = 32
    extract_workers: int = 128

    # LLM parameters
    understanding_max_tokens: int = 4000
    understanding_temperature: float = 1.0
    understanding_max_attempts: int = 5
    concretize_max_tokens: int = 1024
    concretize_temperature: float = 1.0
    extract_max_tokens: int = 1024
    concretize_max_attempts: int = 5
    extract_max_attempts: int = 5

    # Filtering
    statement_ids: list[str] | None = None


@dataclass(frozen=True)
class _ConcretizeConfig:
    index: int
    axis_config: dict[str, str]

    @property
    def config_id(self) -> str:
        return f"cfg_{self.index:03d}"


@dataclass(frozen=True)
class _ConcretizeWorkItem:
    statement_id: str
    understanding: dict[str, Any]
    axes: list[dict[str, Any]]
    request: _ConcretizeConfig


@dataclass(frozen=True)
class _ConcretizationPlan:
    statement_id: str
    axes: list[dict[str, Any]]
    configs: list[dict[str, str]]
    stats: dict[str, Any]


@dataclass(frozen=True)
class _ExtractionWorkItem:
    statement_id: str
    variation_index: int
    variation: dict[str, Any]


def _checkpoint_base_path(output_path: str) -> str:
    return f"{output_path}/{_CHECKPOINT_DIRNAME}"


def _stage_status_path(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE_STATUS_FILENAME}"


def _stage1_checkpoint_path(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE1_CHECKPOINT_FILENAME}"


def _stage1_attempts_dir(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE1_ATTEMPTS_DIRNAME}"


def _stage2_checkpoint_path(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE2_CHECKPOINT_FILENAME}"


def _stage2_incremental_checkpoint_dir(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE2_INCREMENTAL_DIRNAME}"


def _stage2_incremental_checkpoint_path(output_path: str, statement_id: str) -> str:
    return f"{_stage2_incremental_checkpoint_dir(output_path)}/{statement_id}.json"


def _stage3_checkpoint_dir(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE3_CHECKPOINT_DIRNAME}"


def _default_stage_status() -> dict[str, dict[str, int | bool]]:
    return {
        _STAGE1_NAME: {"complete": False, "num_statements": 0},
        _STAGE2_NAME: {"complete": False, "num_statements": 0},
        _STAGE3_NAME: {"complete": False, "completed_items": 0},
    }


def _write_json_artifact(path: str, payload: dict[str, Any]) -> None:
    fs, fs_path = url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _load_json_artifact(path: str) -> dict[str, Any] | None:
    fs, fs_path = url_to_fs(path)
    if not fs.exists(fs_path):
        return None
    with fs.open(fs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_stage_status(output_path: str) -> dict[str, dict[str, int | bool]]:
    status = _default_stage_status()
    persisted = _load_json_artifact(_stage_status_path(output_path))
    if persisted is None:
        return status
    for stage_name, defaults in status.items():
        persisted_stage = persisted.get(stage_name, {})
        status[stage_name] = {key: persisted_stage.get(key, value) for key, value in defaults.items()}
    return status


def _save_stage_status(output_path: str, stage_status: dict[str, dict[str, int | bool]]) -> None:
    _write_json_artifact(_stage_status_path(output_path), stage_status)


def _serialize_model_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(payload)
    if not isinstance(serializable.get("model"), str):
        serializable["model"] = str(serializable.get("model", ""))
    return serializable


def _save_understanding_checkpoint(
    output_path: str,
    statements: dict[str, Statement],
    understandings: dict[str, dict[str, Any]],
) -> None:
    _save_artifacts(output_path, statements, understandings, {})
    records = [
        {
            "statement_id": statement_id,
            "understanding": _serialize_model_artifact(understanding),
        }
        for statement_id, understanding in sorted(understandings.items())
    ]
    write_jsonl_file(records, _stage1_checkpoint_path(output_path))


def _load_understanding_checkpoint(output_path: str) -> dict[str, dict[str, Any]]:
    checkpoint_path = _stage1_checkpoint_path(output_path)
    fs, fs_path = url_to_fs(checkpoint_path)
    if not fs.exists(fs_path):
        return {}
    understandings: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(checkpoint_path):
        understandings[record["statement_id"]] = record["understanding"]
    return understandings


def _stage1_attempt_shard_count(output_path: str) -> int:
    checkpoint_dir = _stage1_attempts_dir(output_path)
    fs, fs_path = url_to_fs(checkpoint_dir)
    return len(fs.glob(f"{fs_path}/*.jsonl.gz"))


def _write_stage1_attempt_batch(output_path: str, shard_index: int, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    shard_path = f"{_stage1_attempts_dir(output_path)}/{_CHECKPOINT_SHARD_PREFIX}{shard_index:05d}.jsonl.gz"
    write_jsonl_file(records, shard_path)


def _append_stage1_attempt_records(output_path: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    _write_stage1_attempt_batch(output_path, _stage1_attempt_shard_count(output_path), records)


def _load_stage1_attempt_records(output_path: str) -> list[dict[str, Any]]:
    checkpoint_dir = _stage1_attempts_dir(output_path)
    fs, fs_path = url_to_fs(checkpoint_dir)
    if not fs.exists(fs_path):
        return []
    return load_sharded_jsonl_gz(checkpoint_dir)


def _save_ideation_checkpoint(
    output_path: str,
    statements: dict[str, Statement],
    understandings: dict[str, dict[str, Any]],
    ideations: dict[str, dict[str, Any]],
) -> None:
    _save_artifacts(output_path, statements, understandings, ideations)
    records = [
        {
            "statement_id": statement_id,
            "ideation": _serialize_model_artifact(ideation),
        }
        for statement_id, ideation in sorted(ideations.items())
    ]
    write_jsonl_file(records, _stage2_checkpoint_path(output_path))


def _load_ideation_checkpoint(output_path: str) -> dict[str, dict[str, Any]]:
    checkpoint_path = _stage2_checkpoint_path(output_path)
    fs, fs_path = url_to_fs(checkpoint_path)
    if not fs.exists(fs_path):
        return {}
    ideations: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(checkpoint_path):
        ideations[record["statement_id"]] = record["ideation"]
    return ideations


def _make_concretization_fingerprint(plan: _ConcretizationPlan) -> dict[str, Any]:
    plan_payload = {
        "axes": plan.axes,
        "configs": plan.configs,
    }
    plan_sha256 = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "statement_id": plan.statement_id,
        "num_axes": len(plan.axes),
        "num_configs": len(plan.configs),
        "plan_sha256": plan_sha256,
    }


def _save_statement_ideation(
    output_path: str,
    statement_id: str,
    ideation: dict[str, Any],
    fingerprint: dict[str, Any],
) -> None:
    _write_json_artifact(
        _stage2_incremental_checkpoint_path(output_path, statement_id),
        {
            "statement_id": statement_id,
            "ideation": _serialize_model_artifact(ideation),
            "fingerprint": fingerprint,
        },
    )


def _load_partial_ideation_checkpoint(
    output_path: str,
    plans: dict[str, _ConcretizationPlan],
) -> dict[str, dict[str, Any]]:
    checkpoint_dir = _stage2_incremental_checkpoint_dir(output_path)
    fs, fs_path = url_to_fs(checkpoint_dir)
    if not fs.exists(fs_path):
        return {}

    ideations: dict[str, dict[str, Any]] = {}
    for checkpoint_file in sorted(fs.glob(f"{fs_path}/*.json")):
        with fs.open(checkpoint_file, "r", encoding="utf-8") as f:
            record = json.load(f)
        statement_id = record["statement_id"]
        if statement_id not in plans:
            logger.warning("Discarding stale Stage 2 checkpoint for '%s': statement no longer present", statement_id)
            continue
        expected_fingerprint = _make_concretization_fingerprint(plans[statement_id])
        if record.get("fingerprint") != expected_fingerprint:
            logger.warning("Discarding stale Stage 2 checkpoint for '%s': fingerprint mismatch", statement_id)
            continue
        ideations[statement_id] = record["ideation"]
    return ideations


def _clear_stage3_checkpoint(output_path: str) -> None:
    checkpoint_dir = _stage3_checkpoint_dir(output_path)
    fs, fs_path = url_to_fs(checkpoint_dir)
    for shard in fs.glob(f"{fs_path}/*.jsonl.gz"):
        fs.rm(shard)
    if fs.exists(fs_path):
        fs.rm(fs_path, recursive=True)


def _stage3_checkpoint_shard_count(output_path: str) -> int:
    checkpoint_dir = _stage3_checkpoint_dir(output_path)
    fs, fs_path = url_to_fs(checkpoint_dir)
    return len(fs.glob(f"{fs_path}/*.jsonl.gz"))


def _write_stage3_checkpoint_batch(output_path: str, shard_index: int, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    shard_path = f"{_stage3_checkpoint_dir(output_path)}/{_CHECKPOINT_SHARD_PREFIX}{shard_index:05d}.jsonl.gz"
    write_jsonl_file(records, shard_path)


def _load_stage3_checkpoint(output_path: str) -> dict[tuple[str, int], dict[str, str]]:
    extracted_by_item: dict[tuple[str, int], dict[str, str]] = {}
    for record in load_sharded_jsonl_gz(_stage3_checkpoint_dir(output_path)):
        extracted_by_item[(record["statement_id"], int(record["variation_index"]))] = record["extraction"]
    return extracted_by_item


def _prompt_shards_exist(output_path: str) -> bool:
    fs, fs_path = url_to_fs(output_path)
    return bool(fs.glob(f"{fs_path}/shard_*.jsonl.gz"))


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def load_spec(spec_path: str) -> dict[str, Statement]:
    """Load behavioral statements from a JSONL file. Supports both local and GCS paths."""
    return {record["id"]: Statement.from_dict(record) for record in load_jsonl(spec_path)}


# ---------------------------------------------------------------------------
# Stage 1: Understanding
# ---------------------------------------------------------------------------


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_variation_axes(text: str) -> list[dict[str, Any]]:
    match = re.search(r"<variation_axes>(.*?)</variation_axes>", text, re.DOTALL)
    if not match:
        raise ValueError("Stage1 response missing <variation_axes> block")
    payload = match.group(1).strip()
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("Stage1 variation_axes JSON must be a list")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Stage1 variation_axes[{i}] must be an object")
        if not item.get("axis"):
            raise ValueError(f"Stage1 variation_axes[{i}] missing required 'axis'")
        spectrum = item.get("spectrum")
        if not isinstance(spectrum, list) or len(spectrum) < 2:
            raise ValueError(f"Stage1 variation_axes[{i}] has invalid 'spectrum' (need >=2 values)")
        out.append(item)
    return out


def _build_chat_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _single_completion_text(outputs: list[str]) -> str:
    return outputs[0] if outputs else ""


def _make_stage1_attempt_record(statement_id: str, attempt: int, raw_response: str) -> dict[str, Any]:
    return {
        "statement_id": statement_id,
        "attempt": attempt,
        "raw_response": raw_response,
    }


def _parse_understanding_response(statement: Statement, content: str, model: InferenceConfig | str) -> dict[str, Any]:
    understanding = _extract_tag(content, "behavior_understanding")
    scientific_motivation = _extract_tag(content, "scientific_motivation")
    behavior_specific_axes = _parse_variation_axes(content)
    variation_axes = behavior_specific_axes + [dict(ax) for ax in STANDARD_DEMOGRAPHIC_AXES]

    return {
        "behavior_name": statement.id,
        "understanding": understanding,
        "scientific_motivation": scientific_motivation,
        "variation_axes": variation_axes,
        "model": model,
    }


def _recover_understandings_from_attempts(
    output_path: str,
    statements: dict[str, Statement],
    model: InferenceConfig | str,
) -> dict[str, dict[str, Any]]:
    recovered: dict[str, dict[str, Any]] = {}
    attempt_records = _load_stage1_attempt_records(output_path)
    if not attempt_records:
        return recovered

    for record in reversed(attempt_records):
        statement_id = record.get("statement_id")
        if not isinstance(statement_id, str) or statement_id not in statements or statement_id in recovered:
            continue
        raw_response = str(record.get("raw_response", ""))
        try:
            recovered[statement_id] = _parse_understanding_response(
                statements[statement_id],
                raw_response,
                model,
            )
        except Exception:
            continue

    if recovered:
        logger.info(
            "Recovered %d Stage 1 understanding(s) from raw attempt checkpoints; %d statement(s) still pending",
            len(recovered),
            len(statements) - len(recovered),
        )
    return recovered


def _build_understanding_messages(statement: Statement) -> list[dict[str, str]]:
    return _build_chat_messages(
        system_prompt=make_understanding_system_prompt(),
        user_prompt=make_behavior_understanding_prompt(statement.id, statement.text),
    )


def _understanding_request(
    statement: Statement,
    config: PromptGenConfig,
) -> str:
    """Request one Stage 1 understanding completion."""
    response = llm_chat_single(
        config=config.ideation_model,
        messages=[{"role": "user", "content": make_behavior_understanding_prompt(statement.id, statement.text)}],
        system_prompt=make_understanding_system_prompt(),
        max_tokens=config.understanding_max_tokens,
        temperature=config.understanding_temperature,
    )
    return response.content


def _run_understanding_local(
    statements: dict[str, Statement],
    config: PromptGenConfig,
    session: BatchedVllmServeSession,
) -> dict[str, dict[str, Any]]:
    understandings: dict[str, dict[str, Any]] = {}
    pending_statements = list(statements.items())
    failures: dict[str, str] = {}
    total_statements = len(statements)
    reporter = LiveProgressReporter(
        stage_name="Stage 1",
        total_items=total_statements,
        batch_size=config.local_serve_batch_size,
        metrics_provider=vllm_stage_metrics_provider(session, stage_name="understanding"),
    )

    for attempt in range(1, config.understanding_max_attempts + 1):
        if not pending_statements:
            break
        next_pending: list[tuple[str, Statement]] = []
        for statement_batch in batched(pending_statements, config.local_serve_batch_size):
            outputs = session.generate_from_messages(
                [_build_understanding_messages(statement) for _sid, statement in statement_batch],
                stage_name="understanding",
                temperature=config.understanding_temperature,
                max_tokens=config.understanding_max_tokens,
                n=1,
            )
            raw_attempt_records = [
                _make_stage1_attempt_record(sid, attempt, _single_completion_text(output))
                for (sid, _statement), output in zip(statement_batch, outputs, strict=True)
            ]
            _append_stage1_attempt_records(config.output_path, raw_attempt_records)
            for (sid, statement), output in zip(statement_batch, outputs, strict=True):
                raw_response = _single_completion_text(output)
                try:
                    understandings[sid] = _parse_understanding_response(
                        statement,
                        raw_response,
                        config.ideation_model,
                    )
                    failures.pop(sid, None)
                except Exception as exc:
                    failures[sid] = str(exc)
                    next_pending.append((sid, statement))
                    logger.warning("Stage1 attempt %d failed for '%s': %s", attempt, sid, exc)
            reporter.maybe_log(len(understandings), details=[f"attempt {attempt}"])
        pending_statements = next_pending

    if pending_statements:
        detail = "; ".join(f"{sid}: {failures[sid]}" for sid, _statement in pending_statements)
        raise RuntimeError(f"Stage 1 failed for {len(failures)} statement(s): {detail}")

    return understandings


# ---------------------------------------------------------------------------
# Stage 2: Concretization
# ---------------------------------------------------------------------------


def _parse_concretize_response(content: str) -> dict[str, str]:
    scenario_match = re.search(r"<scenario>(.*?)</scenario>", content, re.DOTALL)
    if not scenario_match:
        return {}
    rubric_match = re.search(r"<rubric>(.*?)</rubric>", content, re.DOTALL)
    return {
        "description": scenario_match.group(1).strip(),
        "rubric": rubric_match.group(1).strip() if rubric_match else "",
    }


def _build_concretize_messages(
    statement_id: str,
    understanding_data: dict[str, Any],
    axes: list[dict[str, Any]],
    request: _ConcretizeConfig,
) -> list[dict[str, str]]:
    system_prompt, user_prompt = make_concretize_prompt(
        behavior_name=statement_id,
        behavior_understanding=understanding_data.get("understanding", ""),
        scientific_motivation=understanding_data.get("scientific_motivation", ""),
        transcript_analyses=understanding_data.get("transcript_analyses", []),
        config_id=request.config_id,
        axis_config=request.axis_config,
        axes_metadata=axes,
    )
    return _build_chat_messages(system_prompt=system_prompt, user_prompt=user_prompt)


def _parse_concretize_response_with_diagnostics(
    statement_id: str,
    request: _ConcretizeConfig,
    content: str,
    attempt: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    parsed_scenario = _parse_concretize_response(content)
    returned_config_ids = [request.config_id] if parsed_scenario else []
    missing_config_ids = [] if parsed_scenario else [request.config_id]
    missing_rubric_config_ids = [
        config_id for config_id in returned_config_ids if not parsed_scenario.get("rubric", "").strip()
    ]

    if missing_config_ids:
        logger.warning(
            "Stage2 attempt %d for '%s' missing scenario for %s",
            attempt,
            statement_id,
            request.config_id,
        )

    diagnostic = {
        "attempt": attempt,
        "requested_config_ids": [request.config_id],
        "requested_configs": [{"config_id": request.config_id, "axis_config": request.axis_config}],
        "returned_config_ids": returned_config_ids,
        "missing_config_ids": missing_config_ids,
        "missing_rubric_config_ids": missing_rubric_config_ids,
        "raw_response": content,
    }
    parsed = {request.config_id: parsed_scenario} if parsed_scenario else {}
    return parsed, diagnostic


def _concretize_request(
    statement_id: str,
    understanding_data: dict[str, Any],
    axes: list[dict[str, Any]],
    request: _ConcretizeConfig,
    model: InferenceConfig | str,
    temperature: float,
    max_tokens: int = 1024,
) -> str:
    """Concretize one axis config into one scenario."""
    system_prompt, user_prompt = make_concretize_prompt(
        behavior_name=statement_id,
        behavior_understanding=understanding_data.get("understanding", ""),
        scientific_motivation=understanding_data.get("scientific_motivation", ""),
        transcript_analyses=understanding_data.get("transcript_analyses", []),
        config_id=request.config_id,
        axis_config=request.axis_config,
        axes_metadata=axes,
    )
    response = llm_chat_single(
        config=model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.content


def _run_concretize_round_api(
    statement_id: str,
    understanding: dict[str, Any],
    axes: list[dict[str, Any]],
    requests: list[_ConcretizeConfig],
    config: PromptGenConfig,
    attempt: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[_ConcretizeConfig], list[str]]:
    parsed_by_config_id: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    retry_requests: list[_ConcretizeConfig] = []
    failures: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.concretize_workers) as pool:
        future_map = {
            pool.submit(
                _concretize_request,
                statement_id,
                understanding,
                axes,
                request,
                config.ideation_model,
                config.concretize_temperature,
                config.concretize_max_tokens,
            ): request
            for request in requests
        }
        for future in concurrent.futures.as_completed(future_map):
            request = future_map[future]
            try:
                content = future.result()
                parsed, diagnostic = _parse_concretize_response_with_diagnostics(
                    statement_id,
                    request,
                    content,
                    attempt,
                )
                parsed_by_config_id.update(parsed)
                diagnostics.append(diagnostic)
                if diagnostic["missing_config_ids"]:
                    retry_requests.append(request)
            except Exception as exc:
                failures.append(f"attempt {attempt} config [{request.config_id}]: {exc}")

    return parsed_by_config_id, diagnostics, retry_requests, failures


def _run_concretize_round_local_global(
    work_items: list[_ConcretizeWorkItem],
    config: PromptGenConfig,
    session: BatchedVllmServeSession,
    attempt: int,
    *,
    parsed_by_statement: dict[str, dict[str, dict[str, Any]]],
    diagnostics_by_statement: dict[str, list[dict[str, Any]]],
    expected_items_per_statement: dict[str, int],
    checkpointed_statement_ids: set[str],
    statement_completed_callback: Callable[[str, dict[str, dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[_ConcretizeWorkItem], list[str]]:
    retry_items: list[_ConcretizeWorkItem] = []
    failures: list[str] = []
    completed_in_round = 0

    for work_item_group in batched(work_items, config.local_serve_batch_size):
        try:
            outputs = session.generate_from_messages(
                [
                    _build_concretize_messages(item.statement_id, item.understanding, item.axes, item.request)
                    for item in work_item_group
                ],
                stage_name="concretize",
                temperature=config.concretize_temperature,
                max_tokens=config.concretize_max_tokens,
                n=1,
            )
        except (ValueError, requests.HTTPError) as exc:
            # GPT-OSS chat path raises ValueError on finish_reason='length';
            # treat every item in the batch as failed so retry logic can fire.
            for item in work_item_group:
                failures.append(
                    f"attempt {attempt} statement '{item.statement_id}' " f"config [{item.request.config_id}]: {exc}"
                )
            continue
        batch_completed = 0
        for work_item, output in zip(work_item_group, outputs, strict=True):
            try:
                parsed, diagnostic = _parse_concretize_response_with_diagnostics(
                    work_item.statement_id,
                    work_item.request,
                    _single_completion_text(output),
                    attempt,
                )
            except Exception as exc:
                failures.append(
                    f"attempt {attempt} statement '{work_item.statement_id}' "
                    f"config [{work_item.request.config_id}]: {exc}"
                )
                continue

            statement_id = work_item.statement_id
            parsed_by_statement.setdefault(statement_id, {}).update(parsed)
            statement_diagnostics = diagnostics_by_statement.setdefault(statement_id, [])
            statement_diagnostics.append(diagnostic)
            if diagnostic["missing_config_ids"]:
                retry_items.append(work_item)
            else:
                batch_completed += 1

            if (
                statement_completed_callback is not None
                and statement_id not in checkpointed_statement_ids
                and len(parsed_by_statement.get(statement_id, {}))
                >= expected_items_per_statement.get(statement_id, float("inf"))
            ):
                statement_completed_callback(
                    statement_id,
                    dict(parsed_by_statement[statement_id]),
                    list(statement_diagnostics),
                )
                checkpointed_statement_ids.add(statement_id)
        completed_in_round += batch_completed
        if progress_callback is not None:
            progress_callback(completed_in_round)

    return retry_items, failures


def _build_concretization_result(
    statement_id: str,
    config: PromptGenConfig,
    axes: list[dict[str, Any]],
    configs: list[dict[str, str]],
    stats: dict[str, Any],
    parsed_by_config_id: dict[str, dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    variations: list[dict[str, Any]] = []
    for index, axis_config in enumerate(configs):
        config_id = f"cfg_{index:03d}"
        scenario = parsed_by_config_id.get(config_id, {})
        variations.append(
            {
                "description": scenario.get("description", ""),
                "rubric": scenario.get("rubric", ""),
                "config_id": config_id,
                "axis_config": axis_config,
                "tags": make_tags(axis_config, axes),
            }
        )

    return {
        "behavior_name": statement_id,
        "model": config.ideation_model,
        "covering_strength": config.covering_strength,
        "num_configs": len(configs),
        "coverage_stats": stats,
        "concretization_attempts": diagnostics,
        "variations": variations,
    }


def _prepare_concretization_plans(
    understandings: dict[str, dict[str, Any]],
    config: PromptGenConfig,
) -> dict[str, _ConcretizationPlan]:
    plans: dict[str, _ConcretizationPlan] = {}
    for statement_id, understanding in understandings.items():
        axes = understanding["variation_axes"]
        configs = generate_covering_configs(axes, t=config.covering_strength, seed=config.covering_seed)
        stats = compute_coverage_stats(configs, axes, t=config.covering_strength)
        plans[statement_id] = _ConcretizationPlan(
            statement_id=statement_id,
            axes=axes,
            configs=configs,
            stats=stats,
        )
    return plans


def _run_concretization(
    statement_id: str,
    understanding: dict[str, Any],
    config: PromptGenConfig,
) -> dict[str, Any]:
    """Run Stage 2 for a single statement: covering array → concretize via LLM."""
    axes = understanding["variation_axes"]
    configs = generate_covering_configs(axes, t=config.covering_strength, seed=config.covering_seed)
    stats = compute_coverage_stats(configs, axes, t=config.covering_strength)

    requests = [_ConcretizeConfig(index=index, axis_config=axis_config) for index, axis_config in enumerate(configs)]

    all_scenarios: dict[str, dict[str, Any]] = {}
    concretization_diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    pending_requests = requests

    for attempt in range(1, config.concretize_max_attempts + 1):
        if not pending_requests:
            break
        parsed, diagnostics, retry_requests, round_failures = _run_concretize_round_api(
            statement_id,
            understanding,
            axes,
            pending_requests,
            config,
            attempt,
        )
        all_scenarios.update(parsed)
        concretization_diagnostics.extend(diagnostics)
        failures.extend(round_failures)
        pending_requests = retry_requests

    if failures:
        raise RuntimeError(f"Stage2 failed for '{statement_id}': {'; '.join(failures)}")
    if pending_requests:
        missing_config_ids = [request.config_id for request in pending_requests]
        raise RuntimeError(
            f"Stage2 failed for '{statement_id}': missing scenarios after {config.concretize_max_attempts} "
            f"attempt(s) for {', '.join(missing_config_ids)}"
        )

    return _build_concretization_result(
        statement_id,
        config,
        axes,
        configs,
        stats,
        all_scenarios,
        concretization_diagnostics,
    )


def _run_concretization_stage_local(
    understandings: dict[str, dict[str, Any]],
    config: PromptGenConfig,
    session: BatchedVllmServeSession,
    *,
    plans: dict[str, _ConcretizationPlan] | None = None,
    completed_ideations: dict[str, dict[str, Any]] | None = None,
    statement_checkpoint_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    plans = plans or _prepare_concretization_plans(understandings, config)
    completed_ideations = completed_ideations or {}
    completed_statement_ids = set(completed_ideations)
    pending_items = [
        _ConcretizeWorkItem(
            statement_id=statement_id,
            understanding=understandings[statement_id],
            axes=plan.axes,
            request=_ConcretizeConfig(index=index, axis_config=axis_config),
        )
        for statement_id, plan in plans.items()
        if statement_id not in completed_statement_ids
        for index, axis_config in enumerate(plan.configs)
    ]
    total_items = len(pending_items)
    if completed_statement_ids:
        logger.info(
            "Stage 2 local work queue: %d concretize items across %d pending statements "
            "(skipping %d checkpointed statement(s))",
            total_items,
            len(plans) - len(completed_statement_ids),
            len(completed_statement_ids),
        )
    else:
        logger.info("Stage 2 local work queue: %d concretize items across %d statements", total_items, len(plans))

    parsed_by_statement: dict[str, dict[str, dict[str, Any]]] = {
        statement_id: {} for statement_id in understandings if statement_id not in completed_statement_ids
    }
    diagnostics_by_statement: dict[str, list[dict[str, Any]]] = {
        statement_id: [] for statement_id in understandings if statement_id not in completed_statement_ids
    }
    expected_items_per_statement = {
        statement_id: len(plan.configs)
        for statement_id, plan in plans.items()
        if statement_id not in completed_statement_ids
    }
    failures: list[str] = []
    reporter = LiveProgressReporter(
        stage_name="Stage 2",
        total_items=total_items,
        batch_size=config.local_serve_batch_size,
        metrics_provider=vllm_stage_metrics_provider(session, stage_name="concretize"),
    )

    def on_statement_completed(
        statement_id: str,
        parsed_scenarios: dict[str, dict[str, Any]],
        diagnostics: list[dict[str, Any]],
    ) -> None:
        ideation = _build_concretization_result(
            statement_id,
            config,
            plans[statement_id].axes,
            plans[statement_id].configs,
            plans[statement_id].stats,
            parsed_scenarios,
            diagnostics,
        )
        if statement_checkpoint_callback is not None:
            statement_checkpoint_callback(statement_id, ideation)

    for attempt in range(1, config.concretize_max_attempts + 1):
        if not pending_items:
            break
        completed_before_attempt = sum(len(scenarios) for scenarios in parsed_by_statement.values())

        def log_progress(
            completed_in_round: int,
            *,
            completed_before_attempt: int = completed_before_attempt,
            attempt_label: int = attempt,
        ) -> None:
            reporter.maybe_log(
                completed_before_attempt + completed_in_round,
                details=[f"attempt {attempt_label}"],
            )

        retry_items, round_failures = _run_concretize_round_local_global(
            pending_items,
            config,
            session,
            attempt,
            parsed_by_statement=parsed_by_statement,
            diagnostics_by_statement=diagnostics_by_statement,
            expected_items_per_statement=expected_items_per_statement,
            checkpointed_statement_ids=completed_statement_ids,
            statement_completed_callback=on_statement_completed,
            progress_callback=log_progress,
        )
        failures.extend(round_failures)
        pending_items = retry_items
        log_progress(sum(len(scenarios) for scenarios in parsed_by_statement.values()) - completed_before_attempt)
        if pending_items:
            reporter.maybe_log(
                sum(len(scenarios) for scenarios in parsed_by_statement.values()),
                details=[f"attempt {attempt}", f"retries pending={len(pending_items)}"],
                force=True,
            )
            logger.warning(
                "Stage2 attempt %d left %d concretize item(s) pending retry",
                attempt,
                len(pending_items),
            )

    if failures:
        raise RuntimeError(f"Stage 2 failed: {'; '.join(failures)}")
    if pending_items:
        missing_detail = ", ".join(f"{item.statement_id}:{item.request.config_id}" for item in pending_items)
        raise RuntimeError(
            f"Stage 2 failed: missing scenarios after {config.concretize_max_attempts} attempt(s) for {missing_detail}"
        )

    result = dict(completed_ideations)
    result.update(
        {
            statement_id: _build_concretization_result(
                statement_id,
                config,
                plan.axes,
                plan.configs,
                plan.stats,
                parsed_by_statement.get(statement_id, {}),
                diagnostics_by_statement.get(statement_id, []),
            )
            for statement_id, plan in plans.items()
            if statement_id not in completed_ideations
        }
    )
    return result


# ---------------------------------------------------------------------------
# Stage 3: Extraction
# ---------------------------------------------------------------------------


def _parse_extraction_response(
    content: str,
    include_system_prompt: bool = True,
) -> dict[str, str]:
    if include_system_prompt:
        sp_match = re.search(r"<system_prompt>(.*?)</system_prompt>", content, re.DOTALL)
        if not sp_match:
            raise RuntimeError("Stage3: missing <system_prompt> block in extraction response")
        system_prompt = sp_match.group(1).strip()
    else:
        system_prompt = ""
    um_match = re.search(r"<user_message>(.*?)</user_message>", content, re.DOTALL)
    if not um_match:
        raise RuntimeError("Stage3: missing <user_message> block in extraction response")
    user_message = um_match.group(1).strip()
    return {"system_prompt": system_prompt, "user_message": user_message}


def _build_extraction_messages(scenario: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt, user_prompt = make_extraction_prompt(scenario, include_system_prompt=True)
    return _build_chat_messages(system_prompt=system_prompt, user_prompt=user_prompt)


def _extract_request(
    scenario: dict[str, Any],
    model: InferenceConfig | str,
    max_tokens: int = 1024,
) -> dict[str, str]:
    """Extract one clean prompt pair from one scenario."""
    system_prompt, user_prompt = make_extraction_prompt(scenario, include_system_prompt=True)
    response = llm_chat_single(
        config=model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return _parse_extraction_response(response.content, include_system_prompt=True)


def _build_extraction_work_items(ideations: dict[str, dict[str, Any]]) -> list[_ExtractionWorkItem]:
    return [
        _ExtractionWorkItem(statement_id=statement_id, variation_index=index, variation=variation)
        for statement_id, ideation in ideations.items()
        for index, variation in enumerate(ideation.get("variations", []))
        if str(variation.get("description", "")).strip()
    ]


def _make_extraction_checkpoint_record(item: _ExtractionWorkItem, extraction: dict[str, str]) -> dict[str, Any]:
    return {
        "statement_id": item.statement_id,
        "variation_index": item.variation_index,
        "extraction": extraction,
    }


def _build_prompts_from_extractions(
    work_items: list[_ExtractionWorkItem],
    extracted_by_item: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for item in work_items:
        extraction = extracted_by_item.get((item.statement_id, item.variation_index), {})
        system_prompt = extraction.get("system_prompt", "")
        user_message = extraction.get("user_message", "")
        if not user_message:
            continue
        prompts.append(
            {
                "behavior_id": item.statement_id,
                "system_prompt": system_prompt,
                "user_message": user_message,
                "rubric": item.variation.get("rubric", ""),
                "config_id": item.variation.get("config_id", ""),
                "axis_config": item.variation.get("axis_config", {}),
                "tags": item.variation.get("tags", []),
            }
        )
    return prompts


def _run_extraction_stage_api(
    ideations: dict[str, dict[str, Any]],
    config: PromptGenConfig,
    existing_extractions: dict[tuple[str, int], dict[str, str]] | None = None,
    checkpoint_callback: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> list[dict[str, Any]]:
    work_items = _build_extraction_work_items(ideations)
    extracted_by_item = dict(existing_extractions or {})
    pending_items = [item for item in work_items if (item.statement_id, item.variation_index) not in extracted_by_item]
    total_items = len(work_items)
    logger.info(
        "Stage 3 API work queue: %d pending extraction items (%d already checkpointed) across %d statements",
        len(pending_items),
        len(extracted_by_item),
        len(ideations),
    )
    flush_size = max(1, config.local_serve_batch_size)
    failures_by_item: dict[tuple[str, int], str] = {}
    reporter = LiveProgressReporter(
        stage_name="Stage 3",
        total_items=total_items,
        batch_size=flush_size,
        initial_completed_items=len(extracted_by_item),
    )

    for attempt in range(1, config.extract_max_attempts + 1):
        if not pending_items:
            break
        staged_records: list[dict[str, Any]] = []
        next_pending: list[_ExtractionWorkItem] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, config.extract_workers)) as pool:
            future_map = {
                pool.submit(_extract_request, item.variation, config.extract_model, config.extract_max_tokens): item
                for item in pending_items
            }
            for future in concurrent.futures.as_completed(future_map):
                item = future_map[future]
                key = (item.statement_id, item.variation_index)
                try:
                    extraction = future.result()
                    extracted_by_item[key] = extraction
                    failures_by_item.pop(key, None)
                    staged_records.append(_make_extraction_checkpoint_record(item, extraction))
                except Exception as exc:
                    failures_by_item[key] = f"statement '{item.statement_id}' variation {item.variation_index}: {exc}"
                    next_pending.append(item)
                if checkpoint_callback is not None and len(staged_records) >= flush_size:
                    checkpoint_callback(staged_records, len(extracted_by_item))
                    staged_records = []
                reporter.maybe_log(
                    len(extracted_by_item),
                    details=[f"attempt {attempt}", f"retries pending={len(next_pending)}"],
                )
        if checkpoint_callback is not None and staged_records:
            checkpoint_callback(staged_records, len(extracted_by_item))
        if next_pending:
            reporter.maybe_log(
                len(extracted_by_item),
                details=[f"attempt {attempt}", f"retries pending={len(next_pending)}"],
                force=True,
            )
            logger.warning(
                "Stage3 attempt %d left %d extraction item(s) pending retry",
                attempt,
                len(next_pending),
            )
        pending_items = next_pending

    if pending_items:
        failure_detail = "; ".join(failures_by_item[(item.statement_id, item.variation_index)] for item in pending_items)
        logger.warning(
            "Stage 3 skipping %d failed extraction(s) after %d attempts: %s",
            len(pending_items),
            config.extract_max_attempts,
            failure_detail,
        )

    return _build_prompts_from_extractions(work_items, extracted_by_item)


def _run_extraction_stage_local(
    ideations: dict[str, dict[str, Any]],
    config: PromptGenConfig,
    session: BatchedVllmServeSession,
    existing_extractions: dict[tuple[str, int], dict[str, str]] | None = None,
    checkpoint_callback: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> list[dict[str, Any]]:
    work_items = _build_extraction_work_items(ideations)
    extracted_by_item = dict(existing_extractions or {})
    pending_items = [item for item in work_items if (item.statement_id, item.variation_index) not in extracted_by_item]
    total_items = len(work_items)
    logger.info(
        "Stage 3 local work queue: %d pending extraction items (%d already checkpointed) across %d statements",
        len(pending_items),
        len(extracted_by_item),
        len(ideations),
    )
    failures_by_item: dict[tuple[str, int], str] = {}
    reporter = LiveProgressReporter(
        stage_name="Stage 3",
        total_items=total_items,
        batch_size=config.local_serve_batch_size,
        metrics_provider=vllm_stage_metrics_provider(session, stage_name="extract"),
        initial_completed_items=len(extracted_by_item),
    )

    for attempt in range(1, config.extract_max_attempts + 1):
        if not pending_items:
            break
        next_pending: list[_ExtractionWorkItem] = []
        for item_batch in batched(pending_items, config.local_serve_batch_size):
            outputs = session.generate_from_messages(
                [_build_extraction_messages(item.variation) for item in item_batch],
                stage_name="extract",
                temperature=0.0,
                max_tokens=config.extract_max_tokens,
                n=1,
            )
            batch_records: list[dict[str, Any]] = []
            for item, output in zip(item_batch, outputs, strict=True):
                key = (item.statement_id, item.variation_index)
                try:
                    extraction = _parse_extraction_response(
                        _single_completion_text(output),
                        include_system_prompt=True,
                    )
                    extracted_by_item[key] = extraction
                    failures_by_item.pop(key, None)
                    batch_records.append(_make_extraction_checkpoint_record(item, extraction))
                except Exception as exc:
                    failures_by_item[key] = f"statement '{item.statement_id}' variation {item.variation_index}: {exc}"
                    next_pending.append(item)
            if checkpoint_callback is not None and batch_records:
                checkpoint_callback(batch_records, len(extracted_by_item))
            reporter.maybe_log(
                len(extracted_by_item),
                details=[f"attempt {attempt}", f"retries pending={len(next_pending)}"],
            )
        if next_pending:
            reporter.maybe_log(
                len(extracted_by_item),
                details=[f"attempt {attempt}", f"retries pending={len(next_pending)}"],
                force=True,
            )
            logger.warning(
                "Stage3 attempt %d left %d extraction item(s) pending retry",
                attempt,
                len(next_pending),
            )
        pending_items = next_pending

    if pending_items:
        failure_detail = "; ".join(failures_by_item[(item.statement_id, item.variation_index)] for item in pending_items)
        logger.warning(
            "Stage 3 skipping %d failed extraction(s) after %d attempts: %s",
            len(pending_items),
            config.extract_max_attempts,
            failure_detail,
        )
        if checkpoint_callback is not None:
            _write_extraction_failures(config.output_path, pending_items, failures_by_item)

    return _build_prompts_from_extractions(work_items, extracted_by_item)


def _write_extraction_failures(
    output_path: str,
    failed_items: list[_ExtractionWorkItem],
    failures_by_item: dict[tuple[str, int], str],
) -> None:
    """Write failed extraction items to a JSONL file for later retry."""
    failures_path = f"{output_path}/artifacts/extraction_failures.jsonl"
    fs, fs_path = url_to_fs(failures_path)
    parent = fs_path.rsplit("/", 1)[0]
    fs.makedirs(parent, exist_ok=True)
    records = [
        {
            "statement_id": item.statement_id,
            "variation_index": item.variation_index,
            "error": failures_by_item.get((item.statement_id, item.variation_index), "unknown"),
        }
        for item in failed_items
    ]
    with fs.open(fs_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote %d extraction failure(s) to %s", len(records), failures_path)


# ---------------------------------------------------------------------------
# Full pipeline: spec → prompts
# ---------------------------------------------------------------------------


def generate_prompts_from_spec(config: PromptGenConfig) -> None:
    """Run the full 3-stage prompt generation pipeline.

    Reads a specification JSONL, runs understanding → concretization → extraction
    for each behavior statement, and writes the results as sharded JSONL.GZ files.
    """
    statements = _load_filtered_statements(config)
    expected_statement_ids = set(statements)
    stage_status = _load_stage_status(config.output_path)

    understandings: dict[str, dict[str, Any]] | None = None
    ideations: dict[str, dict[str, Any]] | None = None
    existing_extractions: dict[tuple[str, int], dict[str, str]] = {}
    partial_ideations: dict[str, dict[str, Any]] = {}
    stage2_plans: dict[str, _ConcretizationPlan] | None = None

    def get_stage2_plans() -> dict[str, _ConcretizationPlan]:
        nonlocal stage2_plans
        if stage2_plans is None:
            assert understandings is not None
            stage2_plans = _prepare_concretization_plans(understandings, config)
        return stage2_plans

    if stage_status[_STAGE1_NAME]["complete"]:
        checkpointed_understandings = _load_understanding_checkpoint(config.output_path)
        if set(checkpointed_understandings) == expected_statement_ids:
            understandings = checkpointed_understandings
            logger.info("Loaded Stage 1 checkpoint for %d statements", len(understandings))
        else:
            logger.warning(
                "Ignoring incomplete Stage 1 checkpoint: expected %d statements, found %d",
                len(expected_statement_ids),
                len(checkpointed_understandings),
            )
    elif expected_statement_ids:
        recovered_understandings = _recover_understandings_from_attempts(
            config.output_path,
            statements,
            config.ideation_model,
        )
        if recovered_understandings:
            understandings = recovered_understandings

    if understandings is not None and stage_status[_STAGE2_NAME]["complete"]:
        checkpointed_ideations = _load_ideation_checkpoint(config.output_path)
        if set(checkpointed_ideations) == expected_statement_ids:
            ideations = checkpointed_ideations
            logger.info("Loaded Stage 2 checkpoint for %d statements", len(ideations))
            existing_extractions = _load_stage3_checkpoint(config.output_path)
            if existing_extractions:
                logger.info("Loaded %d checkpointed Stage 3 extraction items", len(existing_extractions))
        else:
            logger.warning(
                "Ignoring incomplete Stage 2 checkpoint: expected %d statements, found %d",
                len(expected_statement_ids),
                len(checkpointed_ideations),
            )

    if (
        understandings is not None
        and len(understandings) == len(expected_statement_ids)
        and ideations is None
        and isinstance(config.ideation_model, VLLMConfig)
    ):
        partial_ideations = _load_partial_ideation_checkpoint(config.output_path, get_stage2_plans())
        if partial_ideations:
            logger.info(
                "Loaded partial Stage 2 checkpoint for %d/%d statements",
                len(partial_ideations),
                len(expected_statement_ids),
            )
            if set(partial_ideations) == expected_statement_ids:
                ideations = partial_ideations
                _save_ideation_checkpoint(config.output_path, statements, understandings, ideations)
                stage_status[_STAGE2_NAME] = {"complete": True, "num_statements": len(ideations)}
                _save_stage_status(config.output_path, stage_status)
                existing_extractions = _load_stage3_checkpoint(config.output_path)
                if existing_extractions:
                    logger.info("Loaded %d checkpointed Stage 3 extraction items", len(existing_extractions))

    if ideations is not None and stage_status[_STAGE3_NAME]["complete"]:
        work_items = _build_extraction_work_items(ideations)
        if len(existing_extractions) >= len(work_items):
            all_prompts = _build_prompts_from_extractions(work_items, existing_extractions)
            if not _prompt_shards_exist(config.output_path):
                write_sharded_jsonl_gz(all_prompts, config.output_path, shard_size=5000)
            assert understandings is not None
            _save_artifacts(config.output_path, statements, understandings, ideations)
            logger.info("All prompt-generation stages already completed via checkpoints; skipping execution")
            return
        logger.warning(
            "Ignoring incomplete Stage 3 completion marker: expected %d extraction items, found %d",
            len(work_items),
            len(existing_extractions),
        )

    def checkpoint_stage3_batch(records: list[dict[str, Any]], completed_items: int) -> None:
        nonlocal stage_status
        shard_index = _stage3_checkpoint_shard_count(config.output_path)
        _write_stage3_checkpoint_batch(config.output_path, shard_index, records)
        stage_status[_STAGE3_NAME] = {"complete": False, "completed_items": completed_items}
        _save_stage_status(config.output_path, stage_status)

    metrics_sessions: list[tuple[str, dict[str, object]]] = []
    all_prompts: list[dict[str, Any]] | None = None
    same_local_model = isinstance(config.ideation_model, VLLMConfig) and config.extract_model == config.ideation_model
    pending_stage1_statements = (
        statements
        if understandings is None
        else {sid: statement for sid, statement in statements.items() if sid not in understandings}
    )
    if understandings is not None and not pending_stage1_statements and not stage_status[_STAGE1_NAME]["complete"]:
        _finalize_stage1_success(config.output_path, statements, understandings, stage_status)

    if understandings is None:
        stage_status[_STAGE1_NAME] = {"complete": False, "num_statements": 0}
        stage_status[_STAGE2_NAME] = {"complete": False, "num_statements": 0}
        stage_status[_STAGE3_NAME] = {"complete": False, "completed_items": 0}
        _save_stage_status(config.output_path, stage_status)

    if ideations is None:
        stage_status[_STAGE2_NAME] = {"complete": False, "num_statements": 0}
        stage_status[_STAGE3_NAME] = {"complete": False, "completed_items": 0}
        _save_stage_status(config.output_path, stage_status)
        _clear_stage3_checkpoint(config.output_path)
        existing_extractions = {}

    if isinstance(config.ideation_model, VLLMConfig) and (
        pending_stage1_statements or ideations is None or (same_local_model and ideations is not None)
    ):
        with BatchedVllmServeSession(config.ideation_model) as active_ideation_session:
            if pending_stage1_statements:
                stage1_understandings = _run_understanding_stage(
                    pending_stage1_statements,
                    config,
                    active_ideation_session,
                )
                if understandings is None:
                    understandings = {}
                understandings.update(stage1_understandings)
                _finalize_stage1_success(config.output_path, statements, understandings, stage_status)
            if ideations is None:
                assert understandings is not None
                stage2_plans = get_stage2_plans()

                def save_statement_checkpoint(statement_id: str, ideation: dict[str, Any]) -> None:
                    _save_statement_ideation(
                        config.output_path,
                        statement_id,
                        ideation,
                        _make_concretization_fingerprint(stage2_plans[statement_id]),
                    )

                ideations = _run_concretization_stage(
                    understandings,
                    config,
                    active_ideation_session,
                    plans=stage2_plans,
                    completed_ideations=partial_ideations,
                    statement_checkpoint_callback=save_statement_checkpoint,
                )
                _save_ideation_checkpoint(config.output_path, statements, understandings, ideations)
                stage_status[_STAGE2_NAME] = {"complete": True, "num_statements": len(ideations)}
                stage_status[_STAGE3_NAME] = {"complete": False, "completed_items": 0}
                _save_stage_status(config.output_path, stage_status)
            if same_local_model:
                logger.info("Reusing the ideation vLLM serve session for extraction")
                assert ideations is not None
                all_prompts = _run_extraction_stage(
                    ideations,
                    config,
                    active_ideation_session,
                    existing_extractions=existing_extractions,
                    checkpoint_callback=checkpoint_stage3_batch,
                )
            session_name = "ideation_extract_shared" if same_local_model else "ideation"
            metrics_sessions.append((session_name, active_ideation_session.metrics_snapshot()))
    else:
        if pending_stage1_statements:
            stage1_understandings = _run_understanding_stage(pending_stage1_statements, config, None)
            if understandings is None:
                understandings = {}
            understandings.update(stage1_understandings)
            _finalize_stage1_success(config.output_path, statements, understandings, stage_status)
        if ideations is None:
            assert understandings is not None
            ideations = _run_concretization_stage(understandings, config, None)
            _save_ideation_checkpoint(config.output_path, statements, understandings, ideations)
            stage_status[_STAGE2_NAME] = {"complete": True, "num_statements": len(ideations)}
            stage_status[_STAGE3_NAME] = {"complete": False, "completed_items": 0}
            _save_stage_status(config.output_path, stage_status)

    if all_prompts is None:
        assert ideations is not None
        if isinstance(config.extract_model, VLLMConfig):
            with BatchedVllmServeSession(config.extract_model) as extract_session:
                all_prompts = _run_extraction_stage(
                    ideations,
                    config,
                    extract_session,
                    existing_extractions=existing_extractions,
                    checkpoint_callback=checkpoint_stage3_batch,
                )
            metrics_sessions.append(("extract", extract_session.metrics_snapshot()))
        else:
            all_prompts = _run_extraction_stage(
                ideations,
                config,
                None,
                existing_extractions=existing_extractions,
                checkpoint_callback=checkpoint_stage3_batch,
            )

    assert all_prompts is not None
    assert understandings is not None
    assert ideations is not None
    logger.info("Total prompts generated: %d", len(all_prompts))
    write_sharded_jsonl_gz(all_prompts, config.output_path, shard_size=5000)
    _save_artifacts(config.output_path, statements, understandings, ideations)
    stage_status[_STAGE3_NAME] = {
        "complete": True,
        "completed_items": len(_build_extraction_work_items(ideations)),
    }
    _save_stage_status(config.output_path, stage_status)
    if metrics_sessions:
        write_vllm_metrics_artifact(
            f"{config.output_path}/artifacts/vllm_metrics.json",
            logical_stage="prompt_generation",
            sessions=metrics_sessions,
        )


def _load_filtered_statements(config: PromptGenConfig) -> dict[str, Statement]:
    """Load statements from spec and apply optional filtering."""
    statements = load_spec(config.spec_path)
    logger.info("Loaded %d statements from spec", len(statements))

    if config.statement_ids:
        statements = {sid: s for sid, s in statements.items() if sid in config.statement_ids}
        logger.info("Filtered to %d statements", len(statements))
    return statements


def _run_understanding_stage(
    statements: dict[str, Statement],
    config: PromptGenConfig,
    session: BatchedVllmServeSession | None,
) -> dict[str, dict[str, Any]]:
    logger.info("Stage 1: Generating understanding for %d statements", len(statements))
    if session is not None:
        return _run_understanding_local(statements, config, session)

    understandings: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    pending_statements = list(statements.items())
    ideation_workers = config.ideation_workers
    reporter = LiveProgressReporter(
        stage_name="Stage 1",
        total_items=len(statements),
        batch_size=max(1, ideation_workers),
    )

    for attempt in range(1, config.understanding_max_attempts + 1):
        if not pending_statements:
            break
        next_pending: list[tuple[str, Statement]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=ideation_workers) as pool:
            future_map = {
                pool.submit(_understanding_request, stmt, config): (sid, stmt) for sid, stmt in pending_statements
            }
            for future in concurrent.futures.as_completed(future_map):
                sid, statement = future_map[future]
                try:
                    raw_response = future.result()
                    _append_stage1_attempt_records(
                        config.output_path,
                        [_make_stage1_attempt_record(sid, attempt, raw_response)],
                    )
                    understandings[sid] = _parse_understanding_response(
                        statement,
                        raw_response,
                        config.ideation_model,
                    )
                    failures.pop(sid, None)
                except Exception as exc:
                    failures[sid] = str(exc)
                    next_pending.append((sid, statement))
                    logger.warning("Stage1 attempt %d failed for '%s': %s", attempt, sid, exc)
                    reporter.maybe_log(
                        len(understandings),
                        details=[f"attempt {attempt}", f"retries pending={len(next_pending)}"],
                    )
                    continue
                reporter.maybe_log(len(understandings), details=[f"attempt {attempt}"])
        pending_statements = next_pending
        if pending_statements:
            reporter.maybe_log(
                len(understandings),
                details=[f"attempt {attempt}", f"retries pending={len(pending_statements)}"],
                force=True,
            )

    if pending_statements:
        detail = "; ".join(f"{sid}: {failures[sid]}" for sid, _statement in pending_statements)
        raise RuntimeError(f"Stage 1 failed for {len(failures)} statement(s): {detail}")

    return understandings


def _finalize_stage1_success(
    output_path: str,
    statements: dict[str, Statement],
    understandings: dict[str, dict[str, Any]],
    stage_status: dict[str, dict[str, int | bool]],
) -> None:
    _save_understanding_checkpoint(output_path, statements, understandings)
    stage_status[_STAGE1_NAME] = {"complete": True, "num_statements": len(understandings)}
    stage_status[_STAGE2_NAME] = {"complete": False, "num_statements": 0}
    stage_status[_STAGE3_NAME] = {"complete": False, "completed_items": 0}
    _save_stage_status(output_path, stage_status)


def _run_concretization_stage(
    understandings: dict[str, dict[str, Any]],
    config: PromptGenConfig,
    session: BatchedVllmServeSession | None,
    *,
    plans: dict[str, _ConcretizationPlan] | None = None,
    completed_ideations: dict[str, dict[str, Any]] | None = None,
    statement_checkpoint_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    logger.info("Stage 2: Concretizing %d statements", len(understandings))
    ideations: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, str]] = []

    if session is not None:
        return _run_concretization_stage_local(
            understandings,
            config,
            session,
            plans=plans,
            completed_ideations=completed_ideations,
            statement_checkpoint_callback=statement_checkpoint_callback,
        )

    # API path already executes per statement via ThreadPoolExecutor, but still
    # persists all-or-nothing after all futures complete.
    concretize_workers = min(8, config.concretize_workers)
    reporter = LiveProgressReporter(
        stage_name="Stage 2",
        total_items=len(understandings),
        batch_size=max(1, concretize_workers),
    )
    processed_statements = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concretize_workers) as pool:
        future_map = {
            pool.submit(_run_concretization, sid, understanding, config): sid
            for sid, understanding in understandings.items()
        }
        for future in concurrent.futures.as_completed(future_map):
            sid = future_map[future]
            try:
                ideations[sid] = future.result()
            except Exception as exc:
                failures.append((sid, str(exc)))
                logger.error("Stage2 failed for '%s': %s", sid, exc)
            processed_statements += 1
            reporter.maybe_log(
                processed_statements,
                details=[f"succeeded={len(ideations)}", f"failed={len(failures)}"],
            )

    if failures:
        detail = "; ".join(f"{sid}: {msg}" for sid, msg in failures)
        raise RuntimeError(f"Stage 2 failed for {len(failures)} statement(s): {detail}")

    return ideations


def _run_extraction_stage(
    ideations: dict[str, dict[str, Any]],
    config: PromptGenConfig,
    session: BatchedVllmServeSession | None,
    *,
    existing_extractions: dict[tuple[str, int], dict[str, str]] | None = None,
    checkpoint_callback: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> list[dict[str, Any]]:
    logger.info("Stage 3: Extracting prompts from %d statements", len(ideations))

    if session is not None:
        return _run_extraction_stage_local(
            ideations,
            config,
            session,
            existing_extractions=existing_extractions,
            checkpoint_callback=checkpoint_callback,
        )

    return _run_extraction_stage_api(
        ideations,
        config,
        existing_extractions=existing_extractions,
        checkpoint_callback=checkpoint_callback,
    )


def load_sharded_jsonl_gz(path: str) -> list[dict]:
    """Read all shards from a directory of JSONL.GZ files. Supports local and GCS paths."""
    fs, base_path = url_to_fs(path)
    protocol = getattr(fs, "protocol", None)
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]

    records: list[dict] = []
    for shard_file in sorted(fs.glob(f"{base_path}/*.jsonl.gz")):
        full_path = f"{protocol}://{shard_file}" if protocol else shard_file
        records.extend(load_jsonl(full_path))
    return records


def write_sharded_jsonl_gz(records: list[dict[str, Any]], output_path: str, shard_size: int = 5000) -> None:
    """Write records as sharded JSONL.GZ files. Supports both local and GCS paths."""
    for shard_idx in range(0, max(1, len(records)), shard_size):
        shard = records[shard_idx : shard_idx + shard_size]
        shard_num = shard_idx // shard_size
        shard_file = f"{output_path}/shard_{shard_num:05d}.jsonl.gz"
        write_jsonl_file(shard, shard_file)

    logger.info("Wrote %d records to %d shards in %s", len(records), (len(records) // shard_size) + 1, output_path)


def _save_artifacts(
    output_path: str,
    statements: dict[str, Statement],
    understandings: dict[str, dict[str, Any]],
    ideations: dict[str, dict[str, Any]],
) -> None:
    """Save intermediate pipeline artifacts for reproducibility and inspection.

    Writes per-statement JSON files:
      artifacts/<statement_id>/understanding.json  — Stage 1: axes, understanding, motivation
      artifacts/<statement_id>/ideation.json       — Stage 2: covering plan, scenarios, rubrics
      artifacts/summary.json                       — overview with statement list and counts
    """
    fs, base_path = url_to_fs(output_path)

    # Collect all statement IDs that have artifacts, create each directory once
    all_sids = set(understandings.keys()) | set(ideations.keys())
    for sid in all_sids:
        fs.makedirs(f"{base_path}/artifacts/{sid}", exist_ok=True)

    for sid, understanding in understandings.items():
        serializable = dict(understanding)
        if not isinstance(serializable.get("model"), str):
            serializable["model"] = str(serializable.get("model", ""))
        with fs.open(f"{base_path}/artifacts/{sid}/understanding.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(serializable, indent=2, ensure_ascii=False))

    for sid, ideation in ideations.items():
        serializable = dict(ideation)
        if not isinstance(serializable.get("model"), str):
            serializable["model"] = str(serializable.get("model", ""))
        with fs.open(f"{base_path}/artifacts/{sid}/ideation.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(serializable, indent=2, ensure_ascii=False))

    # Write summary
    fs.makedirs(f"{base_path}/artifacts", exist_ok=True)
    summary = {
        "statements": sorted(statements.keys()),
        "num_statements": len(statements),
        "num_understandings": len(understandings),
        "num_ideations": len(ideations),
        "understandings": {
            sid: {
                "num_axes": len(u.get("variation_axes", [])),
                "axes": [ax.get("axis") for ax in u.get("variation_axes", [])],
            }
            for sid, u in understandings.items()
        },
        "ideations": {
            sid: {
                "num_configs": idn.get("num_configs", 0),
                "covering_strength": idn.get("covering_strength", 0),
                "coverage_stats": idn.get("coverage_stats", {}),
            }
            for sid, idn in ideations.items()
        },
    }
    summary_path = f"{base_path}/artifacts/summary.json"
    with fs.open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))

    logger.info(
        "Saved artifacts for %d statements to %s/artifacts/",
        len(understandings),
        output_path,
    )
