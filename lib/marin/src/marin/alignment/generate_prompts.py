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
import contextlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from iris.marin_fs import url_to_fs
from zephyr import load_jsonl, write_jsonl_file

from marin.alignment.coverage import compute_coverage_stats, generate_covering_configs, make_tags
from marin.alignment.inference_config import InferenceConfig, VLLMConfig
from marin.alignment.llm_client import llm_chat_single, vllm_engine
from marin.alignment.prompts.concretize import make_concretize_prompt
from marin.alignment.prompts.extract import make_extraction_prompt
from marin.alignment.prompts.understanding import (
    STANDARD_DEMOGRAPHIC_AXES,
    make_behavior_understanding_prompt,
    make_understanding_system_prompt,
)
from marin.alignment.types import Statement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptGenConfig:
    """Configuration for prompt generation from a specification."""

    spec_path: str
    output_path: str

    # LLM settings — InferenceConfig or string (string → LiteLLMConfig)
    ideation_model: InferenceConfig | str = "openai/gpt-4.1"
    extract_model: InferenceConfig | str = "openai/gpt-4.1-mini"

    # Covering array
    covering_strength: int = 3
    covering_seed: int = 42

    # Batch sizes
    concretize_batch_size: int = 10
    extract_batch_size: int = 10

    # Parallelism
    ideation_workers: int = 32
    concretize_workers: int = 32
    extract_workers: int = 128

    # LLM parameters
    understanding_max_tokens: int = 4000
    understanding_temperature: float = 1.0
    concretize_max_tokens: int = 16000
    concretize_temperature: float = 1.0
    extract_max_tokens: int = 16000

    # Filtering
    statement_ids: list[str] | None = None


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


def _run_understanding(
    statement: Statement,
    config: PromptGenConfig,
) -> dict[str, Any]:
    """Run Stage 1 for a single statement: understanding + variation axes."""
    system_prompt = make_understanding_system_prompt()
    user_prompt = make_behavior_understanding_prompt(statement.id, statement.text)

    response = llm_chat_single(
        config=config.ideation_model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=config.understanding_max_tokens,
        temperature=config.understanding_temperature,
    )

    understanding = _extract_tag(response.content, "behavior_understanding")
    scientific_motivation = _extract_tag(response.content, "scientific_motivation")
    behavior_specific_axes = _parse_variation_axes(response.content)
    variation_axes = behavior_specific_axes + [dict(ax) for ax in STANDARD_DEMOGRAPHIC_AXES]

    return {
        "behavior_name": statement.id,
        "understanding": understanding,
        "scientific_motivation": scientific_motivation,
        "variation_axes": variation_axes,
        "model": config.ideation_model,
    }


# ---------------------------------------------------------------------------
# Stage 2: Concretization
# ---------------------------------------------------------------------------


def _parse_concretize_response(content: str) -> list[dict[str, Any]]:
    scenarios = re.findall(r"<scenario>(.*?)</scenario>", content, re.DOTALL)
    rubrics = re.findall(r"<rubric>(.*?)</rubric>", content, re.DOTALL)
    out: list[dict[str, Any]] = []
    for i, scenario_text in enumerate(scenarios):
        rubric = rubrics[i].strip() if i < len(rubrics) else ""
        out.append({"description": scenario_text.strip(), "rubric": rubric})
    return out


def _concretize_batch(
    statement_id: str,
    understanding_data: dict[str, Any],
    axes: list[dict[str, Any]],
    configs: list[dict[str, str]],
    batch_start_idx: int,
    model: str,
    temperature: float,
    max_tokens: int = 16000,
) -> list[dict[str, Any]]:
    """Concretize a batch of axis configs into scenarios."""
    system_prompt, user_prompt = make_concretize_prompt(
        behavior_name=statement_id,
        behavior_understanding=understanding_data.get("understanding", ""),
        scientific_motivation=understanding_data.get("scientific_motivation", ""),
        transcript_analyses=understanding_data.get("transcript_analyses", []),
        configs=configs,
        axes_metadata=axes,
        batch_start_idx=batch_start_idx,
    )
    response = llm_chat_single(
        config=model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parsed_scenarios = _parse_concretize_response(response.content)
    if len(parsed_scenarios) < len(configs) // 2:
        raise RuntimeError(
            f"Stage2 concretize parsing mismatch for '{statement_id}': "
            f"expected {len(configs)} scenarios, got {len(parsed_scenarios)} (<50%)"
        )
    if len(parsed_scenarios) < len(configs):
        logger.warning(
            "Stage2: got %d/%d scenarios for '%s' (minor shortfall, continuing with partial batch)",
            len(parsed_scenarios),
            len(configs),
            statement_id,
        )
    return parsed_scenarios[: len(configs)]


def _run_concretization(
    statement_id: str,
    understanding: dict[str, Any],
    config: PromptGenConfig,
) -> dict[str, Any]:
    """Run Stage 2 for a single statement: covering array → concretize via LLM."""
    axes = understanding["variation_axes"]
    configs = generate_covering_configs(axes, t=config.covering_strength, seed=config.covering_seed)
    stats = compute_coverage_stats(configs, axes, t=config.covering_strength)

    # Batch the configs
    batches: list[tuple[int, list[dict[str, str]]]] = []
    for start in range(0, len(configs), config.concretize_batch_size):
        batch_configs = configs[start : start + config.concretize_batch_size]
        batches.append((start + 1, batch_configs))

    # Concretize in parallel
    all_scenarios: list[dict[str, Any]] = [{} for _ in range(len(configs))]
    failures: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.concretize_workers) as pool:
        future_map = {
            pool.submit(
                _concretize_batch,
                statement_id,
                understanding,
                axes,
                batch_configs,
                batch_start_idx,
                config.ideation_model,
                config.concretize_temperature,
                config.concretize_max_tokens,
            ): (batch_start_idx, len(batch_configs))
            for batch_start_idx, batch_configs in batches
        }
        for future in concurrent.futures.as_completed(future_map):
            batch_start_idx, _batch_size = future_map[future]
            try:
                results = future.result()
                offset = batch_start_idx - 1
                for i, scenario in enumerate(results):
                    if offset + i < len(all_scenarios):
                        all_scenarios[offset + i] = scenario
            except Exception as exc:
                failures.append(f"batch {batch_start_idx}: {exc}")

    if failures:
        raise RuntimeError(f"Stage2 failed for '{statement_id}': {'; '.join(failures)}")

    # Build variations with tags
    variations: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs):
        scenario = all_scenarios[i] if i < len(all_scenarios) else {}
        variations.append(
            {
                "description": scenario.get("description", ""),
                "rubric": scenario.get("rubric", ""),
                "config_id": f"cfg_{i:03d}",
                "axis_config": cfg,
                "tags": make_tags(cfg, axes),
            }
        )

    return {
        "behavior_name": statement_id,
        "model": config.ideation_model,
        "covering_strength": config.covering_strength,
        "num_configs": len(configs),
        "coverage_stats": stats,
        "variations": variations,
    }


# ---------------------------------------------------------------------------
# Stage 3: Extraction
# ---------------------------------------------------------------------------


def _parse_extraction_response(
    content: str,
    batch_size: int,
    batch_start_idx: int,
    include_system_prompt: bool = True,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(batch_size):
        idx = batch_start_idx + i
        match = re.search(rf"<scenario_{idx}>(.*?)</scenario_{idx}>", content, re.DOTALL)
        if not match:
            raise RuntimeError(f"Stage3: missing <scenario_{idx}> block in extraction response")
        block = match.group(1)
        if include_system_prompt:
            sp_match = re.search(r"<system_prompt>(.*?)</system_prompt>", block, re.DOTALL)
            system_prompt = sp_match.group(1).strip() if sp_match else ""
        else:
            system_prompt = ""
        um_match = re.search(r"<user_message>(.*?)</user_message>", block, re.DOTALL)
        user_message = um_match.group(1).strip() if um_match else ""
        out.append({"system_prompt": system_prompt, "user_message": user_message})
    return out


def _extract_batch(
    scenarios: list[dict[str, Any]],
    batch_start_idx: int,
    model: str,
    max_tokens: int = 16000,
) -> list[dict[str, str]]:
    """Extract clean prompts from a batch of scenarios."""
    system_prompt, user_prompt = make_extraction_prompt(scenarios, batch_start_idx, include_system_prompt=True)
    response = llm_chat_single(
        config=model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return _parse_extraction_response(response.content, len(scenarios), batch_start_idx, include_system_prompt=True)


def _run_extraction(
    statement_id: str,
    ideation: dict[str, Any],
    config: PromptGenConfig,
) -> list[dict[str, Any]]:
    """Run Stage 3 for a single statement: extract system_prompt + user_message."""
    variations = [v for v in ideation.get("variations", []) if str(v.get("description", "")).strip()]

    batches: list[tuple[int, list[dict[str, Any]]]] = []
    for start in range(0, len(variations), config.extract_batch_size):
        batch = variations[start : start + config.extract_batch_size]
        batches.append((start, batch))

    all_extractions: list[dict[str, str]] = [{"system_prompt": "", "user_message": ""} for _ in range(len(variations))]
    failures: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.extract_workers) as pool:
        future_map = {
            pool.submit(_extract_batch, batch, batch_start, config.extract_model, config.extract_max_tokens): (
                batch_start,
                len(batch),
            )
            for batch_start, batch in batches
        }
        for future in concurrent.futures.as_completed(future_map):
            batch_start, _batch_size = future_map[future]
            try:
                results = future.result()
                for i, extraction in enumerate(results):
                    if batch_start + i < len(all_extractions):
                        all_extractions[batch_start + i] = extraction
            except Exception as exc:
                failures.append(f"batch {batch_start}: {exc}")

    if failures:
        raise RuntimeError(f"Stage3 failed for '{statement_id}': {'; '.join(failures)}")

    # Merge extractions with variation metadata
    prompts: list[dict[str, Any]] = []
    for i, variation in enumerate(variations):
        extraction = all_extractions[i] if i < len(all_extractions) else {}
        system_prompt = extraction.get("system_prompt", "")
        user_message = extraction.get("user_message", "")
        if not user_message:
            continue
        prompts.append(
            {
                "behavior_id": statement_id,
                "system_prompt": system_prompt,
                "user_message": user_message,
                "rubric": variation.get("rubric", ""),
                "config_id": variation.get("config_id", ""),
                "axis_config": variation.get("axis_config", {}),
                "tags": variation.get("tags", []),
            }
        )

    return prompts


# ---------------------------------------------------------------------------
# Full pipeline: spec → prompts
# ---------------------------------------------------------------------------


def generate_prompts_from_spec(config: PromptGenConfig) -> None:
    """Run the full 3-stage prompt generation pipeline.

    Reads a specification JSONL, runs understanding → concretization → extraction
    for each behavior statement, and writes the results as sharded JSONL.GZ files.
    """
    # Keep vLLM engines alive for the duration of all stages
    with contextlib.ExitStack() as stack:
        if isinstance(config.ideation_model, VLLMConfig):
            stack.enter_context(vllm_engine(config.ideation_model))
        if isinstance(config.extract_model, VLLMConfig):
            stack.enter_context(vllm_engine(config.extract_model))
        _generate_prompts_inner(config)


def _generate_prompts_inner(config: PromptGenConfig) -> None:
    """Inner implementation of prompt generation."""
    statements = load_spec(config.spec_path)
    logger.info("Loaded %d statements from spec", len(statements))

    # Filter to requested statements if specified
    if config.statement_ids:
        statements = {sid: s for sid, s in statements.items() if sid in config.statement_ids}
        logger.info("Filtered to %d statements", len(statements))

    all_prompts: list[dict[str, Any]] = []

    # Stage 1: Understanding (sequential for vLLM, parallel for API)
    logger.info("Stage 1: Generating understanding for %d statements", len(statements))
    understandings: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, str]] = []

    # vLLM is not thread-safe — use 1 worker when running locally
    ideation_workers = 1 if isinstance(config.ideation_model, VLLMConfig) else config.ideation_workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=ideation_workers) as pool:
        future_map = {pool.submit(_run_understanding, stmt, config): sid for sid, stmt in statements.items()}
        for future in concurrent.futures.as_completed(future_map):
            sid = future_map[future]
            try:
                understandings[sid] = future.result()
            except Exception as exc:
                failures.append((sid, str(exc)))
                logger.error("Stage1 failed for '%s': %s", sid, exc)

    if failures:
        detail = "; ".join(f"{sid}: {msg}" for sid, msg in failures)
        raise RuntimeError(f"Stage 1 failed for {len(failures)} statement(s): {detail}")

    # Stage 2: Concretization (parallel across statements)
    logger.info("Stage 2: Concretizing %d statements", len(understandings))
    ideations: dict[str, dict[str, Any]] = {}
    failures = []

    concretize_workers = 1 if isinstance(config.ideation_model, VLLMConfig) else min(8, config.concretize_workers)
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

    if failures:
        detail = "; ".join(f"{sid}: {msg}" for sid, msg in failures)
        raise RuntimeError(f"Stage 2 failed for {len(failures)} statement(s): {detail}")

    # Stage 3: Extraction (parallel across statements)
    logger.info("Stage 3: Extracting prompts from %d statements", len(ideations))
    failures = []

    extract_workers = 1 if isinstance(config.extract_model, VLLMConfig) else min(8, config.extract_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=extract_workers) as pool:
        future_map = {pool.submit(_run_extraction, sid, ideation, config): sid for sid, ideation in ideations.items()}
        for future in concurrent.futures.as_completed(future_map):
            sid = future_map[future]
            try:
                prompts = future.result()
                all_prompts.extend(prompts)
                logger.info("Stage3 completed for '%s': %d prompts", sid, len(prompts))
            except Exception as exc:
                failures.append((sid, str(exc)))
                logger.error("Stage3 failed for '%s': %s", sid, exc)

    if failures:
        detail = "; ".join(f"{sid}: {msg}" for sid, msg in failures)
        raise RuntimeError(f"Stage 3 failed for {len(failures)} statement(s): {detail}")

    logger.info("Total prompts generated: %d", len(all_prompts))

    # Write output as sharded JSONL.GZ
    write_sharded_jsonl_gz(all_prompts, config.output_path, shard_size=5000)

    # Save intermediate artifacts for reproducibility and inspection
    _save_artifacts(config.output_path, statements, understandings, ideations)


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
