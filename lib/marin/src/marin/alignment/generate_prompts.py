# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1: Generate diverse evaluation prompts from a behavioral specification.

Implements the 3-stage Bloom pipeline:
  Stage 1: Understanding — analyze each behavior statement, generate variation axes
  Stage 2: Concretization — covering array → concrete scenarios via LLM
  Stage 3: Extraction — extract clean system_prompt + user_message from scenario prose

Ported from bloom/synthetic_pipeline/stage{1,2,3}.py.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from marin.alignment.coverage import compute_coverage_stats, generate_covering_configs, make_tags
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


@dataclass(frozen=True)
class PromptGenConfig:
    """Configuration for prompt generation from a specification."""

    spec_path: str
    output_path: str

    # LLM settings
    ideation_model: str = "openai/gpt-4.1"
    extract_model: str = "openai/gpt-4.1-mini"

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
    concretize_temperature: float = 1.0

    # Filtering
    statement_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def load_spec(spec_path: str) -> dict[str, Statement]:
    """Load behavioral statements from a JSONL file. Supports both local and GCS paths."""
    import fsspec

    fs, resolved_path = fsspec.core.url_to_fs(spec_path)
    statements: dict[str, Statement] = {}
    is_gz = resolved_path.endswith(".gz")

    with fs.open(resolved_path, "rb") as raw_f:
        f = gzip.open(raw_f, "rt", encoding="utf-8") if is_gz else raw_f
        try:
            for line in f:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                stmt = Statement.from_dict(data)
                statements[stmt.id] = stmt
        finally:
            if is_gz:
                f.close()
    return statements


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
        model_id=config.ideation_model,
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
        model_id=model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=16000,
        temperature=temperature,
    )
    parsed_scenarios = _parse_concretize_response(response.content)
    if len(parsed_scenarios) != len(configs):
        logger.warning(
            "Stage2 concretize parsing mismatch for '%s': expected %d scenarios, got %d. Padding with empty scenarios.",
            statement_id,
            len(configs),
            len(parsed_scenarios),
        )
        while len(parsed_scenarios) < len(configs):
            parsed_scenarios.append({"description": "", "rubric": ""})
    return parsed_scenarios


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
    all_scenarios: list[dict[str, Any]] = [{}] * len(configs)
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
            logger.warning("Stage3: missing <scenario_%d> block, using empty", idx)
            out.append({"system_prompt": "", "user_message": ""})
            continue
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
) -> list[dict[str, str]]:
    """Extract clean prompts from a batch of scenarios."""
    system_prompt, user_prompt = make_extraction_prompt(scenarios, batch_start_idx, include_system_prompt=True)
    response = llm_chat_single(
        model_id=model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=16000,
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

    all_extractions: list[dict[str, str]] = [{"system_prompt": "", "user_message": ""}] * len(variations)
    failures: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.extract_workers) as pool:
        future_map = {
            pool.submit(_extract_batch, batch, batch_start, config.extract_model): (batch_start, len(batch))
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
    statements = load_spec(config.spec_path)
    logger.info("Loaded %d statements from spec", len(statements))

    # Filter to requested statements if specified
    if config.statement_ids:
        statements = {sid: s for sid, s in statements.items() if sid in config.statement_ids}
        logger.info("Filtered to %d statements", len(statements))

    all_prompts: list[dict[str, Any]] = []

    # Stage 1: Understanding (parallel across statements)
    logger.info("Stage 1: Generating understanding for %d statements", len(statements))
    understandings: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.ideation_workers) as pool:
        future_map = {pool.submit(_run_understanding, stmt, config): sid for sid, stmt in statements.items()}
        for future in concurrent.futures.as_completed(future_map):
            sid = future_map[future]
            try:
                understandings[sid] = future.result()
            except Exception as exc:
                failures.append((sid, str(exc)))
                logger.error("Stage1 failed for '%s': %s", sid, exc)

    if failures:
        logger.warning("Stage1 failed for %d statements, continuing with %d", len(failures), len(understandings))

    # Stage 2: Concretization (parallel across statements)
    logger.info("Stage 2: Concretizing %d statements", len(understandings))
    ideations: dict[str, dict[str, Any]] = {}
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, config.concretize_workers)) as pool:
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
        logger.warning("Stage2 failed for %d statements, continuing with %d", len(failures), len(ideations))

    # Stage 3: Extraction (parallel across statements)
    logger.info("Stage 3: Extracting prompts from %d statements", len(ideations))
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, config.extract_workers)) as pool:
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
        logger.warning("Stage3 failed for %d statements", len(failures))

    logger.info("Total prompts generated: %d", len(all_prompts))

    # Write output as sharded JSONL.GZ
    _write_sharded_jsonl_gz(all_prompts, config.output_path, shard_size=5000)


def _write_sharded_jsonl_gz(records: list[dict[str, Any]], output_path: str, shard_size: int = 5000) -> None:
    """Write records as sharded JSONL.GZ files. Supports both local and GCS paths."""
    import fsspec

    fs, base_path = fsspec.core.url_to_fs(output_path)
    fs.makedirs(base_path, exist_ok=True)

    for shard_idx in range(0, max(1, len(records)), shard_size):
        shard = records[shard_idx : shard_idx + shard_size]
        shard_num = shard_idx // shard_size
        shard_file = f"{base_path}/shard_{shard_num:05d}.jsonl.gz"
        with fs.open(shard_file, "wb") as raw_f:
            with gzip.open(raw_f, "wt", encoding="utf-8") as f:
                for record in shard:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d records to %d shards in %s", len(records), (len(records) // shard_size) + 1, output_path)
