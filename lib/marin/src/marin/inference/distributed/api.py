# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level distributed inference entry point.

``inference(model, dataset, config) -> InferenceResult`` is the stateless
caller-facing API. It normalizes the input, builds a per-run staging area
in the results bucket, and hands off to the meta-coordinator (which submits
one Fray job per region).

This function **blocks** until every regional job has terminated. The
returned `InferenceResult` has ``missing_shards`` populated if any shard
failed to land in the results bucket.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from marin.rl.placement import marin_prefix_for_region

from .config import InferenceConfig, ModelSpec
from .input import list_input_files, materialize_inline_input
from .meta_coordinator import run_meta_coordinator
from .output import InferenceResult

logger = logging.getLogger(__name__)

# Acceptable input types for the public API.
InferenceInput = str | Sequence[dict[str, Any]]


def inference(
    model: str | ModelSpec,
    dataset: InferenceInput,
    config: InferenceConfig,
) -> InferenceResult:
    """Run distributed inference and return the result handle.

    Args:
        model: Either a `ModelSpec` or a string that gets wrapped into one
            with default ``engine_kwargs={}``. The string follows the
            `ModelSpec.model` shape: HF id, ``marin://...``, or ``gs://...``.
        dataset: Either a path/glob pointing at JSONL[.gz] files of
            `PromptRecord` dicts, or an in-memory sequence of those dicts.
            In-memory input is materialized to JSONL files in the results
            bucket so all regional workers can read it.
        config: Run configuration. See `InferenceConfig`.

    Returns:
        `InferenceResult` summarizing the run. Inspect ``missing_shards`` to
        confirm completeness.
    """
    model_spec = model if isinstance(model, ModelSpec) else ModelSpec(model=model)
    run_id = uuid.uuid4().hex[:12]

    run_prefix = _run_prefix(config, run_id)
    results_uri = f"{run_prefix}/outputs"
    inputs_uri = f"{run_prefix}/inputs"
    logger.info(
        "Starting inference run: job_name=%s, run_id=%s, regions=%s, results_uri=%s",
        config.job_name,
        run_id,
        list(config.regions),
        results_uri,
    )

    input_files = _prepare_input(dataset, inputs_uri, config.shard_size)
    logger.info("Inference input: %d file(s) under %s", len(input_files), inputs_uri)

    return run_meta_coordinator(
        config=config,
        input_files=input_files,
        model_spec=model_spec,
        results_uri=results_uri,
        run_id=run_id,
    )


def _run_prefix(config: InferenceConfig, run_id: str) -> str:
    """``gs://marin-{results_region}/<job_name>/<run_id>``."""
    bucket = marin_prefix_for_region(config.results_region)
    return f"{bucket}/{config.job_name}/{run_id}"


def _prepare_input(dataset: InferenceInput, inputs_uri: str, shard_size: int) -> list[str]:
    """Resolve the caller's input to a sorted list of JSONL file URIs.

    Inline records are chunked into ``shard_size``-sized files so that the
    downstream pipeline (one Zephyr shard per input file) honors the caller's
    requested shard granularity. Path/glob inputs are returned as-is — the
    pre-existing file layout determines sharding there.
    """
    if isinstance(dataset, str):
        return list_input_files(dataset)
    return materialize_inline_input(list(dataset), output_dir=inputs_uri, records_per_file=shard_size)
