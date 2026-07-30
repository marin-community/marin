# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared chunk-execution backend for the joint-decode algorithm modules.

Bridges the marin executor step configs in ``joint_decode.py`` and
``joint_decode_avg.py`` onto the joint-decode package (modern protocol:
sliding-window admission, multi-token holds, force-stop on peer finish).
Those two modules keep their config dataclasses and step names byte-stable
— executor step hashes cover the versioned config values under the step
name, so completed experiments keep resolving — and delegate execution
here.

This module imports the joint-decode package at the top level and therefore
must only be imported inside step-execution functions: the package rides
the linux-only vllm extra, and the algorithm modules must stay importable
for step construction everywhere.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from joint_decode.config import JointDecodeSamplingConfig as PackageSamplingConfig
from joint_decode.coordinator import JointDecoder, SelectTokens
from joint_decode.tpu.config import JointDecodeConfig as PackageConfig
from joint_decode.tpu.config import JointDecodeModelConfig as PackageModelConfig
from joint_decode.tpu.config import TpuPlacement
from joint_decode.tpu.decoder import joint_decoder
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, fsspec_exists, localize_mirror_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineModelParams:
    """One engine's model parameters, decoupled from the hash-frozen configs."""

    model_path: str
    max_model_len: int
    gpu_memory_utilization: float | None
    enable_prefix_caching: bool
    apply_rpa_block_size_patch: bool


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str
    success_path: str


def chunk_specs(chunks_dir: str, num_prompts: int, n_samples: int, chunk_size: int) -> list[ChunkSpec]:
    total_requests = num_prompts * n_samples
    return [
        ChunkSpec(
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_end=min(start + chunk_size, total_requests),
            output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
            success_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.SUCCESS"),
        )
        for chunk_id, start in enumerate(range(0, total_requests, chunk_size))
    ]


def _resolve_model_path(model_path: str) -> str:
    """Resolve a checkpoint root to a concrete regional HF directory.

    Runs on the TPU VM (mirror localization is region-dependent); the
    joint-decode package takes only already-resolved paths.
    """
    resolved = discover_hf_checkpoints(model_path)[-1]
    return localize_mirror_path(resolved)


def _package_model_config(
    params: EngineModelParams,
    placement: EnginePlacement,
    resolved_path: str,
) -> PackageModelConfig:
    return PackageModelConfig(
        model_path=resolved_path,
        placement=TpuPlacement(
            visible_chips=placement.visible_chips,
            chips_per_process_bounds=placement.chips_per_process_bounds,
            tensor_parallel_size=placement.tensor_parallel_size,
        ),
        max_model_len=params.max_model_len,
        gpu_memory_utilization=params.gpu_memory_utilization,
        enable_prefix_caching=params.enable_prefix_caching,
        apply_rpa_block_size_patch=params.apply_rpa_block_size_patch,
    )


@contextlib.contextmanager
def open_joint_decoder(
    *,
    decoder: EngineModelParams,
    advisor: EngineModelParams,
    max_tokens: int,
    # None -> same cap as the decoder side; set for cross-tokenizer pairs
    # where the advisor needs fertility headroom on the same text.
    advisor_max_tokens: int | None = None,
    top_k_a: int,
    top_k_b: int,
    seed: int,
    stop: tuple[str, ...],
    select_token: SelectTokens,
    decoder_placement: EnginePlacement,
    advisor_placement: EnginePlacement,
    max_microbatch_size: int,
    max_num_batched_tokens: int | None,
    barrier_timeout_s: float,
) -> Iterator[JointDecoder]:
    """Open one package JointDecoder pair from marin-side engine parameters.

    Owns the entire marin-to-package mapping: checkpoint resolution (runs
    here, on the TPU VM, where region-dependent mirror localization must
    happen), package config construction, and the worker cache directory.
    """
    with tempfile.TemporaryDirectory(prefix="joint_decode_") as cache_dir:
        decode_config = PackageConfig(
            model_a=_package_model_config(decoder, decoder_placement, _resolve_model_path(decoder.model_path)),
            model_b=_package_model_config(advisor, advisor_placement, _resolve_model_path(advisor.model_path)),
            sampling=PackageSamplingConfig(
                max_tokens_a=max_tokens,
                max_tokens_b=advisor_max_tokens if advisor_max_tokens is not None else max_tokens,
                top_k_a=top_k_a,
                top_k_b=top_k_b,
                barrier_timeout_s=barrier_timeout_s,
                seed=seed,
                stop=stop,
                max_microbatch_size=max_microbatch_size,
                max_num_batched_tokens=max_num_batched_tokens,
            ),
            cache_dir=cache_dir,
        )
        with joint_decoder(decode_config, select_token=select_token) as jd:
            yield jd


def write_chunk(
    chunk: ChunkSpec,
    *,
    decoder: JointDecoder,
    prompt_ids: list[str],
    prompts: list[str],
    n_samples: int,
) -> None:
    request_indices = range(chunk.chunk_start, chunk.chunk_end)
    chunk_prompt_ids = [prompt_ids[i // n_samples] for i in request_indices]
    chunk_completion_indices = [i % n_samples for i in request_indices]
    chunk_prompts = [prompts[i // n_samples] for i in request_indices]

    outputs = decoder.generate(chunk_prompts, chunk_prompts)

    records = []
    for prompt_id, completion_index, output in zip(
        chunk_prompt_ids,
        chunk_completion_indices,
        outputs,
        strict=True,
    ):
        records.append(
            {
                "id": prompt_id,
                "completion_index": completion_index,
                "completion": {
                    "text": output.text,
                    "metadata": {"finish_reason": output.finish_reason},
                },
            }
        )

    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def run_completion_chunks(
    *,
    output_path: str,
    prompts_path: str,
    decoder: EngineModelParams,
    advisor: EngineModelParams,
    n_samples: int,
    max_tokens: int,
    advisor_max_tokens: int | None = None,
    top_k_a: int,
    top_k_b: int,
    seed: int,
    stop: tuple[str, ...],
    select_token: SelectTokens,
    decoder_placement: EnginePlacement,
    advisor_placement: EnginePlacement,
    max_microbatch_size: int,
    max_num_batched_tokens: int | None,
    barrier_timeout_s: float,
    chunk_size: int,
    aggregate_workers: int,
    algorithm: str,
) -> None:
    """Run all chunks through one package JointDecoder pair and aggregate.

    Chunk layout and record schema match the generation-0 modules exactly,
    so partially-complete steps resume across the engine swap.
    """
    prompt_rows = list(read_prompt_rows(prompts_path))
    prompt_ids = [row["id"] for row in prompt_rows]
    prompts = [row["prompt"] for row in prompt_rows]
    chunks_dir = os.path.join(output_path, "chunks", f"chunk_size={chunk_size}")
    chunks = chunk_specs(chunks_dir, len(prompt_rows), n_samples, chunk_size)

    with open_joint_decoder(
        decoder=decoder,
        advisor=advisor,
        max_tokens=max_tokens,
        advisor_max_tokens=advisor_max_tokens,
        top_k_a=top_k_a,
        top_k_b=top_k_b,
        seed=seed,
        stop=stop,
        select_token=select_token,
        decoder_placement=decoder_placement,
        advisor_placement=advisor_placement,
        max_microbatch_size=max_microbatch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        barrier_timeout_s=barrier_timeout_s,
    ) as jd:
        for chunk in chunks:
            if fsspec_exists(chunk.success_path):
                logger.info("chunk %d already done; skipping", chunk.chunk_id)
                continue
            write_chunk(
                chunk,
                decoder=jd,
                prompt_ids=prompt_ids,
                prompts=prompts,
                n_samples=n_samples,
            )
            with fsspec.open(chunk.success_path, "wt") as f:
                f.write("ok\n")

    path = completions_file(output_path)
    aggregate_pipeline = (
        Dataset.from_files(os.path.join(chunks_dir, "chunk-*.jsonl.gz"))
        .load_jsonl()
        .group_by(
            key=lambda record: record["id"],
            reducer=lambda prompt_id, items: {
                "id": prompt_id,
                "completions": [item["completion"] for item in items],
                "metadata": {
                    "completion_algorithm": algorithm,
                    "decoder_model_path": decoder.model_path,
                    "advisor_model_path": advisor.model_path,
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="joint-decode-completions-aggregate",
        max_workers=aggregate_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(aggregate_pipeline)
    logger.info("Wrote %s completion rows to %s", algorithm, path)
