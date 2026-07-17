# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a base HF checkpoint for chat-SFT as a reproducible marin ``ArtifactStep``.

Some base models reserve unnamed vocabulary slots (Llama-3's ``<|reserved_special_token_N|>``)
that a chat protocol later repurposes as control tokens — for Delphi, the think/tool tokens
``<|start_think|>`` … . Two things have to happen before such a checkpoint can be fine-tuned on
that protocol, and this step makes both reproducible instead of a manual out-of-band prerequisite:

  1. **Rename the reserved slots** so the canonical strings tokenize to a single id each. The raw
     tokenizer has no entry for ``<|start_think|>``, so it fragments into ~6 byte pieces; training
     on that is wrong. :func:`inject_special_tokens` rewrites the slot contents in place (no new
     ids, no vocab growth).
  2. **Reinitialize the corresponding embedding rows.** A reserved slot's embedding (and LM-head
     row) is untrained, so it starts from noise. :func:`_reinit_rows` reseeds those rows from the
     matrix's per-column mean/std (a truncated normal), mirroring
     :func:`levanter.utils.token_init.reinitialize_some_tokens`, so SFT starts from a sensible
     point rather than the reserved slot's noise.

The edit only touches the ``embed_tokens`` and ``lm_head`` rows, so the step rewrites just the
safetensors shard(s) that hold them and copies every other shard byte-for-byte — the prepared
checkpoint is the base checkpoint with those rows reseeded and the tokenizer renamed, nothing else.
That keeps it cheap (one shard in memory, not the whole model) and makes the untouched weights
provably identical to the base.

:func:`prepare_checkpoint_step` expresses this as an ``ArtifactStep`` that emits a prepared HF
checkpoint + tokenizer directory. The SFT spec depends on it, so ``sft_step`` builds the prepared
inputs from a clean prefix rather than assuming a staged directory exists. Pass ``override_path``
to pin an already-staged prepared checkpoint instead of regenerating. The reserved-slot map is a
parameter, so the step is not tied to one model; ``configs/delphi_1e22.py`` is the first example.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fray.types import ResourceConfig
from huggingface_hub import snapshot_download
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from rigging.filesystem import prefix_join, url_to_fs
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
from scipy.stats import truncnorm
from transformers import AutoTokenizer

from experiments.marin_tokenizer import inject_special_tokens

# Runtime-arg key for where the preparation job runs (excluded from the fingerprint).
_PREPARE_RESOURCES = "prepare_resources"

# Input-embedding and (untied) LM-head tensors, the only weights the reinit touches. A tied model
# omits lm_head.weight; only the tensors actually present are reinitialized. These are the standard
# Llama/Qwen HF names — the Delphi checkpoints are Qwen3ForCausalLM.
_EMBEDDING_TENSORS = ("model.embed_tokens.weight", "lm_head.weight")
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SINGLE_SAFETENSORS = "model.safetensors"

# Only one shard is loaded at a time (Delphi's embed + lm_head share shard 1 of 8, ~5 GB); the rest
# are streamed to the output untouched. No accelerator — a few-row edit does not need one.
DEFAULT_PREPARE_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="32g", disk="150g")


@dataclass(frozen=True)
class PrepareCheckpointConfig:
    """Identity-bearing inputs for one checkpoint preparation.

    ``token_renames`` maps a reserved vocabulary id to its canonical single-id string; those ids
    are exactly the embedding rows reinitialized. ``resources`` and ``output_path`` are pulled from
    the step context at run time, so they never enter the artifact's identity.
    """

    source_model: str  # HF repo id or dir of the raw base checkpoint
    source_revision: str  # commit pin for the base checkpoint (fingerprint stability)
    token_renames: Mapping[int, str]  # reserved-slot id -> canonical string
    seed: int  # PRNG seed for the embedding reinitialization
    output_path: str
    resources: ResourceConfig


def _reinit_rows(matrix: np.ndarray, ids: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Reseed rows ``ids`` from the matrix's per-column mean/std (truncated normal in [-3, 3]).

    Mirrors :func:`levanter.utils.token_init.reinitialize_some_tokens`: a reserved slot's untrained
    row starts from the pretrained embedding distribution instead of noise. Statistics are taken
    over the whole matrix (the handful of reserved rows are a negligible fraction).
    """
    mu = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    draw = truncnorm.rvs(-3.0, 3.0, size=(len(ids), matrix.shape[1]), random_state=rng)
    out = matrix.copy()
    out[list(ids)] = (draw * std + mu).astype(matrix.dtype)
    return out


def _reinit_embedding_shards(local_dir: str, ids: Sequence[int], seed: int) -> None:
    """Reinitialize rows ``ids`` of the embedding + LM-head tensors, rewriting only their shard(s).

    Every other shard is left on disk untouched (and later uploaded byte-for-byte). Raises if none
    of :data:`_EMBEDDING_TENSORS` are present, since then the checkpoint layout is unexpected.
    """
    index_path = os.path.join(local_dir, _SAFETENSORS_INDEX)
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {name: _SINGLE_SAFETENSORS for name in _EMBEDDING_TENSORS}

    targets = [name for name in _EMBEDDING_TENSORS if name in weight_map]
    if not targets:
        raise ValueError(f"none of {_EMBEDDING_TENSORS} found in {local_dir}; unexpected checkpoint layout")

    shard_to_tensors: dict[str, list[str]] = {}
    for name in targets:
        shard_to_tensors.setdefault(weight_map[name], []).append(name)

    rng = np.random.default_rng(seed)
    for shard in sorted(shard_to_tensors):
        path = os.path.join(local_dir, shard)
        with safe_open(path, framework="numpy") as handle:
            metadata = handle.metadata()
        tensors = load_file(path)
        for name in shard_to_tensors[shard]:
            tensors[name] = _reinit_rows(tensors[name], ids, rng)
        save_file(tensors, path, metadata=metadata)


def _upload_dir(local_dir: str, output_path: str) -> None:
    """Upload the prepared checkpoint files to ``output_path`` (skips the HF download cache)."""
    fs, _ = url_to_fs(output_path)
    for dirpath, _dirs, files in os.walk(local_dir):
        if ".cache" in Path(dirpath).relative_to(local_dir).parts:
            continue  # huggingface_hub's local-dir metadata, not part of the checkpoint
        for name in files:
            src = os.path.join(dirpath, name)
            fs.put_file(src, prefix_join(output_path, os.path.relpath(src, local_dir)))


def run_prepare_checkpoint(config: PrepareCheckpointConfig) -> None:
    """Rename the reserved slots, reinit their embedding + LM-head rows, and publish the result.

    Runs on a worker: it downloads the base checkpoint, so it needs ``HF_TOKEN`` in the environment
    for gated repos (propagated the same way as the training job).
    """
    ids = sorted(config.token_renames)
    with tempfile.TemporaryDirectory() as local_dir:
        snapshot_download(
            repo_id=config.source_model,
            revision=config.source_revision or None,
            local_dir=local_dir,
        )
        # (1) Rename the reserved slots so the canonical strings are single ids, overwriting the
        # tokenizer files in place. The renamed count is unchanged, so the tokenizer length still
        # matches the checkpoint's embedding rows.
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        inject_special_tokens(tokenizer, dict(config.token_renames)).save_pretrained(local_dir)
        # (2) Reseed the repurposed rows in place, touching only the shard(s) that hold them.
        _reinit_embedding_shards(local_dir, ids, config.seed)
        # (3) Publish; every unmodified shard is uploaded byte-for-byte.
        _upload_dir(local_dir, config.output_path)


def _prepare_job(config: PrepareCheckpointConfig) -> None:
    """The step's ``run``: dispatch the preparation as its own CPU Fray job."""
    remote(run_prepare_checkpoint, resources=config.resources)(config)


def prepare_checkpoint_step(
    *,
    name: str,
    version: str,
    source_model: str,
    source_revision: str,
    token_renames: Mapping[int, str],
    seed: int = 0,
    override_path: str | None = None,
    resources: ResourceConfig = DEFAULT_PREPARE_RESOURCES,
) -> ArtifactStep[Artifact]:
    """The checkpoint preparation as a lazy ``ArtifactStep``.

    Its output directory is a prepared HF checkpoint + tokenizer, consumed by ``sft_step`` as both
    ``initialize_from_hf`` and the tokenizer. ``override_path`` pins an already-staged prepared
    checkpoint (adopted, not recomputed) so a config can reuse a validated artifact instead of
    regenerating it.
    """
    if override_path is not None:
        return ArtifactStep.adopt(name, version, source=override_path, kind=Artifact)

    def build_config(ctx: StepContext) -> PrepareCheckpointConfig:
        return PrepareCheckpointConfig(
            source_model=source_model,
            source_revision=source_revision,
            token_renames=dict(token_renames),
            seed=seed,
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_PREPARE_RESOURCES),
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=_prepare_job,
        build_config=build_config,
        runtime_args={_PREPARE_RESOURCES: resources},
    )
