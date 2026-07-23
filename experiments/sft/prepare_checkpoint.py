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

The edit only touches the ``embed_tokens`` and ``lm_head`` rows, so the step streams the base
checkpoint from the Hub straight to the output prefix: it rewrites only the safetensors shard that
holds those tensors (in memory) and copies every other file byte-for-byte without staging the whole
model on local disk. The prepared checkpoint is the base checkpoint with those rows reseeded and
the tokenizer renamed, nothing else — so the untouched weights are provably identical to the base.

:func:`prepare_checkpoint_step` expresses this as an ``ArtifactStep`` that emits a prepared HF
checkpoint + tokenizer directory. A config can depend on it (via ``PreparedModel``) to build its
inputs from a clean prefix, or point ``model_ref`` at a staged copy of its output; ``override_path``
adopts an already-staged prepared checkpoint instead of regenerating. The reserved-slot map is a
parameter, so the step is not tied to one model; ``configs/delphi_1e22.py`` is the first example.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from fray.types import ResourceConfig
from fsspec import AbstractFileSystem
from huggingface_hub import HfFileSystem, list_repo_files, snapshot_download
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from rigging.filesystem import StoragePath, prefix_join
from safetensors.numpy import load, save
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
# Files that the renamed tokenizer replaces; skipped when copying the base checkpoint through.
_TOKENIZER_PATTERNS = (
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "*.model",
    "vocab.json",
    "merges.txt",
)
_COPY_CHUNK = 32 * 1024 * 1024

# Only the one embedding shard is held in memory (Delphi's embed + lm_head share shard 1 of 8,
# ~5 GB); everything else streams Hub -> output. No accelerator — a few-row edit does not need one.
DEFAULT_PREPARE_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="32g", disk="20g")


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


def _reinit_shard_bytes(data: bytes, ids: Sequence[int], seed: int) -> bytes:
    """Reinitialize rows ``ids`` of the embedding/LM-head tensors in one safetensors shard's bytes.

    Every other tensor in the shard is round-tripped unchanged. Each embedding tensor draws from an
    independent, name-keyed PRNG stream so the result does not depend on tensor iteration order.
    """
    header_len = int.from_bytes(data[:8], "little")
    metadata = json.loads(data[8 : 8 + header_len]).get("__metadata__")
    tensors = load(data)
    for name in _EMBEDDING_TENSORS:
        if name in tensors:
            rng = np.random.default_rng([seed, _EMBEDDING_TENSORS.index(name)])
            tensors[name] = _reinit_rows(tensors[name], ids, rng)
    return save(tensors, metadata=metadata)


def _stream_copy(src_fs: AbstractFileSystem, src: str, dst: str) -> None:
    """Stream a Hub file to the output prefix in chunks, without staging it whole on local disk."""
    with src_fs.open(src, "rb") as fin, StoragePath(dst).open("wb") as fout:
        shutil.copyfileobj(fin, fout, _COPY_CHUNK)


def _prepare_tokenizer(
    source_model: str, revision: str | None, renames: Mapping[int, str], output_path: str
) -> set[str]:
    """Download just the tokenizer files, rename the reserved slots, upload; return the files written.

    ``save_pretrained`` regenerates only the files it owns (e.g. ``tokenizer.json`` /
    ``tokenizer_config.json``); the returned set is what the base-checkpoint copy must then skip so
    it does not overwrite the renamed files. Any other tokenizer file the base ships (e.g.
    ``special_tokens_map.json``, which carries bos/eos, unaffected by the rename) is copied through.
    """
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as prepared_dir:
        snapshot_download(source_model, revision=revision, local_dir=raw_dir, allow_patterns=list(_TOKENIZER_PATTERNS))
        tokenizer = AutoTokenizer.from_pretrained(raw_dir)
        inject_special_tokens(tokenizer, dict(renames)).save_pretrained(prepared_dir)
        written = [name for name in os.listdir(prepared_dir) if not name.startswith(".")]
        for name in written:
            StoragePath(prefix_join(output_path, name)).upload_from(os.path.join(prepared_dir, name))
        return set(written)


def _embedding_shards(hf_fs: AbstractFileSystem, repo_at: str) -> set[str]:
    """The shard filename(s) holding the embedding/LM-head tensors, from the safetensors index."""
    index = f"{repo_at}/{_SAFETENSORS_INDEX}"
    weight_map = json.loads(hf_fs.cat_file(index))["weight_map"] if hf_fs.exists(index) else {}
    shards = {weight_map.get(name, _SINGLE_SAFETENSORS) for name in _EMBEDDING_TENSORS}
    return shards


def run_prepare_checkpoint(config: PrepareCheckpointConfig) -> None:
    """Rename the reserved slots, reinit their embedding + LM-head rows, and publish the result.

    Runs on a worker: it reads the base checkpoint from the Hub, so it needs ``HF_TOKEN`` in the
    environment for gated repos (propagated the same way as the training job).
    """
    ids = sorted(config.token_renames)
    revision = config.source_revision or None
    repo_at = f"{config.source_model}@{revision}" if revision else config.source_model
    hf_fs = HfFileSystem()

    target_shards = _embedding_shards(hf_fs, repo_at)

    # (1) Rename the reserved slots so the canonical strings are single ids.
    renamed = _prepare_tokenizer(config.source_model, revision, config.token_renames, config.output_path)

    # (2) Stream every other file to the output; rewrite only the embedding shard, in memory.
    for name in list_repo_files(config.source_model, revision=revision):
        if name in renamed:  # already uploaded as the renamed tokenizer
            continue
        src, dst = f"{repo_at}/{name}", prefix_join(config.output_path, name)
        if name in target_shards:
            StoragePath(dst).write_bytes(_reinit_shard_bytes(hf_fs.cat_file(src), ids, config.seed))
        else:
            _stream_copy(hf_fs, src, dst)


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
