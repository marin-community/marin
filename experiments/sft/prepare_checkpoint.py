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
     row) is untrained, so it starts from noise. :func:`reinitialize_some_tokens` reseeds those
     rows from the pretrained embedding statistics (mean + std matched, per the landed Levanter
     helper) so SFT starts from a sensible point.

:func:`prepare_checkpoint_step` expresses this as an ``ArtifactStep`` that emits a prepared HF
checkpoint + tokenizer directory. The SFT spec depends on it, so ``sft_step`` builds the prepared
inputs from a clean prefix rather than assuming a staged directory exists. Pass ``override_path``
to pin an already-staged prepared checkpoint instead of regenerating.

The preparation is model-agnostic: the reserved-slot map and the Levanter model-registry key
(``model_type``) are parameters. ``configs/delphi_1e22.py`` is the first worked example.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
from fray.types import ResourceConfig
from levanter.compat.hf_checkpoints import RepoRef
from levanter.models.lm_model import LmConfig
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.token_init import reinitialize_some_tokens
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from transformers import AutoTokenizer

from experiments.marin_tokenizer import inject_special_tokens

# Runtime-arg key for where the preparation job runs (excluded from the fingerprint).
_PREPARE_RESOURCES = "prepare_resources"

# Loading + re-emitting a multi-billion-parameter checkpoint on a CPU host: generous RAM/disk for
# the state dict plus the download, no accelerator (a 6-row embedding edit does not need one).
DEFAULT_PREPARE_RESOURCES = ResourceConfig.with_cpu(cpu=8, ram="120g", disk="120g")


@dataclass(frozen=True)
class PrepareCheckpointConfig:
    """Identity-bearing inputs for one checkpoint preparation.

    ``token_renames`` maps a reserved vocabulary id to its canonical single-id string; those
    strings are exactly the rows reinitialized. ``resources`` and ``output_path`` are pulled from
    the step context at run time, so they never enter the artifact's identity.
    """

    source_model: str  # HF repo id or dir of the raw base checkpoint
    source_revision: str  # commit pin for the base checkpoint (fingerprint stability)
    token_renames: Mapping[int, str]  # reserved-slot id -> canonical string
    model_type: str  # Levanter model-registry key selecting the HF converter (e.g. "qwen3")
    seed: int  # PRNG seed for the embedding reinitialization
    output_path: str
    resources: ResourceConfig


def run_prepare_checkpoint(config: PrepareCheckpointConfig) -> None:
    """Rename the reserved slots, reinit their embedding + LM-head rows, and save the result.

    Runs on a worker: it downloads the base checkpoint, so it needs ``HF_TOKEN`` in the
    environment for gated repos (propagated the same way as the training job).
    """
    revision = config.source_revision or None
    ref = RepoRef(config.source_model, revision)

    # (1) Rename the reserved slots so the canonical strings are single ids. The renamed count is
    # unchanged, so the tokenizer length still matches the checkpoint's embedding rows.
    base_tokenizer = AutoTokenizer.from_pretrained(config.source_model, revision=revision)
    tokenizer = inject_special_tokens(base_tokenizer, dict(config.token_renames))
    tokens_to_reinit = list(config.token_renames.values())

    model_config = LmConfig.get_choice_class(config.model_type)()
    # Bind the base checkpoint and the renamed tokenizer; save_pretrained writes this tokenizer.
    converter = model_config.hf_checkpoint_converter().replaced(reference_checkpoint=ref, tokenizer=tokenizer)

    with local_cpu_mesh():
        # config=None re-derives the architecture from the checkpoint (its exact Delphi config),
        # matching the SFT job's use_hf_model_config path.
        model = converter.load_pretrained(model_config.model_type, config=None)
        # (2) Reseed the repurposed rows from the pretrained embedding statistics.
        model = reinitialize_some_tokens(model, tokenizer, tokens_to_reinit, key=jax.random.PRNGKey(config.seed))
        converter.save_pretrained(model, config.output_path, save_tokenizer=True)


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
    model_type: str = "qwen3",
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
            model_type=model_type,
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
