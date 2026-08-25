# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF->Levanter checkpoint conversion as a marin ``ArtifactStep``.

:func:`hf_to_levanter` converts an HF checkpoint to a native Levanter checkpoint once, as a cacheable
graph node, in place of an inline ``initialize_from_hf`` conversion re-run on every launch. It returns
a handle a training step initializes weights from via ``initialize_model_from_checkpoint_path``.

The conversion runs ``levanter.main.export_hf_to_lm`` (``converter.load_pretrained`` + a checkpoint
save), configured to save the model under a ``model`` subtree and to emit a tokenizer padded to the
model vocab. Two inputs are resolved here rather than left implicit:

- the ``LmConfig`` is read from the pinned HF ``config.json`` at graph-construction time and carried on
  the returned handle, because native init builds the model pytree from a concrete ``LmConfig`` instead
  of re-deriving it from the checkpoint; resolving it now makes it part of the step's identity;
- the weights are converted at a given compute dtype (bf16 by default), the dtype the inline
  ``initialize_from_hf`` path loads them in.
"""
from __future__ import annotations

from dataclasses import dataclass

import draccus
from fray.types import ResourceConfig
from levanter.compat.hf_checkpoints import RepoRef
from levanter.main.export_hf_to_lm import ImportHfConfig
from levanter.main.export_hf_to_lm import main as import_hf_to_levanter
from levanter.models.lm_model import LmConfig

from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from marin.utils import get_directory_friendly_name

# Runtime-arg key for where the conversion job runs (excluded from the fingerprint).
_CONVERT_RESOURCES = "convert_resources"
# The subtree the model is saved under, so it loads with load_checkpoint(..., subpath="model") and is
# layout-compatible with a full TrainerState's model field. The tokenizer + metadata sit at the root.
_MODEL_SUBPATH = "model"
# Compute in bf16 by default, matching marin's compute precision (`p=f32,c=bfloat16`) so the weights
# round-trip through the same dtype as the inline `initialize_from_hf` path.
DEFAULT_COMPUTE_DTYPE = "bfloat16"
# The conversion loads one model into host memory and streams it back out; no accelerator is needed.
# Sized for models up to ~10B in bf16 — pass a larger host for bigger bases (peak host memory exceeds
# the raw bf16 size because of HF shard staging and serialization buffers, so do not size to the model
# bytes exactly).
DEFAULT_CONVERT_RESOURCES = ResourceConfig.with_cpu(cpu=8, ram="64g", disk="64g")


@dataclass(frozen=True)
class HfToLevanterConfig:
    """Identity-bearing inputs for one HF->Levanter conversion.

    ``model_config_json`` is the draccus-encoded ``LmConfig`` resolved from the pinned HF checkpoint at
    graph-construction time; the worker decodes it (keyed by ``model_type``) so the model it loads and
    saves is exactly the architecture the downstream builds its pytree from. ``resources`` and
    ``output_path`` are pulled from the step context at run time and never enter the artifact identity.
    """

    hf_id: str  # HF repo id of the base checkpoint
    hf_revision: str  # commit pin for the base checkpoint (fingerprint stability)
    model_type: str  # Levanter model-registry key, e.g. "qwen3"
    tokenizer: str  # tokenizer repo id/dir the converter binds to
    tokenizer_revision: str  # commit pin for the tokenizer
    compute_dtype: str  # dtype the weights are converted in (matches the run's compute precision)
    model_config_json: dict  # draccus-encoded resolved LmConfig
    output_path: str
    resources: ResourceConfig


def resolve_lm_config(model_type: str, hf_id: str, hf_revision: str) -> LmConfig:
    """Resolve the concrete ``LmConfig`` for ``hf_id@hf_revision`` — the arch native init needs.

    Reads the pinned HF ``config.json`` through the model class's ``HFCheckpointConverter`` (the same
    derivation ``use_hf_model_config`` performs at train start). Called at graph-construction time so the
    architecture is an identity-bearing input, not something reconstructed only on the worker.
    """
    config_cls = LmConfig.get_choice_class(model_type)
    converter = config_cls().hf_checkpoint_converter().replaced(reference_checkpoint=RepoRef(hf_id, hf_revision))
    return converter.default_config


def run_hf_to_levanter(config: HfToLevanterConfig) -> None:
    """Convert the HF checkpoint to a native Levanter model checkpoint and emit a padded tokenizer.

    Runs on a worker; reads the base checkpoint from the Hub, so it needs ``HF_TOKEN`` in the
    environment for gated repos. Saves the model under a ``model`` subtree (weights only, no optimizer
    state) with the padded tokenizer at the output root.
    """
    model_config = draccus.decode(LmConfig.get_choice_class(config.model_type), config.model_config_json)
    import_hf_to_levanter(
        ImportHfConfig(
            hf_checkpoint=RepoRef(config.hf_id, config.hf_revision),
            output_path=config.output_path,
            model=model_config,
            use_hf_model_config=False,  # use the resolved arch, do not re-derive it
            tokenizer=config.tokenizer,
            tokenizer_revision=config.tokenizer_revision,
            dtype=config.compute_dtype,
            resize_vocab_to_match_tokenizer=False,  # keep the model's own (padded) vocab
            subpath=_MODEL_SUBPATH,
            emit_padded_tokenizer=True,
        )
    )


def _convert_job(config: HfToLevanterConfig) -> None:
    """The step's ``run``: dispatch the conversion as its own Fray job."""
    remote(run_hf_to_levanter, resources=config.resources)(config)


@dataclass(frozen=True)
class HfToLevanterCheckpoint:
    """A materialized HF->Levanter conversion: the checkpoint step plus the resolved architecture.

    ``step`` produces the native model checkpoint (under a ``model`` subtree) + padded tokenizer at its
    output directory; ``model`` is the ``LmConfig`` the checkpoint was saved with, so a consumer builds
    the matching pytree without re-deriving it.
    """

    step: ArtifactStep[LevanterCheckpoint]
    model: LmConfig


def hf_to_levanter(
    hf_id: str,
    *,
    model_type: str,
    hf_revision: str,
    version: str,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE,
    resources: ResourceConfig = DEFAULT_CONVERT_RESOURCES,
) -> HfToLevanterCheckpoint:
    """Convert ``hf_id@hf_revision`` to a native Levanter model checkpoint as a lazy ``ArtifactStep``.

    ``tokenizer`` defaults to ``hf_id``; ``tokenizer_revision`` defaults to ``hf_revision``. The returned
    handle carries the resolved ``LmConfig`` (read from the pinned HF ``config.json`` now, so it is
    identity-bearing) and the conversion step.
    """
    tokenizer = tokenizer or hf_id
    tokenizer_revision = tokenizer_revision or hf_revision
    model_config = resolve_lm_config(model_type, hf_id, hf_revision)
    model_config_json = draccus.encode(model_config)
    name = f"checkpoints/hf-to-levanter/{get_directory_friendly_name(hf_id)}-{model_type}"

    def build_config(ctx: StepContext) -> HfToLevanterConfig:
        return HfToLevanterConfig(
            hf_id=hf_id,
            hf_revision=hf_revision,
            model_type=model_type,
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            compute_dtype=compute_dtype,
            model_config_json=model_config_json,
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_CONVERT_RESOURCES),
        )

    step: ArtifactStep[LevanterCheckpoint] = ArtifactStep(
        name=name,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=_convert_job,
        build_config=build_config,
        runtime_args={_CONVERT_RESOURCES: resources},
    )
    return HfToLevanterCheckpoint(step=step, model=model_config)
