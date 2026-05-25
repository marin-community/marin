# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for protein-docs distance-masked training experiments.

Every protein-docs distance-masked training script in this directory shares:

* The protein-docs tokenizer.
* The ``contacts-and-distances-v1-5x`` HuggingFace train/val parquets (and the
  marin tokenize steps that materialize their token caches — sharing the steps
  here means all training runs reuse one cache).
* The distance-bin-only loss mask: zero loss everywhere except at the
  ``<d_value>`` token of each ``<distance>`` statement.
* The TPU resource config pinned to ``us-east5-a`` (co-located with the
  ``marin-us-east5`` checkpoint bucket).

Per-experiment scripts choose the model config, learning rate, output path, and
any additional eval components on top of these shared bits.
"""

import dataclasses
from collections.abc import Sequence
from copy import deepcopy

import jax
import jax.numpy as jnp
from fray import ResourceConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorStep, output_path_of, versioned
from marin.export import convert_checkpoint_to_hf_step
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

from experiments.defaults import default_tokenize, default_train
from experiments.protein.create_protein_tokenizer import create_protein_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# Pinned to a specific revision of protein-docs-tokenizer (2840-vocab).
# HEAD is also 2840-vocab now, but workers on this cluster have a stale
# 2849-vocab tokenizer in their local HF cache from when HEAD briefly was 2849
# on 2026-05-06. The @<sha> suffix forces levanter's patched load_tokenizer to
# key the cache by revision so the stale entry is bypassed.
#
# Known issue: at periodic export checkpoints (steps_per_export trigger), some
# code path inside levanter/marin's export passes this string to
# huggingface_hub directly, which rejects ``repo@sha`` as an invalid repo id.
# That crash recurs every 5K steps until either marin/levanter export is
# patched to strip the @sha, or workers' stale tokenizer cache is purged.
PROTEIN_TOKENIZER = "timodonnell/protein-docs-tokenizer@83f597d88e9b"
DISTANCE_TOKEN_ID: int = create_protein_tokenizer().convert_tokens_to_ids("<distance>")

# A distance statement is 6 tokens: <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>.
# For next-token prediction, loss_weight[i] weights the loss on predicting tokens[i+1].
# We zero loss_weight at positions s..s+3 (where s is the <distance> index) so that only
# the prediction of <d_value> (at loss_weight[s+4]) contributes inside a distance statement.
_NUM_NON_BIN_STATEMENT_POSITIONS = 4

HF_DATASET_BASE = "hf://datasets/timodonnell/protein-docs@main/contacts-and-distances-v1-5x"


def distance_bin_only_loss_weight(tokens: jax.Array) -> jax.Array:
    """Zero the loss at all distance-statement positions except the bin itself."""
    is_distance = tokens == DISTANCE_TOKEN_ID
    mask_zero = is_distance
    for shift in range(1, _NUM_NON_BIN_STATEMENT_POSITIONS):
        mask_zero = mask_zero | jnp.roll(is_distance, shift)
    return jnp.where(mask_zero, 0.0, 1.0).astype(jnp.float32)


# Pin to us-east5-a so the TPU is co-located with the `marin-us-east5`
# checkpoint bucket. The v5p-preemptible pool spans {us-central1-a, us-east5-a};
# without this pin a worker can land in us-central1 and pay cross-region I/O
# latency on every checkpoint write.
PROTEIN_RESOURCES_USE5 = ResourceConfig.with_tpu(
    "v5p-8",
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
    zone="us-east5-a",
)

# Pin the tokenize-step output paths to the EXISTING cache directories so
# that pinning ``PROTEIN_TOKENIZER`` to a specific revision (which changes
# the executor hash for these steps) doesn't force a from-scratch retokenize
# of ~125M sequences. The existing caches were tokenized with the same
# vocab content (legacy 2840) so the bytes are reusable.
_PROTEIN_DOCS_CD_CACHE = "gs://marin-us-east5/tokenized/protein-docs-cd-763252"
_PROTEIN_DOCS_CD_VAL_CACHE = "gs://marin-us-east5/tokenized/protein-docs-cd-val-33837a"

protein_docs_tokenized = dataclasses.replace(
    default_tokenize(
        name="protein-docs-cd",
        dataset=f"{HF_DATASET_BASE}/train/",
        tokenizer=PROTEIN_TOKENIZER,
        format=TextLmDatasetFormat(text_key="document"),
    ),
    override_output_path=_PROTEIN_DOCS_CD_CACHE,
)
protein_docs_val_tokenized = dataclasses.replace(
    default_tokenize(
        name="protein-docs-cd-val",
        dataset=f"{HF_DATASET_BASE}/val/",
        tokenizer=PROTEIN_TOKENIZER,
        format=TextLmDatasetFormat(text_key="document"),
        is_validation=True,
    ),
    override_output_path=_PROTEIN_DOCS_CD_VAL_CACHE,
)


def distance_masked_components() -> dict[str, DatasetComponent]:
    """Train+val DatasetComponents with the distance-bin-only loss mask applied.

    Suitable for splatting into ``LmDataConfig.components``. Use ``pack=True``
    to avoid concat-and-split, which would create partial documents — protein
    docs are nonsensical without their header.
    """
    train_component = dataclasses.replace(
        step_to_lm_mixture_component(protein_docs_tokenized, include_raw_paths=True),
        pack=True,
        loss_weight_fn=distance_bin_only_loss_weight,
    )
    val_component = dataclasses.replace(
        step_to_lm_mixture_component(protein_docs_val_tokenized, include_raw_paths=True),
        pack=True,
        loss_weight_fn=distance_bin_only_loss_weight,
    )
    return {"protein-docs-cd": train_component, "protein-docs-cd-val": val_component}


def build_distance_masked_train_step(
    *,
    name: str,
    model_config: LlamaConfig,
    learning_rate: float,
    extra_tags: Sequence[str] = (),
    num_train_steps: int = 50_000,
    train_batch_size: int = 128,
    train_seq_len: int = 8192,
    weight_decay: float = 0.01,
    warmup: float = 0.1,
    steps_per_eval: int = 500,
    steps_per_export: int = 5000,
    resources: ResourceConfig = PROTEIN_RESOURCES_USE5,
    override_output_path: str | None = None,
) -> ExecutorStep:
    """Build a distance-masked training step with the standard recipe.

    The recipe matches ``continue_train_protein_1b_distance_masked.py``: TPU
    pinned to ``us-east5-a``, no in-training distogram benchmark, no Paloma
    validation, no eval-harness tasks. Only ``learning_rate`` is wrapped in
    ``versioned()``, so other knobs can be tuned without busting the cache —
    bump them via versioned() in the caller if you need a fresh run.

    Args:
        name: Used to derive the output path. Passed to ``default_train``.
        model_config: ``LlamaConfig`` for the model architecture.
        learning_rate: Peak learning rate (wrapped in ``versioned()``).
        extra_tags: Additional W&B tags merged with the standard set
            ``("protein", "contacts-and-distances", "llama", "distance-masked")``.
        override_output_path: When set, pins the output directory (used by
            continuation runs to resume from an existing checkpoint dir).
    """
    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=versioned(learning_rate),
        weight_decay=weight_decay,
        warmup=warmup,
        train_seq_len=train_seq_len,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        env_vars={"WANDB_ENTITY": "timodonnell"},
    )

    protein_docs_data = LmDataConfig(
        components=distance_masked_components(),
        train_weights={"protein-docs-cd": 1.0, "protein-docs-cd-val": 0.0},
        tokenizer=PROTEIN_TOKENIZER,
        cache_dir=None,
        block_cross_document_attention=True,
    )

    return default_train(
        name=name,
        tokenized=protein_docs_data,
        model_config=model_config,
        train_config=train_config,
        tags=["protein", "contacts-and-distances", "llama", "distance-masked", *extra_tags],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="protein-training",
        wandb_name=None,
        override_output_path=override_output_path,
    )


def build_hf_export_step(
    *,
    train_step: ExecutorStep,
    model_config: LlamaConfig,
    checkpoint_step: int,
    name_prefix: str,
    checkpoint_path_override: str | None = None,
) -> ExecutorStep:
    """Build a CPU-only HF export step for a distance-masked training run.

    The export reads the latest available checkpoint inside ``train_step``'s
    ``checkpoints/`` subdirectory (``discover_latest=True``) — ``checkpoint_step``
    is used only to label the output directory (``hf/step-{checkpoint_step}``)
    so multiple checkpoints from the same run can coexist.

    By default, ``checkpoint_path`` is wired up via ``output_path_of(train_step,
    "checkpoints")`` so the export step depends on ``train_step`` and waits for
    it to reach ``SUCCEEDED`` before running. Set ``checkpoint_path_override``
    to a literal gs:// path to bypass that dependency — useful when you want to
    snapshot a checkpoint while training is still in progress.

    Args:
        train_step: The training ExecutorStep to export from.
        model_config: The same ``LlamaConfig`` used to train ``train_step``.
        checkpoint_step: Step number used in the output dir name. Update before
            export if you want to label a different checkpoint.
        name_prefix: Used in the export step's name and tags, e.g.
            ``"protein-contacts-30m-distance-masked"``.
        checkpoint_path_override: When set, use this literal gs:// path instead
            of the dependency-creating ``output_path_of(train_step, ...)``. The
            export step then has no marin-executor dependency on ``train_step``
            and will run immediately, regardless of ``train_step``'s status.
    """
    trainer = train_step.config.train_config.trainer
    if not isinstance(trainer, TrainerConfig):
        raise TypeError(f"Expected TrainerConfig on train_step, got {type(trainer)!r}")

    checkpoint_path: str | object
    if checkpoint_path_override is not None:
        checkpoint_path = checkpoint_path_override
    else:
        checkpoint_path = output_path_of(train_step, "checkpoints")

    # Strip the ``@<revision>`` suffix when passing to the export step. The pin
    # exists to bust workers' stale local tokenizer cache during training; the
    # export container starts fresh so doesn't need it, and various transformers
    # code paths (AutoTokenizer falling back to AutoConfig) reject the @sha form.
    export_tokenizer = PROTEIN_TOKENIZER.split("@", 1)[0]

    return convert_checkpoint_to_hf_step(
        name=f"hf/{name_prefix}-step-{checkpoint_step}",
        checkpoint_path=checkpoint_path,  # pyrefly: ignore
        trainer=deepcopy(trainer),
        model=model_config,
        tokenizer=export_tokenizer,
        use_cpu=True,
        discover_latest=True,
    )
