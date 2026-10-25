# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite LR0 trainer lifecycle binding; parity and export audits are mandatory inputs."""

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, replace

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.data.text.examples import GrugLmExample
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig, preserve_hf_runtime_settings
from levanter.models.lm_model import LmExample
from levanter.models.snowball import GrugMoeHfConfig
from levanter.optim.config import AdamConfig
from levanter.tokenizers import MarinTokenizer
from levanter.trainer import Trainer, initialize
from levanter.trainer_state import TrainerState
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_context_distill.runtime_recipe import check_precision_runtime_receipt
from experiments.post_training.identity_context_distill.sft import (
    BASE_MODEL,
    GLOBAL_BATCH_SIZE,
    SEQUENCE_LENGTH,
    check_export_metadata,
    check_restored_progress,
    check_runtime_recipe,
    check_source_metadata_receipt,
    source_export_overrides,
)


def synthetic_lr0_batch(tokenizer: MarinTokenizer) -> LmExample:
    """Render eight explicitly synthetic native chats; never use as training data.

    Uses ChatProcessor's IDs/masks and ChatDataset's one-token causal mask shift.
    Rows are not packed or truncated; padding carries no loss.
    """
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Synthetic precision probe {index}: reply with the word ready."},
                {"role": "assistant", "content": "ready", "reasoning_content": ""},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for index in range(GLOBAL_BATCH_SIZE)
    ]
    processed = ChatLmDatasetFormat(pack=False, slice_strategy="raise", mask_user_turns=True).build_preprocessor(
        tokenizer
    )(rows)
    examples = []
    for row in processed:
        tokens = np.asarray(row["input_ids"], dtype=np.int32)
        mask = np.asarray(row["assistant_masks"], dtype=np.int32)
        if len(tokens) > SEQUENCE_LENGTH or tokens.shape != mask.shape or not np.any(mask):
            raise ValueError("Synthetic chat must fit native context with nonempty matching masks")
        padding = SEQUENCE_LENGTH - len(tokens)
        padded = np.pad(tokens, (0, padding), constant_values=128001)
        shifted = np.roll(np.pad(mask, (0, padding)), -1).astype(np.float32)
        examples.append(GrugLmExample.causal(jnp.asarray(padded), loss_weight=jnp.asarray(shifted)))
    if len(examples) != GLOBAL_BATCH_SIZE:
        raise ValueError("Synthetic ChatProcessor did not return eight rows")
    axes = (hax.Axis("batch", GLOBAL_BATCH_SIZE), hax.Axis("position", SEQUENCE_LENGTH))
    return LmExample(
        hax.named(jnp.stack([row.tokens for row in examples]), axes),
        hax.named(jnp.stack([row.loss_weight for row in examples]), axes),
    )


def adam_gradient_witness(state: TrainerState) -> float:
    """Require every Adam first moment finite and their global maximum nonzero."""
    maximum = 0.0
    for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        if not any(isinstance(key, jax.tree_util.GetAttrKey) and key.name == "mu" for key in path):
            continue
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
            magnitude = float(multihost_utils.process_allgather(jnp.max(jnp.abs(leaf))).max())
            if not math.isfinite(magnitude):
                raise ValueError("Nonfinite Adam gradient witness")
            maximum = max(maximum, magnitude)
    if maximum == 0:
        raise ValueError("No nonzero Adam gradient witness; LR0 backward proof failed")
    return maximum


def gated_zero_lr_step(
    trainer: Trainer,
    state: TrainerState,
    batch: LmExample,
    pre_update_gate: Callable[[Trainer, TrainerState], None],
    master_audit: Callable[[TrainerState, int], None],
) -> tuple[TrainerState, float, float]:
    """Run the raising parity gate before the sole update; do not catch gate failures.

    The caller must construct the optimizer at zero LR. Every Adam first-moment
    leaf must be finite, with a nonzero global maximum. Master audits run at both
    steps; the callback must use the supplied step to write distinct receipts.
    """
    if int(state.step) != 0:
        raise ValueError("LR0 precision probe must start fresh at update zero")
    master_audit(state, 0)
    pre_update_gate(trainer, state)
    info = trainer.train_step(state, batch)
    if int(info.state.step) != 1 or not math.isfinite(info.loss):
        raise ValueError("Expected exactly one finite LR0 update")
    witness = adam_gradient_witness(info.state)
    master_audit(info.state, 1)
    return info.state, info.loss, witness


def _router_values(model) -> dict[str, np.ndarray]:
    tensors = model.to_state_dict()
    expected = {f"model.layers.{layer}.mlp.router.bias" for layer in range(26)}
    found = {name for name in tensors if name.endswith(".mlp.router.bias")}
    if found != expected or len(tensors) != 502:
        raise ValueError("Unexpected source/export tensor keys or router count")
    result = {}
    for name in sorted(expected):
        if tensors[name].dtype != jnp.float32 or tensors[name].shape != (256,):
            raise ValueError(f"Router bias is not native FP32: {name}")
        gathered = np.asarray(multihost_utils.process_allgather(tensors[name], tiled=True)).reshape(-1, 256)
        for replica in gathered:
            np.testing.assert_array_equal(replica, gathered[0])
        result[name] = gathered[0].copy()
    return result


def run_precision_probe(
    config: TrainLmConfig,
    *,
    expected_source_receipt: dict,
    expected_runtime_receipt: dict,
    source_audit: Callable[[str], None],
    master_audit: Callable[[TrainerState, int], None],
    pre_update_gate: Callable[[Trainer, TrainerState], None],
    export_audit: Callable[[str, str], None],
) -> dict:
    """Bind source load, one gated LR0 update, standard HF save, and raising readback audits.

    This is a worker function, not a launcher. Call on every JAX rank. Audit callbacks
    run on process zero and must raise on failure. The parity callback runs on every
    rank and must synchronize failures before returning. The master audit also
    runs collectively at steps zero and one, checks every HF-mapped master leaf
    against losslessly upcast source bytes, and writes step-specific receipts.
    The parity callback owns stored-token-id
    scoring and the registered log-probability and routing gates. No default pass
    implementations are supplied.

    The regional tokenizer produces an explicitly synthetic native-chat batch for
    LR0 only. Source audit verifies regional tokenizer/template,
    headers and pinned keys. Export audit streams all 502 tensors against the source,
    requiring 476 BF16 plus 26 FP32 and exact LR0 values, plus tokenizer/generation
    metadata. These bindings must be reviewed before this function is launchable.
    """
    if config.initialize_from_hf != BASE_MODEL or not isinstance(config.optimizer, AdamConfig):
        raise ValueError("Probe requires pinned regional source and the recipe's Adam optimizer")
    if config.trainer.load_checkpoint_path is not None or config.trainer.initialize_from is not None:
        raise ValueError("Precision probe is fresh source initialization, not continuation")
    if not config.hf_save_path or not config.hf_save_path.startswith("s3://marin-rno2a/"):
        raise ValueError("Probe export must use an explicit regional RNO2a output root")
    if (StoragePath(config.hf_save_path) / "lr0-step-0").exists():
        raise FileExistsError("Probe needs a fresh export path; refusing to reuse a partial export")
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    check_source_metadata_receipt(expected_source_receipt, config_bytes, generation_bytes)
    if expected_source_receipt != expected_runtime_receipt["source_metadata"]:
        raise ValueError("Source and runtime receipts bind different native metadata")
    rounding_path = expected_runtime_receipt["routing_reference"]["path"]
    if not rounding_path.startswith("s3://marin-rno2a/"):
        raise ValueError("Router reference must be the approved regional artifact")
    check_precision_runtime_receipt(
        config,
        expected_runtime_receipt,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(rounding_path).read_bytes(),
    )
    source = json.loads(config_bytes)
    generation = json.loads(generation_bytes)
    probe_config = replace(
        config,
        trainer=replace(config.trainer, load_checkpoint=False, num_train_steps=1),
        optimizer=replace(config.optimizer, learning_rate=0.0, warmup=0, weight_decay=0.0),
    )
    initialize(probe_config)
    started = time.monotonic()

    def loss(model, example, *, key=None):
        return model.compute_next_token_loss(example, key=key, logsumexp_weight=probe_config.z_loss_weight)

    with Trainer(probe_config.trainer, probe_config.optimizer.build(1), loss, add_default_hooks=False) as trainer:
        check_runtime_recipe(probe_config, source_config=source, source_generation_config=generation)
        run_on_export_writer(lambda: source_audit(BASE_MODEL))
        tokenizer = probe_config.data.the_tokenizer
        batch = synthetic_lr0_batch(tokenizer)
        if batch.tokens.axes != (hax.Axis("batch", GLOBAL_BATCH_SIZE), hax.Axis("position", SEQUENCE_LENGTH)):
            raise ValueError("Synthetic probe batch must be native 8 x 8192, unpacked")
        if batch.loss_weight.axes != batch.tokens.axes or float(hax.sum(batch.loss_weight).scalar()) <= 0:
            raise ValueError("Synthetic probe needs a matching, nonempty loss mask")
        converter = probe_config.model.hf_checkpoint_converter().replaced(
            reference_checkpoint=BASE_MODEL, tokenizer=tokenizer
        )
        # The HF metadata is architecture-authoritative; runtime precision is restored AFTER reload.
        reloaded = converter.config_from_hf_config(GrugMoeHfConfig(**source))
        reloaded = preserve_hf_runtime_settings(
            reloaded, probe_config.model, probe_config.trainer.compute_dtype_overrides
        )
        probe_config = replace(probe_config, model=reloaded)
        check_runtime_recipe(probe_config, source_config=source, source_generation_config=generation)
        model = converter.load_pretrained(
            probe_config.model.model_type,
            config=reloaded,
            axis_mapping=trainer.parameter_axis_mapping,
            dtype=None,
        )
        loaded_model_config = asdict(model.config)
        if loaded_model_config != expected_runtime_receipt["runtime_model_config"]:
            raise ValueError("HF loader discarded or changed the approved runtime model config")
        run_on_export_writer(
            lambda: write_export_json(
                StoragePath(probe_config.hf_save_path) / "runtime-recipe-receipt.json",
                {"approved": expected_runtime_receipt, "loaded_model_config": loaded_model_config},
            )
        )
        source_biases = _router_values(model)
        state = trainer.initial_state(
            jax.random.PRNGKey(probe_config.trainer.seed),
            model=model,
            is_trainable=probe_config.adapter.trainable_filter(model),
        )
        del model
        check_restored_progress(probe_config, state, completed_updates=0)
        for name, value in _router_values(state.model).items():
            np.testing.assert_array_equal(value, source_biases[name])
        state, measured_loss, witness = gated_zero_lr_step(trainer, state, batch, pre_update_gate, master_audit)
        check_restored_progress(probe_config, state, completed_updates=1)
        for name, value in _router_values(state.model).items():
            np.testing.assert_array_equal(value, source_biases[name])
        export_path = str(StoragePath(probe_config.hf_save_path) / "lr0-step-0")
        converter = converter.with_config_overrides(source_export_overrides(source))
        save_started = time.monotonic()
        converter.save_pretrained(
            state.model,
            export_path,
            dtype=jnp.bfloat16,
            dtype_overrides={f"model.layers.{layer}.mlp.router.bias": "float32" for layer in range(26)},
            generation_config=generation,
            upload_to_hf=False,
        )
        save_seconds = time.monotonic() - save_started

        def readback():
            exported_bytes = (StoragePath(export_path) / "config.json").read_bytes()
            exported = json.loads(exported_bytes)
            fields = {
                field: {
                    "source_value": value,
                    "export_present": field in exported,
                    "export_value": exported.get(field),
                    "equal": field in exported and exported[field] == value,
                }
                for field, value in source_export_overrides(source).items()
            }
            write_export_json(
                StoragePath(export_path) / "source-config-comparison.json",
                {
                    "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "export_config_sha256": hashlib.sha256(exported_bytes).hexdigest(),
                    "fields": fields,
                },
            )
            check_export_metadata(source, exported)
            export_audit(BASE_MODEL, export_path)

        run_on_export_writer(readback)
        return {
            "completed_updates": 1,
            "learning_rate": 0.0,
            "loss": measured_loss,
            "adam_first_moment_witness": witness,
            "hf_path": export_path,
            "save_seconds": save_seconds,
            "worker_seconds": time.monotonic() - started,
            "compute_dtype_overrides": probe_config.trainer.compute_dtype_overrides,
            "runtime_recipe_receipt": expected_runtime_receipt,
        }
