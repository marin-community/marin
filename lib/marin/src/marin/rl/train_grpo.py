# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a Levanter policy on an ordered stream of completed GRPO batches."""

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import partial

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import levanter.config
import numpy as np
from haliax.partitioning import named_jit, round_axis_for_partitioning
from levanter.checkpoint import is_checkpoint_path, save_checkpoint
from levanter.grpo import GrpoConfig, KlGradient, grpo_advantages, grpo_objective_weights
from levanter.grpo_model import GrpoExample, grpo_model_loss, response_logprobs
from levanter.main.model_init import load_model_from_source, prepare_model_init_context
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.trainer import Trainer, TrainerConfig, initialize
from levanter.utils.types import FilterTree
from marin.rl.grpo_artifact import GoldenRollout, read_golden_rollout
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class OfflineGrpoConfig:
    captures: list[str]
    initial_model: str
    tokenizer: str
    hf_save_path: str
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Qwen3Config)
    optimizer: AdamConfig = field(
        default_factory=lambda: AdamConfig(
            learning_rate=1e-6,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup=0,
            lr_schedule="constant",
            default_weight_decay_mask=False,
        )
    )
    grpo: GrpoConfig = field(default_factory=lambda: GrpoConfig(0.2, 0.2, 0.0, KlGradient.DETACHED))
    normalize_by_std: bool = True
    vocab_block_size: int = 1024
    use_hf_model_config: bool = True
    stop_after: int | None = None
    """Stop at this absolute learner step, preserving the full schedule for resumption."""

    def trainable_filter(self, _model: LmHeadModel) -> FilterTree:
        """Return an Equinox filter tree; model adapters may select individual leaves."""
        return True


def prepare_grpo_example(rollout: GoldenRollout, *, normalize_by_std: bool) -> GrpoExample:
    """Preserve complete groups and objective partitions before execution slicing."""
    rollout.validate()
    batch_size, response_width = rollout.old_logprobs.shape
    sequence_width = rollout.sequences.shape[1]
    if response_width >= sequence_width:
        raise ValueError("Each response needs a preceding prompt token")
    if not np.array_equal(rollout.loss_mask, rollout.response_mask):
        raise ValueError("Offline GRPO currently requires binary response-only loss masks")
    attention = rollout.attention_mask
    if np.any((attention != 0) & (attention != 1)):
        raise ValueError("attention_mask must be binary")
    if np.any(rollout.response_mask > attention[:, -response_width:]):
        raise ValueError("Response tokens cannot occupy padding positions")
    if np.any(rollout.response_mask > attention[:, -response_width - 1 : -1]):
        raise ValueError("Each response token must have an attended predecessor")
    Batch, Position, Response = (
        hax.Axis("batch", batch_size),
        hax.Axis("position", sequence_width),
        hax.Axis("response", response_width),
    )
    _, groups = np.unique(rollout.group_ids, return_inverse=True)
    _, partitions = np.unique(rollout.objective_partition_ids, return_inverse=True)
    mask = hax.named(jnp.asarray(rollout.response_mask), (Batch, Response))
    advantages = grpo_advantages(
        hax.named(jnp.asarray(rollout.rewards), (Batch, Response)),
        mask,
        hax.named(jnp.asarray(groups), Batch),
        Batch=Batch,
        Position=Response,
        num_groups=int(groups.max()) + 1,
        normalize_by_std=normalize_by_std,
    )
    policy_weights, kl_weights = grpo_objective_weights(
        mask,
        hax.named(jnp.asarray(partitions), Batch),
        Batch=Batch,
        Position=Response,
        num_partitions=int(partitions.max()) + 1,
    )
    return GrpoExample(
        tokens=hax.named(jnp.asarray(rollout.sequences), (Batch, Position)),
        attention_mask=hax.named(jnp.asarray(attention), (Batch, Position)),
        position_ids=hax.named(jnp.asarray(np.maximum(attention.cumsum(-1) - 1, 0), dtype=jnp.int32), (Batch, Position)),
        response_mask=mask,
        advantages=advantages,
        policy_weights=policy_weights,
        kl_weights=kl_weights,
        old_logprobs=hax.zeros((Batch, Response), dtype=jnp.float32),
        reference_logprobs=(
            None
            if rollout.reference_logprobs is None
            else hax.named(jnp.asarray(rollout.reference_logprobs), (Batch, Response))
        ),
    )


@named_jit
def _score_microbatch(model, microbatch, *, mp, block_size):
    return response_logprobs(mp.cast_to_compute(model), microbatch, block_size=block_size)


def _slice_scoring_batch(value, *, start: int, size: int):
    if not isinstance(value, hax.NamedArray):
        return value
    Batch = value.resolve_axis("batch")
    array = jax.lax.slice_in_dim(value.array, start, start + size, axis=value.axis_indices(Batch))
    return hax.named(array, tuple(axis.resize(size) if axis == Batch else axis for axis in value.axes))


def score_grpo_batch(trainer: Trainer, model: LmHeadModel, batch: GrpoExample, *, block_size: int) -> GrpoExample:
    """Recompute raw old-policy scores at the same execution shape as training."""
    Batch = batch.tokens.resolve_axis("batch")
    microbatch_size = trainer.config.microbatch_size or Batch.size
    if Batch.size % microbatch_size:
        raise ValueError("Complete rollout batch must divide evenly into execution microbatches")

    scores = []
    for start in range(0, Batch.size, microbatch_size):
        microbatch = jax.tree.map(
            partial(_slice_scoring_batch, start=start, size=microbatch_size),
            batch,
            is_leaf=lambda x: isinstance(x, hax.NamedArray),
        )
        with hax.axis_mapping(trainer.compute_axis_mapping):
            scores.append(_score_microbatch(model, microbatch, mp=trainer.mp, block_size=block_size))
    old_logprobs = hax.concatenate(Batch, scores)
    return eqx.tree_at(lambda x: x.old_logprobs, batch, old_logprobs)


def _capture_identity(uri: str) -> dict[str, str]:
    digest = hashlib.sha256()
    with StoragePath(uri).open("rb") as source:
        for chunk in iter(partial(source.read, 1024 * 1024), b""):
            digest.update(chunk)
    return {"uri": uri, "sha256": digest.hexdigest()}


def main(config: OfflineGrpoConfig):
    if config.trainer.initialize_from or config.trainer.load_checkpoint_path:
        raise ValueError("Resume offline runs using the same checkpointer base path and run id")
    if config.trainer.batch_axis_name != "batch":
        raise ValueError("Offline GRPO uses the batch axis named 'batch'")
    if config.trainer.num_train_steps != len(config.captures) or not config.captures:
        raise ValueError("num_train_steps must equal the number of ordered capture batches")
    if config.stop_after is not None and not 0 < config.stop_after <= len(config.captures):
        raise ValueError("stop_after must select a step in the ordered capture stream")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    context = prepare_model_init_context(
        config.model,
        tokenizer=tokenizer,
        initialize_from_hf=config.initial_model,
        use_hf_model_config=config.use_hf_model_config,
    )
    config = dataclasses.replace(config, model=context.model)
    initialize(config)
    microbatch_size = config.trainer.microbatch_size or config.trainer.TrainBatch.size
    accumulation_steps = config.trainer.TrainBatch.size // microbatch_size
    loss = partial(
        grpo_model_loss, config=config.grpo, accumulation_steps=accumulation_steps, block_size=config.vocab_block_size
    )
    with Trainer(
        config.trainer, config.optimizer.build(config.trainer.num_train_steps), loss, add_default_hooks=False
    ) as trainer:
        contract = {
            "captures": [_capture_identity(uri) for uri in config.captures],
            "initial_model": config.initial_model,
            "tokenizer": config.tokenizer,
            "optimizer": draccus.encode(config.optimizer),
            "grpo": draccus.encode(config.grpo),
            "normalize_by_std": config.normalize_by_std,
            "num_train_steps": config.trainer.num_train_steps,
            "model": draccus.encode(config.model),
            "mp": str(config.trainer.mp),
            "seed": config.trainer.seed,
            "train_batch_size": config.trainer.train_batch_size,
            "microbatch_size": microbatch_size,
            "vocab_block_size": config.vocab_block_size,
        }
        contract_path = StoragePath(trainer.checkpoint_path) / "offline-grpo.json"
        if contract_path.exists():
            with contract_path.open("r") as source:
                if json.load(source) != contract:
                    raise ValueError("Offline continuation requires the same capture stream and training recipe")
        elif is_checkpoint_path(trainer.checkpoint_path):
            raise ValueError("Existing checkpoints need their offline capture-stream contract")
        elif jax.process_index() == 0:
            contract_path.parent.mkdirs()
            with contract_path.open("wt") as output:
                json.dump(contract, output, indent=2)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", len(tokenizer)), trainer.parameter_axis_mapping)
        model_key, training_key = jax.random.split(jax.random.PRNGKey(config.trainer.seed))
        state = trainer.initial_state(
            training_key,
            is_trainable=config.trainable_filter(eqx.filter_eval_shape(context.model.build, Vocab, key=model_key)),
            model_init=partial(
                load_model_from_source,
                context=context,
                Vocab=Vocab,
                model_key=model_key,
                parameter_axis_mapping=trainer.parameter_axis_mapping,
                compute_dtype=trainer.mp.param_dtype,
                cast_to_param=trainer.mp.cast_to_param,
                hf_ref=config.initial_model,
            ),
        )
        end_step = config.stop_after or config.trainer.num_train_steps
        while int(state.step) < end_step:
            uri = config.captures[int(state.step)]
            rollout, manifest = read_golden_rollout(uri)
            if manifest["tokenizer"] != config.tokenizer:
                raise ValueError("Capture tokenizer does not match the policy tokenizer")
            if rollout.sequences.min() < 0 or rollout.sequences.max() >= len(tokenizer):
                raise ValueError("Capture contains token IDs outside the policy vocabulary")
            batch = prepare_grpo_example(rollout, normalize_by_std=config.normalize_by_std)
            if batch.tokens.axis_size("batch") != trainer.TrainBatch.size:
                raise ValueError("Each capture must contain exactly one full trainer batch")
            if config.grpo.kl_loss_coef and batch.reference_logprobs is None:
                raise ValueError("KL requires fixed reference logprobs in the capture")
            batch = hax.shard(batch, trainer.compute_axis_mapping)
            batch = score_grpo_batch(trainer, state.model, batch, block_size=config.vocab_block_size)
            info = trainer.train_step(state, batch)
            state = info.state
            save_checkpoint(
                state,
                int(state.step),
                str(StoragePath(trainer.checkpoint_path) / f"step-{int(state.step)}"),
                is_temporary=False,
            )
            logger.info("Completed offline GRPO step %d from %s: loss=%g", int(state.step), uri, info.loss)
        if int(state.step) == config.trainer.num_train_steps:
            assert context.converter is not None
            context.converter.save_pretrained(state.model, config.hf_save_path, upload_to_hf=False)
    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
