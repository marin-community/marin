# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
import levanter.callbacks
import levanter.config
import levanter.eval
import levanter.tracker
import haliax.nn as hnn
from levanter import callbacks
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.distillation import (
    DistillationModel,
    DistillationObjective,
    TaidConfig,
    TaidState,
    distillation_loss,
    distillation_trainable_filter,
    projected_hidden_distillation_loss,
    taid_loss_with_state_update,
)
from levanter.distillation_initialization import (
    QwenAxisMapping,
    TeacherInitialization,
    initialize_qwen_from_teacher,
    saliency_qwen_axis_mapping,
)
from levanter.main.model_init import load_model_from_source, prepare_model_init_context
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


@dataclass
class TrainLmDistillationConfig:
    data: LmDataConfig = field(default_factory=LmDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    student_model: LmConfig = field(default_factory=Qwen3Config)
    teacher_model: LmConfig = field(default_factory=Qwen3Config)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    train_seq_len: int | None = None
    objective: DistillationObjective = DistillationObjective.FORWARD_KL
    taid: TaidConfig = field(default_factory=TaidConfig)

    student_initialize_from_hf: bool | str = False
    student_use_hf_model_config: bool = False
    teacher_initialize_from_hf: bool | str = "Qwen/Qwen3-32B"
    teacher_use_hf_model_config: bool = True
    teacher_initialization: TeacherInitialization | None = None
    initialization_axis_mapping: QwenAxisMapping | None = None
    structured_use_saliency: bool = True
    factorization_rank: int = 512
    hf_save_path: Optional[str] = None
    hf_save_steps: int = 10_000

    student_anchor_indices: tuple[int, ...] = (6, 13, 20, 27)
    teacher_anchor_indices: tuple[int, ...] = (15, 31, 47, 63)

    data_seed: Optional[int] = None


def _named_eval_example(
    batch: LmExample | GrugLmExample,
    *,
    EvalBatch: Axis,
    Pos: Axis,
) -> LmExample:
    if isinstance(batch, LmExample):
        return batch
    if batch.tokens.ndim == 1:
        return named_lm_example_from_grug(batch, Pos=Pos.resize(batch.tokens.shape[0]))
    if batch.tokens.ndim == 2:
        return named_lm_example_from_grug(
            batch,
            Pos=Pos.resize(batch.tokens.shape[1]),
            batch_axis=EvalBatch,
        )
    raise ValueError(f"GrugLmExample tokens must be rank-1 or rank-2, got rank={batch.tokens.ndim}")


def _validate_config(config: TrainLmDistillationConfig) -> None:
    if not config.teacher_initialize_from_hf:
        raise ValueError("teacher_initialize_from_hf must identify the frozen teacher checkpoint")
    if config.student_initialize_from_hf and config.teacher_initialization is not None:
        raise ValueError("Specify either student_initialize_from_hf or teacher_initialization, not both")
    if config.objective == DistillationObjective.TAID and config.trainer.microbatch_size is not None:
        raise ValueError("TAID controller updates require microbatch_size=None")


def main(config: TrainLmDistillationConfig) -> None:
    _validate_config(config)
    tokenizer = config.data.the_tokenizer

    student_context = prepare_model_init_context(
        config.student_model,
        tokenizer=tokenizer,
        initialize_from_hf=config.student_initialize_from_hf,
        use_hf_model_config=config.student_use_hf_model_config,
    )
    teacher_context = prepare_model_init_context(
        config.teacher_model,
        tokenizer=tokenizer,
        initialize_from_hf=config.teacher_initialize_from_hf,
        use_hf_model_config=config.teacher_use_hf_model_config,
    )

    levanter.trainer.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def loss_function(model: DistillationModel, example: LmExample, *, key=None):
        if config.objective == DistillationObjective.PROJECTED_HIDDEN:
            return projected_hidden_distillation_loss(
                model,
                example,
                student_anchor_indices=config.student_anchor_indices,
                teacher_anchor_indices=config.teacher_anchor_indices,
                key=key,
            )
        if config.objective == DistillationObjective.TAID:
            if model.taid_state is None:
                raise ValueError("TAID objective requires taid_state")
            loss = distillation_loss(
                model,
                example,
                objective=config.objective,
                taid_state=model.taid_state,
                key=key,
            )
            updated_loss = taid_loss_with_state_update(
                loss,
                model.taid_state,
                config.trainer.num_train_steps,
                config.taid,
            )
            return updated_loss, {
                "distillation_loss": loss,
                "taid_interpolation": model.taid_state.interpolation,
                "taid_loss_momentum": model.taid_state.loss_momentum,
            }
        loss = distillation_loss(model, example, objective=config.objective, key=key)
        return loss, {"distillation_loss": loss}

    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        seed = config.trainer.seed
        data_key, model_key, teacher_key, projector_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 5)
        if config.data_seed is not None:
            data_key = jrandom.PRNGKey(config.data_seed)

        student_config = student_context.model
        teacher_config = teacher_context.model
        train_length = config.train_seq_len if config.train_seq_len is not None else student_config.max_seq_len
        if train_length <= 0:
            raise ValueError(f"train_seq_len must be positive, got {train_length}")
        if train_length > student_config.max_seq_len or train_length > teacher_config.max_seq_len:
            raise ValueError(
                f"train_seq_len={train_length} exceeds student or teacher context length "
                f"({student_config.max_seq_len}, {teacher_config.max_seq_len})"
            )

        parameter_axis_mapping = trainer.parameter_axis_mapping
        compute_axis_mapping = trainer.compute_axis_mapping
        Pos = student_config.max_Pos.resize(train_length)
        checkpoint_vocab_sizes = [
            context.converter.default_hf_config.vocab_size
            for context in (student_context, teacher_context)
            if context.converter is not None
        ]
        vocab_size = max(len(tokenizer), *checkpoint_vocab_sizes)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info("Rounding vocabulary from %d to %d for partitioning", vocab_size, Vocab.size)

        train_dataset = config.data.train_set(Pos, config.trainer.batch_schedule, key=data_key)
        tagged_eval_datasets = config.data.tagged_eval_sets(Pos)

        def model_init() -> DistillationModel:
            hidden_projector = None
            if config.objective == DistillationObjective.PROJECTED_HIDDEN:
                hidden_projector = hnn.Linear.init(
                    In=student_config.Embed,
                    Out=teacher_config.Embed.alias("teacher_embed"),
                    key=projector_key,
                    use_bias=False,
                    out_first=True,
                )
            return DistillationModel(
                student=student_config.build(Vocab, key=model_key),
                teacher=teacher_config.build(Vocab, key=teacher_key),
                hidden_projector=hidden_projector,
                taid_state=TaidState.init(config.taid) if config.objective == DistillationObjective.TAID else None,
            )

        trainable_filter = distillation_trainable_filter(eqx.filter_eval_shape(model_init))
        state = trainer.initial_state(
            training_key,
            model_init=model_init,
            is_trainable=trainable_filter,
        )

        if int(state.step) == 0 and config.student_initialize_from_hf:
            student = load_model_from_source(
                context=student_context,
                Vocab=Vocab,
                model_key=model_key,
                parameter_axis_mapping=parameter_axis_mapping,
                compute_dtype=trainer.mp.compute_dtype,
                cast_to_param=trainer.mp.cast_to_param,
                hf_ref=config.student_initialize_from_hf,
            )
        else:
            student = state.model.student

        teacher = load_model_from_source(
            context=teacher_context,
            Vocab=Vocab,
            model_key=teacher_key,
            parameter_axis_mapping=parameter_axis_mapping,
            compute_dtype=trainer.mp.compute_dtype,
            cast_to_param=trainer.mp.cast_to_compute,
            hf_ref=config.teacher_initialize_from_hf,
        )
        teacher = inference_mode(teacher, True)
        if int(state.step) == 0 and config.teacher_initialization is not None:
            if not isinstance(student, Qwen3LMHeadModel) or not isinstance(teacher, Qwen3LMHeadModel):
                raise TypeError("Teacher-derived initialization currently requires Qwen3 teacher and student models")
            axis_mapping = config.initialization_axis_mapping
            if (
                config.teacher_initialization == TeacherInitialization.STRUCTURED
                and config.structured_use_saliency
                and axis_mapping is None
            ):
                axis_mapping = saliency_qwen_axis_mapping(student, teacher)
            student = initialize_qwen_from_teacher(
                student,
                teacher,
                method=config.teacher_initialization,
                axis_mapping=axis_mapping,
                rank=config.factorization_rank,
                key=projector_key,
            )
        state = dataclasses.replace(
            state,
            model=DistillationModel(
                student=student,
                teacher=teacher,
                hidden_projector=state.model.hidden_projector,
                taid_state=state.model.taid_state,
            ),
        )

        levanter.tracker.log_summary(
            {
                "student_parameter_count": parameter_count(student),
                "teacher_parameter_count": parameter_count(teacher),
            }
        )

        max_eval_examples = config.trainer.max_eval_batches
        if max_eval_examples is not None:
            max_eval_examples *= config.trainer.eval_batch_size

        if tagged_eval_datasets:

            def eval_loss(model: DistillationModel, batch: LmExample | GrugLmExample):
                student_model = inference_mode(model.student, True)
                student_model = trainer.mp.cast_to_compute(student_model)
                example = _named_eval_example(batch, EvalBatch=trainer.EvalBatch, Pos=Pos)
                per_position = student_model.compute_next_token_loss(
                    example,
                    reduction=None,
                    reduction_axis=(),
                ).array
                token_ids = jnp.roll(example.tokens.array, -1, axis=-1)
                return per_position, example.loss_weight.array, token_ids

            eval_callback = levanter.eval.cb_tagged_lm_evaluate(
                trainer.EvalBatch,
                tagged_eval_datasets,
                tokenizer=tokenizer,
                device_mesh=trainer.device_mesh,
                axis_mapping=compute_axis_mapping,
                max_examples_per_dataset=max_eval_examples,
                eval_ema=False,
                checkpoint_path=trainer.checkpoint_path,
                loss_fn=eval_loss,
            )
            trainer.add_hook(eval_callback, every=config.trainer.steps_per_eval)
        else:
            logger.warning("No evaluation datasets provided.")

        student_flops = student_config.flops_per_token(vocab_size, Pos.size)
        teacher_flops = teacher_config.flops_per_token(vocab_size, Pos.size)
        flops_per_example = None
        if student_flops is not None and teacher_flops is not None:
            flops_per_example = (3 * student_flops + teacher_flops) * Pos.size
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example),
            every=1,
        )
        trainer.add_hook(
            callbacks.iris_status_reporter(
                Pos.size,
                trainer.config.batch_schedule,
                trainer.config.num_train_steps,
                flops_per_example,
            ),
            every=10,
        )
        if isinstance(train_dataset, MixtureDataset):
            trainer.add_hook(
                callbacks.mixture_weight_logging_hook(trainer.config.batch_schedule, train_dataset),
                every=1,
            )

        train_loader = trainer.data_loader(train_dataset).iter_from_step(state.step)
        trainer.train(state, train_loader)

    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
