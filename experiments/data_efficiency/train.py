# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Produces training step given hyper-parameters
"""

import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.defaults import _prepare_data_config
from experiments.evals.task_configs import convert_to_levanter_task_config
from experiments.data_efficiency.models import model_dict
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.log_probs import default_lm_log_probs, default_ensemble_log_probs
from marin.execution.executor import ExecutorStep, this_output_path, output_path_of
from marin.processing.tokenize.data_configs import (
    LMMixtureDatasetConfig,
    add_validation_sets_to_mixture,
    lm_mixture_data_config,
)
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from fray.cluster import ResourceConfig

from experiments.data_efficiency.data import data_dict


@dataclass
class DataEfficiencyConfig:
    ### Data distribution
    data_name: str
    epochs: float
    train_seed: int = 0
    data_seed: int = 42

    val_name: str | None = None
    teacher_data_name: str | None = None
    teacher_data_weight: float = 0.0

    ### Trainer config
    base_train_steps: int = 1024
    train_batch_size: int = 1024
    train_seq_len: int | None = None
    steps_per_eval: int | None = None
    wandb_project_name: str = "suhas-data-efficiency"
    wandb_additional_tags: list[str] = field(default_factory=list)
    steps_to_export_list: list[int] = field(default_factory=list)
    nametag: str = ""

    ### Hardware
    tpu_type: str = "v4-128"
    slice_count: int = 1
    per_device_parallelism: int = -1

    ### Optimizer config
    lr_schedule: str = "cosine"
    lr: float = 3e-3
    lr_cooldown_duration: float | None = None
    weight_decay: float = 0.0
    min_lr_ratio: float = 0.0

    ### Model config
    model_name: str = "150m4k"
    initialize_from_checkpoint_path: str | None = None
    initialize_from_hf: str | None = None

    ### Eval config
    eval_harness_tasks: list[EvalTaskConfig] | None = None
    eval_harness_steps: int | None = None
    effective_train_seq_len: int = field(init=False)

    def __post_init__(self):
        assert self.lr_cooldown_duration is None, "Cooldown duration is not supported for data efficiency experiments"

        if self.teacher_data_name is not None:
            assert self.teacher_data_weight > 0.0, "Teacher data weight must be greater than 0.0"

        if self.teacher_data_weight > 0.0:
            assert (
                self.teacher_data_name is not None
            ), "Teacher data name must be specified if teacher data weight is greater than 0.0"

        self.model_config = model_dict[self.model_name]
        self.total_train_steps = int(self.base_train_steps * self.epochs / (1.0 - self.teacher_data_weight))
        self.effective_train_seq_len = (
            self.model_config.max_seq_len if self.train_seq_len is None else self.train_seq_len
        )
        if self.effective_train_seq_len > self.model_config.max_seq_len:
            raise ValueError(
                f"train_seq_len {self.effective_train_seq_len} exceeds model max_seq_len {self.model_config.max_seq_len}."
            )

        assert (
            self.initialize_from_checkpoint_path is None or self.initialize_from_hf is None
        ), "Cannot specify both initialize_from_checkpoint_path and initialize_from_hf"

        if self.steps_per_eval is None:
            print(f"steps_per_eval not specified, defaulting to {self.total_train_steps // 20}")
            self.steps_per_eval = self.total_train_steps // 20

        self.steps_per_eval = min(self.steps_per_eval, self.total_train_steps // 2)
 
        if self.eval_harness_steps is None and self.eval_harness_tasks is not None:
            print(f"eval_harness_steps not specified, defaulting to {self.total_train_steps // 4}")
            self.eval_harness_steps = self.total_train_steps // 4

        self.data = data_dict[self.data_name]
        self.one_epoch_tokens = self.effective_train_seq_len * self.base_train_steps * self.train_batch_size
        self.total_tokens = self.one_epoch_tokens * self.epochs

    def build_name(self) -> str:
        data_str = f"{self.format_num_tokens()}x{self.epochs}-{self.data_name}"
        if self.teacher_data_name is not None and self.teacher_data_weight > 0.0:
            data_str += f"+{self.teacher_data_name}^{self.teacher_data_weight}"
        name = f"{self.model_name}-{data_str}-{self.format_lr_schedule()}{self.nametag}"
        assert len(name) <= 64, f"Name is too long with length {len(name)}: {name}"
        return name

    def build_data_config(self) -> LMMixtureDatasetConfig:
        components = {self.data_name: self.data}
        weights = {self.data_name: 1.0}
        num_validation_sequences = {self.data_name: 1024}
        validation_only_components = {}

        if self.val_name is not None:
            if "," in self.val_name:
                val_names = [v.strip() for v in self.val_name.split(",") if v.strip()]
            else:
                val_names = [self.val_name.strip()]

            for val_name in val_names:
                validation_only_components[val_name] = data_dict[val_name]

            # If we have explicit validation-only components, don't request in-mixture validation sequences.
            num_validation_sequences = {}

        max_train_batches = {self.data_name: self.base_train_steps}

        if self.teacher_data_name is not None and self.teacher_data_weight > 0.0:
            components[self.teacher_data_name] = data_dict[self.teacher_data_name]
            weights[self.teacher_data_name] = self.teacher_data_weight
            weights[self.data_name] = 1.0 - self.teacher_data_weight
            if not validation_only_components:  
                num_validation_sequences[self.teacher_data_name] = 1024

        data_config = lm_mixture_data_config(
            components=components,
            weights=weights,
            max_train_batches=max_train_batches,
            num_validation_sequences=num_validation_sequences,
        )

        if validation_only_components:
            data_config = add_validation_sets_to_mixture(data_config, validation_only_components)

        return _prepare_data_config(data_config, use_default_validation=True)

    def format_num_tokens(self) -> str:
        """Format total number of tokens in B/M/K notation."""
        tokens = self.one_epoch_tokens
        if tokens >= 1_000_000_000:
            return f"{tokens/1_000_000_000:.1f}B"
        elif tokens >= 10_000_000:
            return f"{int(tokens/1_000_000)}M"
        elif tokens >= 1_000_000:
            return f"{tokens/1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens/1_000:.1f}K"
        return f"{tokens}"

    def format_sci(self, val):
        return f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e+")

    def format_lr_schedule(self) -> str:
        schedule_short_name = {
            "cosine": "cos",
            "linear": "wsd",
        }

        # cooldown_str = f"{self.lr_cooldown_duration:.2f}" if self.lr_cooldown_duration is not None else "na"
        assert self.lr >= 1e-5
        lr_str = f"{self.lr:.4f}" if self.lr >= 1e-4 else f"{self.lr:.5f}"
        return f"{schedule_short_name[self.lr_schedule]}-lr{lr_str}-wd{self.weight_decay:.2f}"

    def build_trainer_config(self) -> TrainerConfig:
        return TrainerConfig(
            seed=self.train_seed,
            tracker=WandbConfig(
                project=self.wandb_project_name,
                tags=["data-efficiency", *self.wandb_additional_tags],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=self.train_batch_size,
            num_train_steps=self.total_train_steps,
            steps_per_eval=self.steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[
                    dict(every=step_to_export, until=step_to_export + 1) for step_to_export in self.steps_to_export_list
                ],
            ),
            replica_dcn_axis_size=-1,
            per_device_parallelism=self.per_device_parallelism,
            per_device_eval_parallelism=self.per_device_parallelism,
        )

    def build_optimizer_config(self) -> AdamConfig:
        return AdamConfig(
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            lr_schedule=self.lr_schedule,
            decay=self.lr_cooldown_duration,
            min_lr_ratio=self.min_lr_ratio,
        )

    def build_eval_harness_config(self) -> LmEvalHarnessConfig:
        if self.eval_harness_tasks is None:
            return None
        return LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(self.eval_harness_tasks))

    def build_resource_config(self) -> ResourceConfig:
        return ResourceConfig.with_tpu(self.tpu_type, slice_count=self.slice_count)

    def build_train_lm_config(self) -> TrainLmConfig:
        return TrainLmConfig(
            data=self.build_data_config(),
            trainer=self.build_trainer_config(),
            model=self.model_config,
            train_seq_len=self.effective_train_seq_len,
            optimizer=self.build_optimizer_config(),
            z_loss_weight=None,
            initialize_from_checkpoint_path=self.initialize_from_checkpoint_path,
            initialize_from_hf=self.initialize_from_hf,
            eval_harness_steps=self.eval_harness_steps,
            eval_harness=self.build_eval_harness_config(),
            data_seed=self.data_seed,
        )

    def build_train_lm_on_pod_config(self) -> TrainLmOnPodConfig:
        return TrainLmOnPodConfig(
            train_config=self.build_train_lm_config(),
            resources=self.build_resource_config(),
            output_path=this_output_path(),
        )

    def __hash__(self):
        # Hash based on the initial configuration values
        return hash(self.build_name())

    def __eq__(self, other):
        if not isinstance(other, DataEfficiencyConfig):
            return False
        return hash(self) == hash(other)


def data_efficiency_train_step(data_efficiency_config: DataEfficiencyConfig) -> ExecutorStep:
    train_lm_on_pod_config = data_efficiency_config.build_train_lm_on_pod_config()

    executor_step_name = os.path.join("checkpoints", "data_efficiency", data_efficiency_config.build_name())

    return ExecutorStep(
        name=executor_step_name,
        override_output_path=executor_step_name,
        fn=run_levanter_train_lm,
        description=f"Train a model for "
        f"{data_efficiency_config.epochs} (epochs) * "
        f"{data_efficiency_config.base_train_steps} (base steps) * "
        f"{data_efficiency_config.train_batch_size} (batch_size) * "
        f"{data_efficiency_config.effective_train_seq_len} (train_seq_len) "
        f"= {data_efficiency_config.total_tokens:,} total tokens.",
        config=train_lm_on_pod_config,
        pip_dependency_groups=["tokenize_train"],
    )


def data_efficiency_eval_model(data_efficiency_executor_step: ExecutorStep) -> ExecutorStep:
    data_efficiency_train_lm_on_pod_config = data_efficiency_executor_step.config
    data_efficiency_train_lm_config = data_efficiency_train_lm_on_pod_config.train_config

    return default_lm_log_probs(
        checkpoint=output_path_of(
            data_efficiency_executor_step, f"hf/step-{int(data_efficiency_train_lm_config.trainer.num_train_steps-1)}"
        ),
        model=data_efficiency_train_lm_config.model,
        data=data_efficiency_train_lm_config.data,
        checkpoint_is_hf=True,
    )


def data_efficiency_eval_ensemble(
    data_efficiency_executor_steps: list[ExecutorStep],
    run_prefix: str = "ppl-eval",
    name_prefix: str = "ensemble-",
    key: str | None = None,
) -> ExecutorStep:
    data_efficiency_train_lm_on_pod_config = data_efficiency_executor_steps[0].config
    data_efficiency_train_lm_config = data_efficiency_train_lm_on_pod_config.train_config

    return default_ensemble_log_probs(
        checkpoints=[
            output_path_of(
                data_efficiency_executor_step,
                f"hf/step-{int(data_efficiency_executor_step.config.train_config.trainer.num_train_steps-1)}",
            )
            for data_efficiency_executor_step in data_efficiency_executor_steps
        ],
        model=data_efficiency_train_lm_config.model,
        data=data_efficiency_train_lm_config.data,
        checkpoint_is_hf=True,
        run_prefix=run_prefix,
        name_prefix=name_prefix,
        key=key,
    )
