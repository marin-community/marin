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

import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
import numpy as np
from levanter.checkpoint import CheckpointerConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.defaults import _prepare_data_config
from experiments.evals.task_configs import convert_to_levanter_task_config
from experiments.two_stage.data import data_dict
from experiments.two_stage.models import model_dict
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize.data_configs import LMMixtureDatasetConfig, lm_varying_mixture_data_config
from fray.cluster import ResourceConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


@dataclass
class TwoStageConfig:
    """
    Configuration for two-stage training.

    We are interested in the problem setting where there is
        (1) high-quality rare data for a task of interest (i.e. math, instruction following)
        (2) a large amount of common web data that's only generally helpful for task
    We study how to best order your data to maximize performance on the rare task.

    Our experimental approach:
        - Define two pools of data: common (effectively infinite) and rare (limited)
        - Define two stages of training (akin to pre-training and fine-tuning)
        - Select a data mixture for each stage (via `set_data_schedule_params` parameterization)
    This config defines this search space alongside lr schedule, epochs, model size, etc.
    """

    ### Data distribution (dists defined in data.py)
    rare_data_name: str
    common_data_name: str

    rare_fraction: float = 0.01
    rare_data_epochs: int = 1

    ### Data schedule (defined in set_data_schedule_params)
    # Can specify any 2 of the following 3 since the 3rd is implied.
    # If you specify all three, will ensure they are consistent.
    replay_ratio: float | None = None
    rare_stage2_allocation: float | None = None
    stage2_duration: float | None = None
    data_seed: int = 42

    ### Trainer config
    num_train_steps: int = 1000
    train_batch_size: int | None = 1024
    steps_per_eval: int | None = None
    wandb_project_name: str = "suhas-two-stage"
    wandb_additional_tags: list[str] = field(default_factory=list)
    steps_to_export_list: list[int] = field(default_factory=list)
    nametag: str = ""

    ### Hardware
    tpu_type: str = "v4-128"
    slice_count: int = 1

    ### Optimizer config
    lr_schedule: str = "cosine"
    lr: float = 3e-3
    lr_cooldown_duration: float | None = None
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.0

    ### Model config
    model_name: str = "150m4k"
    initialize_from_checkpoint_path: str | None = None
    initialize_from_hf: str | None = None

    ### Eval config
    eval_harness_tasks: list[EvalTaskConfig] | None = None
    eval_harness_steps: int | None = None

    def __post_init__(self):
        self.model_config = model_dict[self.model_name]

        self.set_data_schedule_params()

        assert (
            self.initialize_from_checkpoint_path is None or self.initialize_from_hf is None
        ), "Cannot specify both initialize_from_checkpoint_path and initialize_from_hf"

        if self.steps_per_eval is None:
            print(f"steps_per_eval not specified, defaulting to {self.num_train_steps // 20}")
            self.steps_per_eval = self.num_train_steps // 20

        self.steps_per_eval = min(self.steps_per_eval, self.num_train_steps // 2)

        if self.eval_harness_steps is None and self.eval_harness_tasks is not None:
            print(f"eval_harness_steps not specified, defaulting to {self.num_train_steps // 4}")
            self.eval_harness_steps = self.num_train_steps // 4

        self.rare_data = data_dict[self.rare_data_name]
        self.common_data = data_dict[self.common_data_name]

    def set_data_schedule_params(self):
        """
        A data schedule represents two stages of training subject to a constraint on rare steps.
        Though there are many equivalent ways to parameterize this, we use the following

        ┌──────────────────────────┬──────────┐
        │                          │ agr/(1-r)│ ← r (common data replay)
        │         remainder        │──────────│
        │                          │ ████████ │
        │──────────────────────────┤ ██ ag ██ │ ← 1-r (rare data)
        │ ████████ (1-a)g ████████ | ████████ │
        └──────────────────────────┴──────────┘
                   Stage 1            Stage 2

        Key:
        █ = rare data, blank = common data

        a = self.rare_stage2_allocation
        g = self.rare_fraction * self.rare_data_epochs = self.rare_fraction_epoched
        r = self.replay_ratio
        d = self.stage2_duration

        total rare data = g
        rare data in stage 1 = g * (1 - a)
        rare data in stage 2 = g * a
        common data in stage 2 = g * a * r / (1 - r)
        total data in stage 2 = d = g * a * / (1 - r)
            implies r = 1 - (g * a * / d)
            implies a = d / (g * (1 - r))
        """
        self.rare_fraction_epoched = self.rare_fraction * self.rare_data_epochs
        assert self.rare_fraction_epoched <= 1.0, "Rare fraction * rare data epochs must be less than or equal to 1.0"
        assert self.replay_ratio != 1.0, "Replay ratio cannot be 1.0"

        if self.stage2_duration is None:
            assert self.replay_ratio is not None and self.rare_stage2_allocation is not None
            self.stage2_duration = self.rare_fraction_epoched * self.rare_stage2_allocation / (1 - self.replay_ratio)
        elif self.replay_ratio is None:
            assert self.stage2_duration is not None and self.rare_stage2_allocation is not None
            self.replay_ratio = 1 - (self.rare_fraction_epoched * self.rare_stage2_allocation / self.stage2_duration)
        elif self.rare_stage2_allocation is None:
            assert self.stage2_duration is not None and self.replay_ratio is not None
            self.rare_stage2_allocation = self.stage2_duration / (self.rare_fraction_epoched * (1 - self.replay_ratio))

        # Check if parameters are consistent
        if not np.isclose(
            self.stage2_duration * (1 - self.replay_ratio), self.rare_fraction_epoched * self.rare_stage2_allocation
        ):
            raise ValueError(
                f"Parameters are inconsistent: "
                f"stage2_duration = {self.stage2_duration}, "
                f"rare_fraction_epoched = {self.rare_fraction_epoched}, "
                f"rare_stage2_allocation = {self.rare_stage2_allocation}, "
                f"replay_ratio = {self.replay_ratio}. "
                f"However, {self.stage2_duration} * (1 - {self.replay_ratio}) = "
                f"{self.stage2_duration * (1 - self.replay_ratio)} != "
                f"{self.rare_fraction_epoched} * {self.rare_stage2_allocation} = "
                f"{self.rare_fraction_epoched * self.rare_stage2_allocation}"
            )

        self.total_tokens = self.num_train_steps * self.train_batch_size * self.model_config.seq_len
        self.rare_batches = int(self.num_train_steps * self.rare_fraction)

        self.stage1_duration = 1.0 - self.stage2_duration

        if self.stage1_duration != 0.0:
            self.rare_weight_stage1 = (
                self.rare_fraction_epoched * (1 - self.rare_stage2_allocation) / self.stage1_duration
            )
        else:
            self.rare_weight_stage1 = 0.0
        self.rare_weight_stage2 = 1 - self.replay_ratio
        self.common_weight_stage1 = 1 - self.rare_weight_stage1
        self.common_weight_stage2 = 1 - self.rare_weight_stage2

        if not np.isclose(
            self.rare_weight_stage1 * self.stage1_duration + self.rare_weight_stage2 * self.stage2_duration,
            self.rare_fraction_epoched,
        ):
            raise ValueError(
                f"Rare weight stage 1 * stage 1 duration + "
                f"rare weight stage 2 * stage 2 duration = "
                f"{self.rare_weight_stage1 * self.stage1_duration + self.rare_weight_stage2 * self.stage2_duration} != "
                f"{self.rare_fraction_epoched}"
            )

        assert (
            0.0 <= self.rare_weight_stage1 <= 1.0
        ), f"Rare weight stage 1 must be between 0.0 and 1.0, but is {self.rare_weight_stage1}"
        assert (
            0.0 <= self.rare_weight_stage2 <= 1.0
        ), f"Rare weight stage 2 must be between 0.0 and 1.0, but is {self.rare_weight_stage2}"

    def build_name(self) -> str:
        rare_data_str = f"{self.rare_data_name}x{self.format_rare_fraction()}x{self.rare_data_epochs}"
        data_str = f"{self.format_num_tokens()}-{rare_data_str}-{self.common_data_name}"
        schedule_str = f"rr{self.replay_ratio:.2f}-rs{self.rare_stage2_allocation:.2f}"
        name = f"{self.model_name}-{data_str}-{schedule_str}-{self.format_lr_schedule()}{self.nametag}"
        assert len(name) <= 64, f"Name is too long with length {len(name)}: {name}"
        return name

    def build_data_config(self) -> LMMixtureDatasetConfig:
        components = {self.rare_data_name: self.rare_data, self.common_data_name: self.common_data}

        transition_idx = int(self.stage1_duration * self.num_train_steps)
        # adjust by default block size
        transition_idx = (transition_idx // 2) * 2  # TODO: fix this
        stage1 = (0, {self.rare_data_name: self.rare_weight_stage1, self.common_data_name: self.common_weight_stage1})
        stage2 = (
            transition_idx,
            {self.rare_data_name: self.rare_weight_stage2, self.common_data_name: self.common_weight_stage2},
        )

        if transition_idx == 0:
            weights_list = [stage2]
        else:
            weights_list = [stage1, stage2]

        max_train_batches = {self.rare_data_name: self.rare_batches}
        num_validation_sequences = {self.rare_data_name: 10240, self.common_data_name: 10240}

        no_rare_data = all(stage[1][self.rare_data_name] == 0.0 for stage in weights_list)
        if no_rare_data:
            for stage in weights_list:
                stage[1].pop(self.rare_data_name)
            max_train_batches.pop(self.rare_data_name)
            num_validation_sequences.pop(self.rare_data_name)

        no_common_data = all(stage[1][self.common_data_name] == 0.0 for stage in weights_list)
        if no_common_data:
            for stage in weights_list:
                stage[1].pop(self.common_data_name)
            num_validation_sequences.pop(self.common_data_name)

        data_config = lm_varying_mixture_data_config(
            components=components,
            weights_list=weights_list,
            permutation_type="linear",
            max_train_batches=max_train_batches,
            num_validation_sequences=num_validation_sequences,
        )

        return _prepare_data_config(data_config, use_default_validation=True)

    def format_num_tokens(self) -> str:
        """Format total number of tokens in B/M/K notation."""
        tokens = self.total_tokens
        if tokens >= 1_000_000_000:
            return f"{tokens / 1_000_000_000:.1f}B"
        elif tokens >= 10_000_000:
            return f"{int(tokens / 1_000_000)}M"
        elif tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        return str(tokens)

    def format_sci(self, val):
        return f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e+")

    def format_lr_schedule(self) -> str:
        schedule_short_name = {
            "cosine": "cos",
            "linear": "wsd",
        }
        lr_str = f"{self.lr:.3f}" if self.lr >= 1e-3 else f"{self.format_sci(self.lr)}"

        # TODO: Remove later
        if self.rare_data_name == "latxa":
            lr_str = f"{self.lr:.1e}"

        cooldown_str = f"{self.lr_cooldown_duration:.2f}" if self.lr_cooldown_duration is not None else "na"
        return f"{schedule_short_name[self.lr_schedule]}-{lr_str}-{cooldown_str}"

    def format_rare_fraction(self) -> str:
        if self.rare_fraction >= 0.01:
            return f"{self.rare_fraction:.2f}"
        else:
            return f"{self.rare_fraction:.3f}"

    def build_trainer_config(self) -> TrainerConfig:
        return TrainerConfig(
            tracker=WandbConfig(
                project=self.wandb_project_name,
                tags=["two-stage", *self.wandb_additional_tags],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=self.train_batch_size,
            num_train_steps=self.num_train_steps,
            steps_per_eval=self.steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[
                    dict(every=step_to_export, until=step_to_export + 1) for step_to_export in self.steps_to_export_list
                ],
            ),
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

    def build_pod_config(self) -> ResourceConfig:
        return ResourceConfig.with_tpu(self.tpu_type, slice_count=self.slice_count)

    def build_train_lm_config(self) -> TrainLmConfig:
        return TrainLmConfig(
            data=self.build_data_config(),
            trainer=self.build_trainer_config(),
            model=self.model_config,
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
            resources=self.build_pod_config(),
            output_path=this_output_path(),
        )

    def __hash__(self):
        # Hash based on the initial configuration values
        return hash(self.build_name())

    def __eq__(self, other):
        if not isinstance(other, TwoStageConfig):
            return False
        return hash(self) == hash(other)


def two_stage_train_step(two_stage_config: TwoStageConfig) -> ExecutorStep:
    train_lm_on_pod_config = two_stage_config.build_train_lm_on_pod_config()

    executor_step_name = os.path.join("checkpoints", "two_stage", two_stage_config.build_name())

    return ExecutorStep(
        name=executor_step_name,
        override_output_path=executor_step_name,
        fn=run_levanter_train_lm,
        description=f"Train a model for "
        f"{two_stage_config.num_train_steps} (steps) * "
        f"{two_stage_config.train_batch_size} (batch_size) * "
        f"{two_stage_config.model_config.seq_len} (seq_len) "
        f"= {two_stage_config.total_tokens:,} tokens.",
        config=train_lm_on_pod_config,
        pip_dependency_groups=["tokenize_train"],
    )
