import os
from datetime import timedelta
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import jmp

from levanter.main.train_lm import TrainLmConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.eval_harness import LmEvalHarnessConfig

from experiments.defaults import _prepare_data_config
from experiments.evals.task_configs import convert_to_levanter_task_config

from marin.execution.executor import ExecutorStep, this_output_path
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.processing.tokenize.data_configs import LMMixtureDatasetConfig
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config
from marin.training.training import PodConfig

from experiments.two_stage.data import data_dict
from experiments.two_stage.models import model_dict

@dataclass
class TwoStageConfig:
    """Configuration for two-stage training."""

    ### Data quantity
    rare_data_name: str
    common_data_name: str

    rare_fraction: float = 0.01
    rare_data_epochs: int = 1

    ### Data schedule
    # Can specify any 2 of the following 3 since the 3rd is implied. If you specify all three, will ensure they are consistent.
    replay_ratio: Optional[float] = None
    rare_stage2_allocation: Optional[float] = None
    stage2_duration: Optional[float] = None
    data_seed: int = 42

    ### Trainer config
    num_train_steps: int = 1000
    train_batch_size: Optional[int] = 1024
    steps_per_eval: Optional[int] = None
    wandb_project_name: str = "suhas-two-stage"
    wandb_additional_tags: list[str] = field(default_factory=list)
    steps_to_export_list: list[int] = field(default_factory=list)
    nametag: str = ""

    ### Hardware
    tpu_type: str = "v4-128"
    node_count: int = 1

    ### Optimizer config
    lr_schedule: str = "cosine"
    lr: float = 3e-3
    lr_cooldown_duration: float = 1.0
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.0

    ### Model config
    model_name: str = "150m4k"
    initialize_from_checkpoint_path: Optional[str] = None
    initialize_from_hf: Optional[str] = None

    ### Eval config
    eval_harness_tasks: Optional[list[EvalTaskConfig]] = None
    eval_harness_steps: Optional[int] = None

    def __post_init__(self):
        self.model_config = model_dict[self.model_name]
        
        self.set_data_schedule_params()

        assert self.initialize_from_checkpoint_path is None or self.initialize_from_hf is None, "Cannot specify both initialize_from_checkpoint_path and initialize_from_hf"

        if self.steps_per_eval is None:
            self.steps_per_eval = self.num_train_steps // 20
        
        self.steps_per_eval = min(self.steps_per_eval, self.num_train_steps)

        if self.eval_harness_steps is None and self.eval_harness_tasks is not None:
            self.eval_harness_steps = self.num_train_steps // 4

        self.rare_data = data_dict[self.rare_data_name]
        self.common_data = data_dict[self.common_data_name]

    def set_data_schedule_params(self):
        """
        ┌──────────────────────────┬──────────┐
        │                          │ αγρ/(1-ρ)│ ← ρ (common data replay)
        │         remainder        │──────────│
        │                          │ ████████ │
        │──────────────────────────┤ ██ αγ ██ │ ← 1-ρ (rare data)
        │ ████████ (1-α)γ ████████ | ████████ │  
        └──────────────────────────┴──────────┘
                   Stage 1            Stage 2

        Key:  
        █ = rare data, blank = common data

        α = self.rare_stage2_allocation
        γ = self.rare_fraction * self.rare_data_epochs = self.rare_fraction_epoched
        ρ = self.replay_ratio
        δ = self.stage2_duration

        total rare data = γ
        rare data in stage 1 = γ * (1 - α)
        rare data in stage 2 = γ * α
        common data in stage 2 = γ * α * ρ / (1 - ρ)
        total data in stage 2 = δ = γ * α * / (1 - ρ)
            implies ρ = 1 - (γ * α * / δ)
            implies α = δ / (γ * (1 - ρ))
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
        
        assert np.isclose(self.stage2_duration * (1 - self.replay_ratio), self.rare_fraction_epoched * self.rare_stage2_allocation), f"Parameters are inconsistent: stage2_duration = {self.stage2_duration}, rare_fraction_epoched = {self.rare_fraction_epoched}, rare_stage2_allocation = {self.rare_stage2_allocation}, replay_ratio = {self.replay_ratio}. However, {self.stage2_duration} * (1 - {self.replay_ratio}) = {self.stage2_duration * (1 - self.replay_ratio)} != {self.rare_fraction_epoched} * {self.rare_stage2_allocation} = {self.rare_fraction_epoched * self.rare_stage2_allocation}"

        self.total_tokens = self.num_train_steps * self.train_batch_size * self.model_config.seq_len
        self.rare_batches = int(self.num_train_steps * self.rare_fraction_epoched)
        self.common_batches = self.num_train_steps - self.rare_batches

        self.rare_weight_stage1 = self.rare_fraction_epoched * (1 - self.rare_stage2_allocation)
        self.rare_weight_stage2 = self.rare_fraction_epoched * self.rare_stage2_allocation
        self.common_weight_stage1 = 1 - self.rare_weight_stage1
        self.common_weight_stage2 = 1 - self.rare_weight_stage2

        assert 0.0 <= self.rare_weight_stage1 <= 1.0, "Rare weight stage 1 must be between 0.0 and 1.0"
        assert 0.0 <= self.rare_weight_stage2 <= 1.0, "Rare weight stage 2 must be between 0.0 and 1.0"
        
    def build_name(self) -> str:
        return f"{self.model_name}-{self.format_num_tokens()}-{self.rare_data_name}x{self.rare_fraction:.2f}x{self.rare_data_epochs}-{self.common_data_name}-rr{self.replay_ratio:.2f}-rs{self.rare_stage2_allocation:.2f}-{self.format_lr_schedule()}{self.nametag}"

    def build_data_config(self) -> LMMixtureDatasetConfig:
        components = {
            self.rare_data_name: self.rare_data,
            self.common_data_name: self.common_data
        }

        transition_idx = int((1 - self.stage2_duration) * self.num_train_steps)
        # adjust by default block size
        transition_idx = (transition_idx // 2) * 2 # TODO: fix this
        stage1 = (0, {self.rare_data_name: self.rare_weight_stage1, self.common_data_name: self.common_weight_stage1})
        stage2 = (transition_idx, {self.rare_data_name: self.rare_weight_stage2, self.common_data_name: self.common_weight_stage2})

        if transition_idx == 0:
            weights_list = [stage2]
        else:
            weights_list = [stage1, stage2]

        max_train_batches = {self.rare_data_name: self.rare_batches}
        num_validation_batches = {self.rare_data_name: 10, self.common_data_name: 10}

        no_rare_data = all(stage[1][self.rare_data_name] == 0.0 for stage in weights_list)
        if no_rare_data:
            for stage in weights_list:
                stage[1].pop(self.rare_data_name)
            max_train_batches.pop(self.rare_data_name)
            num_validation_batches.pop(self.rare_data_name)

        no_common_data = all(stage[1][self.common_data_name] == 0.0 for stage in weights_list)
        if no_common_data:
            for stage in weights_list:
                stage[1].pop(self.common_data_name)
            num_validation_batches.pop(self.common_data_name)

        data_config = lm_varying_mixture_data_config(
            components=components,
            weights_list=weights_list,
            max_train_batches=max_train_batches,
            num_validation_batches=num_validation_batches,
        )

        return _prepare_data_config(data_config, use_default_validation=True)

    def format_num_tokens(self) -> str:
        """Format total number of tokens in B/M/K notation."""
        tokens = self.total_tokens
        if tokens >= 1_000_000_000:
            return f"{tokens/1_000_000_000:.1f}B"
        elif tokens >= 10_000_000:
            return f"{int(tokens/1_000_000)}M"
        elif tokens >= 1_000_000:
            return f"{tokens/1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens/1_000:.1f}K"
        return str(tokens)
        
    def format_lr_schedule(self) -> str:
        schedule_short_name = {
            "cosine": "cos",
            "linear": "wsd",
        }
        return f"{schedule_short_name[self.lr_schedule]}-{self.lr}-{self.lr_cooldown_duration:.2f}"

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
                keep=[dict(every=step_to_export, until=step_to_export+1) for step_to_export in self.steps_to_export_list],
            ),
            replica_dcn_axis_size=-1,
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
    
    def build_pod_config(self) -> PodConfig:
        return PodConfig(
            tpu_type=self.tpu_type,
            node_count=self.node_count,
        )
    
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
            config=self.build_train_lm_config(),
            pod_config=self.build_pod_config(),
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
