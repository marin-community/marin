# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level configuration for alternating RL runs."""

import math
from dataclasses import dataclass, field

from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.trainer import TrainerConfig

from marin.rl.curriculum import CurriculumConfig
from marin.rl.rl_losses import RLLossModule


@dataclass
class AlternatingRLConfig:
    """Everything the controller needs to orchestrate an alternating RL run."""

    run_id: str
    shared_root: str
    tpu_name: str
    tpu_type: str
    zone: str
    project: str

    # Model
    model_name_or_path: str
    model_config_class: type[HFCompatConfig]
    tokenizer: str
    initial_checkpoint: str

    # Training
    trainer: TrainerConfig
    loss: RLLossModule
    curriculum: CurriculumConfig

    # Phase sizing
    steps_per_phase: int = 80
    global_batch_size: int = 1024
    n_generations_per_prompt: int = 16
    eval_groups_per_host: int = 128

    # Seed
    seed: int = 42

    # Override per-host train groups (None = derived from steps_per_phase)
    train_groups_per_host: int | None = None

    # Max phases (None = run until trainer.num_train_steps)
    max_phases: int | None = None

    # Container image (None = auto-build)
    image: str | None = None

    # Compilation cache paths (on persistent Docker volume)
    training_compilation_cache_dir: str = "/home/levanter/compilation-cache/train"
    sampling_compilation_cache_dir: str = "/home/levanter/compilation-cache/vllm"

    # Container names
    sampler_container_name: str = "marin-alt-sampler"
    materializer_container_name: str = "marin-alt-materializer"
    trainer_container_name: str = "marin-alt-trainer"

    # Learning rate and optimizer
    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0

    # Inference
    max_model_len: int | None = None
    inference_gpu_memory_utilization: float = 0.90

    # Extra env vars to pass to containers
    extra_env: dict[str, str] = field(default_factory=dict)

    # Capacity type for TPU creation
    capacity_type: str = "on-demand"

    # ---------------------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------------------

    @property
    def state_dir(self) -> str:
        return f"{self.shared_root}/state"

    @property
    def run_state_path(self) -> str:
        return f"{self.state_dir}/run_state.json"

    @property
    def curriculum_dir(self) -> str:
        return f"{self.shared_root}/curriculum"

    @property
    def policies_dir(self) -> str:
        return f"{self.shared_root}/policies"

    @property
    def levanter_checkpoints_dir(self) -> str:
        return f"{self.shared_root}/levanter_checkpoints"

    @property
    def sampling_dir(self) -> str:
        return f"{self.shared_root}/sampling"

    @property
    def materialized_dir(self) -> str:
        return f"{self.shared_root}/materialized"

    @property
    def max_seq_len(self) -> int:
        return self.curriculum.max_seq_len

    def required_train_groups_per_host(self, num_hosts: int) -> int:
        """Derive per-host train quota from phase budget."""
        if self.train_groups_per_host is not None:
            return self.train_groups_per_host
        required_individual = self.steps_per_phase * self.global_batch_size
        required_groups = math.ceil(required_individual / self.n_generations_per_prompt)
        return math.ceil(required_groups / num_hosts)

    def policy_path(self, version: int) -> str:
        return f"{self.policies_dir}/policy_{version:04d}"

    def policy_manifest_path(self, version: int) -> str:
        return f"{self.policy_path(version)}/manifest.json"

    def sampling_phase_dir(self, phase_id: int) -> str:
        return f"{self.sampling_dir}/phase_{phase_id:04d}"

    def sampling_manifest_path(self, phase_id: int) -> str:
        return f"{self.sampling_phase_dir(phase_id)}/manifest.json"

    def materialized_phase_dir(self, phase_id: int) -> str:
        return f"{self.materialized_dir}/phase_{phase_id:04d}"

    def materialized_manifest_path(self, phase_id: int) -> str:
        return f"{self.materialized_phase_dir(phase_id)}/manifest.json"

    def levanter_checkpoint_dir(self, phase_id: int) -> str:
        return f"{self.levanter_checkpoints_dir}/phase_{phase_id:04d}"

    def host_output_dir(self, phase_id: int, host_ordinal: int) -> str:
        return f"{self.sampling_phase_dir(phase_id)}/host_{host_ordinal:03d}"

    def host_status_path(self, phase_id: int, host_ordinal: int) -> str:
        return f"{self.host_output_dir(phase_id, host_ordinal)}/status.json"
