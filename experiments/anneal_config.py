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

from dataclasses import dataclass, field

from fray.cluster import ResourceConfig
from marin.execution import InputName
from marin.processing.tokenize.data_configs import LMMixtureDatasetConfig


@dataclass(frozen=True)
class AnnealConfig:
    # 198468 steps is roughly 198468 steps * 1024 batch size * 4096 seq len = 0.832T tokens
    # Numbers were taken from exp600_tootsie.py. We start with this 8B model because it would take a long time to train
    DEFAULT_CHECKPOINT_PATH = InputName.hardcoded("checkpoints/llama-8b-tootsie-0.001-19ad63/checkpoints/step-660000")
    LLAMA_MAX_SEQ_LEN = 4096

    # Annealing dataset and proportions
    dataset_config: LMMixtureDatasetConfig

    # Model Checkpoint related
    # The path to the checkpoint to initialize from. This is the checkpoint you start the annealing from.
    initialize_from_checkpoint_path: str | InputName = DEFAULT_CHECKPOINT_PATH

    # Training schedule related
    # The learning rate to use for training. Since our checkpoint has a stable phase LR of 1e-3, we use that.
    learning_rate: float = 1e-3

    # The minimum learning rate to use. We default anneal to 0 learning rate similar to other works.
    min_lr_ratio: float = 0.0

    # The weight decay used by the training optimizer.
    weight_decay: float = 0.05

    # The learning rate schedule to use. Linear is recommended and commonly used for annealing.
    lr_schedule: str = "linear"

    # The batch size to use for training.
    train_batch_size: int = 1024

    # The number of tokens to anneal for. 50B tokens is a rough default.
    num_anneal_training_tokens: int = 50_000_000_000  # 50B tokens

    # Hardware related
    # The number of TPUs to use, type of TPU, and the number of pods to use.
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v4-128", slice_count=2))

    # Checkpoint related
    # The number of steps between saving checkpoints. Larger values will save checkpoints more frequently.
    steps_per_export: int = 10000

    # Validation related
    # This argument is used in the default_train. If set to True, the validation set is Paloma.
    # If set to False, we will not calculate validation loss.
    use_default_validation: bool = True
