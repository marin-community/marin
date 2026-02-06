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

"""Audio-specific default configurations."""


from experiments.anneal_config import AnnealConfig
from experiments.defaults import SimpleTrainConfig, default_train
from marin.execution import InputName
from marin.execution.executor import ExecutorStep
from levanter.models.qwen import Qwen3Config


def default_audio_anneal(name: str, model_config: Qwen3Config, anneal_config: AnnealConfig) -> ExecutorStep:
    """
    Runs an annealing training run for audio models using model_config.

    This is similar to default_anneal but uses the model config for the annealing run.

    Args:
        name: The name of the training run. Will form the basis of the output path for the executor step.
              `checkpoints/` will be prepended to the name.
        model_config: Configuration for the model.
        anneal_config: Configuration for the annealing run.

    Returns:
        An ExecutorStep configured for annealing.
    """
    checkpoint_path = anneal_config.initialize_from_checkpoint_path
    imputed_checkpoint_step = _impute_checkpoint_step(checkpoint_path)

    num_anneal_steps = anneal_config.num_anneal_training_tokens / (
        anneal_config.train_batch_size * AnnealConfig.LLAMA_MAX_SEQ_LEN
    )
    num_train_steps = imputed_checkpoint_step + num_anneal_steps

    # Calculate the max learning rate to simulate a linear decay schedule
    learning_rate = num_train_steps * (anneal_config.learning_rate / num_anneal_steps)

    anneal_stage_train_config = SimpleTrainConfig(
        resources=anneal_config.resources,
        train_batch_size=anneal_config.train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=anneal_config.weight_decay,
        min_lr_ratio=anneal_config.min_lr_ratio,
        steps_per_export=anneal_config.steps_per_export,
        lr_schedule=anneal_config.lr_schedule,
        initialize_from_checkpoint_path=checkpoint_path,
    )

    # Use qwen3_0_6b with the same modifications as the original training
    # yodas_qwen = dataclasses.replace(
    #     qwen3_0_6b, tie_word_embeddings=False, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload")
    # )

    return default_train(
        name=name,
        tokenized=anneal_config.dataset_config,
        model_config=model_config,
        train_config=anneal_stage_train_config,
        use_default_validation=anneal_config.use_default_validation,
        eval_harness_tasks=[],
    )


def _impute_checkpoint_step(checkpoint_path: str | InputName) -> int:
    """
    Extracts the checkpoint step from a checkpoint path.

    Args:
        checkpoint_path: Path to the checkpoint, e.g. "checkpoints/model/step-12345"

    Returns:
        The step number as an integer.
    """
    if isinstance(checkpoint_path, InputName):
        checkpoint_path = checkpoint_path.name or ""

    parts = checkpoint_path.split("/")
    for part in reversed(parts):
        if part.startswith("step-"):
            return int(part.split("-")[1])

    raise ValueError(f"Could not extract checkpoint step from path: {checkpoint_path}")
