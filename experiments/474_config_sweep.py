# https://github.com/stanford-crfm/marin/issues/474
# Sweep to determine optimal training config
import math
from levanter.models.llama import LlamaConfig


from experiments.simple_train_config import SimpleTrainConfig
from experiments.llama import llama_150m

from marin.execution.executor import ExecutorStep, this_output_path, versioned, executor_main


# TODO: might be nice to do use wandb sweeps, but not today.
# TODO: redo with mup

# Sweep to determine optimal training config
LR_CHOICES = [1e-4, 3e-4, 1e-3, 3e-3]
WD_CHOICES = [0.05, 0.1, 0.25, 0.33]
TPU_TYPES = ["v4-16", "v4-32", "v4-64"]
TOKEN_TARGETS = [1, 3, 10, 30, 50] * 1_000_000_000
BATCH_SIZE = [256, 512, 1024, 2048, 4096]
SEQ_LEN = 4096

def step_target(token_target, batch_size):
    return math.ceil(token_target / (batch_size * SEQ_LEN))


train_configs = []

for lr in LR_CHOICES:
    for wd in WD_CHOICES:
        for tpu_type in TPU_TYPES:
            for batch_size in BATCH_SIZE:
                for token_target in TOKEN_TARGETS:
                    num_train_steps = step_target(token_target, batch_size)
                    train_configs.append(
                        SimpleTrainConfig(
                            tpu_type=tpu_type,
                            train_batch_size=batch_size,
                            num_train_steps=num_train_steps,
                            learning_rate=lr,
                            weight_decay=wd,
                        )
                    )


def make_sweep_steps(
        model_config: LlamaConfig,
        train_configs: list[SimpleTrainConfig],
        data: ExecutorStep,
):
    tokenized =
    steps = []
    for i, train_config in enumerate(train_configs):
        steps.append(
            ExecutorStep(
                name=f"train/{i}",
                fn=dclm_baseline,
                config=train_config,
                override_output_path=this_output_path(),
                model_config=model_config,
                data_config=data_config,
            )
        )
    return steps
