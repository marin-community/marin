"""
Speedrun code for a 300M parameter model based on the LLaMA architecture.
"""

# import logging

# from experiments.llama import llama_300m
# from experiments.simple_train_config import SimpleTrainConfig
# from marin.execution.executor import executor_main
# from marin.resources import TpuPodConfig
# from marin.speedrun.speedrun import SpeedrunConfig

# logger = logging.getLogger("ray")

# speedrun_config = SpeedrunConfig(
#     author="Nikil Ravi",
#     affiliation="Marin Community",
#     description="300M parameter model based on LLaMA architecture.",
#     model_config=llama_300m,
#     train_config=SimpleTrainConfig(
#         TpuPodConfig(tpu_type="v4-256"),
#         train_batch_size=1024,
#         num_train_steps=3000,
#         learning_rate=3e-3,
#         weight_decay=0.1,
#         steps_per_eval=1000,
#     ),
# )

"""
 Speedrun code for a 300M activated parameter MoE model based on the Mixtral architecture.
 This model has 32 experts and only activates 4 of them.
 """

import logging

from levanter.models.mixtral import MixtralConfig

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import SpeedrunConfig

logger = logging.getLogger("ray")

moe_300m = MixtralConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_heads=12,
    num_kv_heads=12,
    num_layers=12,
    n_routed_experts=32,
    num_experts_per_tok=4,
)

speedrun_config = SpeedrunConfig(
    author="Nikil Ravi",
    affiliation="Marin Community",
    description="300M activated parameter MoE model based on the Mixtral architecture. This model has 32 experts and only activates 4 of them.",
    model_config=moe_300m,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-256"),
        train_batch_size=1024,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":

    print(speedrun_config.estimate_model_flops())
    print(f"Number of devices: {speedrun_config.num_devices}")
    print(f"Device FLOPs: {speedrun_config.device_flops}")
    print(f"Total peak FLOPs: {speedrun_config.device_flops * speedrun_config.num_devices}")
    # executor_main(steps=default_speedrun("300M_llama", speedrun_config))
