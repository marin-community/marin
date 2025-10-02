"""
# ===================================================================
# === ADD THIS SECTION FOR DETAILED LOGGING =========================
# ===================================================================
import logging
import os
import sys

# 1. Force Ray worker logs to stream to your driver's terminal (stderr)
# This is the most important change to see the remote errors.
os.environ['RAY_LOG_TO_STDERR'] = '1'

# 2. Get full, unfiltered tracebacks from JAX
# By default, JAX hides a lot of the traceback. 'off' gives you everything.
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

# 3. Configure Python's root logger to be more verbose
# This will capture detailed logs from Ray's components and other libraries.
# We set the level to INFO; you can change it to DEBUG for even more detail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    stream=sys.stdout,
)

print("--- Detailed logging has been enabled. Waiting for job to start... ---")
# ===================================================================
# === END OF LOGGING SECTION ========================================
# ===================================================================
"""
# nodryrun
import logging
import os

from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

#logger = logging.getLogger("ray")
#os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'
#os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

# This config uses MixtralConfig directly for MoE functionality
moe_300m_config = MixtralConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    # Disable MoE auxiliary loss logging to prevent JAX tracer leaks
    lbl_coef=None,  # Disables load balancing loss logging
    rzl_coef=None,  # Disables router z-loss logging
    use_gmm=False,  # Use ragged dot implementation with debug prints
)
train_batch_size = 1536
speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University",
        url="https://stanford.edu/~pranshu",
    ),
    description="Training a 300M parameter Mixtral-style MoE model on a TPU.",
    model_config=moe_300m_config,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v5p-8", slice_count=1),
        train_batch_size=train_batch_size, #targetting 6 billion tokens in total
        num_train_steps=6000,
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun(f"pranshu_mixtral_300m_run_central1_fresh_bs{train_batch_size}", speedrun_config))
