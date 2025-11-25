# nodryrun
import logging

from experiments.llama import llama3_tokenizer
from experiments.nemotron_cc.tokenize_nemotron import NEMOTRON_WEIGHTS, tokenize_nemotron_steps
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

nemotron_cc_steps = tokenize_nemotron_steps(tokenizer=llama3_tokenizer)
nemotron_cc_mixture = lm_mixture_data_config(
    components=nemotron_cc_steps,
    weights=NEMOTRON_WEIGHTS,
    permutation_type="linear",
)
# Full Mixtral 8x7B configuration using ragged-dot MoE kernels.
mixtral_8x7b_ragged = MixtralConfig(
    seq_len=512,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
    n_routed_experts=8,
    num_experts_per_tok=2,
    gradient_checkpointing=True,
    scan_layers=True,
    use_gmm=False,  # stick with ragged-dot experts
    cross_entropy_block_size=32000,
    lbl_coef=None,
    rzl_coef=None,
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Marin Team",
        affiliation="Marin Project",
        url=None,
    ),
    description=(
        "Train a Mixtral 8x7B MoE model (ragged dot) for 20 steps on the Nemotron-CC tokenized cache hosted in "
        "gs://marin-us-east5, using batch size 128 (global), seq_len 512, and a 32k cross-entropy block size."
    ),
    model_config=mixtral_8x7b_ragged,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v5p-64"),
        train_batch_size=32, #multiple of num devices 
        num_train_steps=20,
        learning_rate=3e-4,
        weight_decay=0.1,
        steps_per_eval=20,
        steps_per_export=20,
    ),
    tokenized_dataset=nemotron_cc_mixture,
)

if __name__ == "__main__":
    logger.info("Launching Mixtral 8x7B ragged-dot speedrun.")
    logger.info(
        "Settings: batch_size=%s, seq_len=%s, steps=%s, cross_entropy_block_size=%s",
        speedrun_config.train_config.train_batch_size,
        mixtral_8x7b_ragged.seq_len,
        speedrun_config.train_config.num_train_steps,
        mixtral_8x7b_ragged.cross_entropy_block_size,
    )
    executor_main(steps=default_speedrun("mixtral_8x7b_ragged_speedrun_bs8_useast5a_v5p_64", speedrun_config))
