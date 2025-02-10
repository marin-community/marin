from datetime import timedelta

import jmp
from instruction_datasets import get_instruction_dataset
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMSupervisedDatasetConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft
from marin.training.training import TrainSFTOnPodConfig, run_levanter_sft

# Get instruction dataset
instruction_dataset = get_instruction_dataset("allenai/tulu-v2-sft-mixture-olmo-4096")

# Number of tokens in the SFT dataset below
NUM_TRAIN_TOKENS = 150849275
# number of epochs over the dataset set to reproduce Olmo SFT
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 3  # 3 epochs

# Add tokenization step
tokenize_step = ExecutorStep(
    name="tokenized/olmo702024_sft_4096_3eps",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(instruction_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer="EleutherAI/gpt-neox-20b",
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",  # Or whatever tokenizer you're using
    ),
    description="Tokenize chat SFT data",
)


train_step = ExecutorStep(
    name="checkpoints/olmo7_072024_sft_4096_3eps_debug",
    fn=run_levanter_sft,
    config=TrainSFTOnPodConfig(
        output_path=this_output_path(),
        tpu_type="v4-128",
        tokenizer="EleutherAI/gpt-neox-20b",
        chat_train_urls=[output_path_of(instruction_dataset, "**/*.jsonl.gz")],
        supervised_data=LMSupervisedDatasetConfig(
            cache_dir=output_path_of(tokenize_step), input_field="user", output_field="assistant"
        ),
        # Modify the nested trainer config by creating a new one
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=128,
            num_train_steps=NUM_TRAIN_STEPS,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=25000)],
            ),
            initialize_from="gs://levanter-checkpoints/marin/olmoish7b_v4_1024_0627/dlwh_7b0627/step-510000/",
        ),
        model=LlamaConfig(
            seq_len=4096,  # Seq len set to reproduce Olmo SFT
            hidden_dim=4096,
            intermediate_dim=11008,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            use_bias=False,
            use_layer_norm_weight=False,
            initializer_range=0.02,
            use_flash_attention=True,
            flash_attention_block_size=512,
        ),
        # Reproduce OLMO SFT Linear warmup for the first 3% of total training time, then cooldown to 0
        optimizer=AdamConfig(
            learning_rate=2e-6,  #  2x10^-6
            weight_decay=0.0,
            warmup=0.03,
            cooldown=0.0,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            max_grad_norm=None,
            haps=None,
            weight_decay_modules=None,
            default_weight_decay_mask=None,
        ),
        hf_save_steps=500,
    ),
)


if __name__ == "__main__":
    executor_main(steps=[tokenize_step, train_step])
