from datetime import timedelta

import jmp
from instruction_datasets import get_instruction_dataset
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMSupervisedDatasetConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft
from marin.training.training import TrainSFTOnPodConfig, run_levanter_sft

# Get instruction dataset
tulu_3_dataset = get_instruction_dataset("allenai/tulu-3-sft-mixture")

# Number of tokens is 670,426,314
NUM_TRAIN_TOKENS = 670426314
# number of epochs over the dataset set to reproduce Olmo SFT v2
# or Tulu 3 starting from Llama 3.1 8B. This script
# is used to reproduce the Tulu 3 SFT model.
# Link: https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 2  # 2 epochs

import transformers; print(f"Transformers version: {transformers.__version__}")
# Add tokenization step
olmo2_tokenize_step = ExecutorStep(
    name="tokenized/tulu_sft_v3_olmo2_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(tulu_3_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer="allenai/OLMo-2-1124-7B",
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize chat SFT data",
)

olmo2_model = ExecutorStep(
    name="checkpoints/olmo2_sft",
    fn=run_levanter_sft,
    config=TrainSFTOnPodConfig(
        output_path=this_output_path(),
        tpu_type="v4-128",
        tokenizer=llama3_tokenizer,
        chat_train_urls=[output_path_of(tulu_3_dataset, "**/*.jsonl.gz")],
        supervised_data=LMSupervisedDatasetConfig(
            cache_dir=output_path_of(olmo2_tokenize_step), input_field="user", output_field="assistant"
        ),
        initialize_from_hf=True,
        model_name_or_path="allenai/OLMo-2-1124-7B",
        max_seq_len=4096,
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
        ),
        model=LlamaConfig(
            seq_len=4096,  # Seq len set to reproduce Tulu SFT
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
        # Reproduce Tulu 3 SFT Linear warmup for the first 3% of total training time, then cooldown to 0
        optimizer=AdamConfig(
            learning_rate=5e-6,  #  5x10^-6
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
    executor_main(steps=[olmo2_tokenize_step, olmo2_model])
