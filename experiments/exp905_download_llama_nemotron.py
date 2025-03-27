from datetime import timedelta

import jmp
from instruction_datasets import get_instruction_dataset
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMSupervisedDatasetConfig
from levanter.models.llama import LlamaConfig
from levanter.models.rotary import Llama3RotaryEmbeddingsConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft
from marin.training.training import TrainSFTOnPodConfig, run_levanter_sft

# Get instruction dataset
nemotron_dataset = get_instruction_dataset(
    "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
)  # all splits

# # TODO: Calculate actual number of tokens for proper steps
# NUM_TRAIN_TOKENS = 150849275  # This needs to be updated with actual token count
# NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 3  # 3 epochs

NUM_TRAIN_STEPS = 2500

# Add tokenization step
nemotron_llama_tokenize_step = ExecutorStep(
    name="tokenized/llama_nemotron_post_training_v1_sft",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(nemotron_dataset, "**/*.jsonl")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        input_field="input",
        output_field="output",
    ),
    description="Tokenize Llama-Nemotron-Post-Training-Dataset-v1-SFT data",
)

seed = 1
llama_8b_nemotron_sft_model = ExecutorStep(
    name=f"checkpoints/llama3.1_8b_nemotron_post_training_v1_sft_seed{seed}",
    fn=run_levanter_sft,
    config=TrainSFTOnPodConfig(
        output_path=this_output_path(),
        tpu_type="v4-128",
        tokenizer=llama3_tokenizer,
        chat_train_urls=[output_path_of(nemotron_dataset, "**/*.jsonl")],
        supervised_data=LMSupervisedDatasetConfig(
            cache_dir=output_path_of(nemotron_llama_tokenize_step), input_field="input", output_field="output"
        ),
        initialize_from_hf=False,
        model_name_or_path="meta-llama/Llama-3.1-8B",
        max_seq_len=4096,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            seed=seed,
            train_batch_size=128,
            num_train_steps=NUM_TRAIN_STEPS,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=25000)],
            ),
        ),
        model=LlamaConfig(
            seq_len=4096,
            hidden_dim=4096,
            intermediate_dim=14336,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            use_bias=False,
            use_layer_norm_weight=True,
            initializer_range=0.02,
            use_flash_attention=True,
            flash_attention_block_size=512,
            rope=Llama3RotaryEmbeddingsConfig(
                theta=500000,
                factor=8.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=8192,
            ),
        ),
        optimizer=AdamConfig(
            learning_rate=5e-6,
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
    executor_main(
        steps=[
            nemotron_llama_tokenize_step,
            # llama_8b_nemotron_sft_model
        ]
    )
