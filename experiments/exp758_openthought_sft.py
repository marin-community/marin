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
openthoughts_dataset = get_instruction_dataset("open-r1/OpenThoughts-114k-math")

# TODO: tune this for a good number of steps
NUM_TRAIN_STEPS = 2500

# Add tokenization step
openthoughts_llama_tokenize_step = ExecutorStep(
    name="tokenized/openthoughts_llama3_tokenizer",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[output_path_of(openthoughts_dataset, "**/*.jsonl.gz")],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
        # fixed to OAI chat format
        input_field="user",
        output_field="assistant",
    ),
    description="Tokenize chat SFT data",
)

seed = 1
tulu3_sft_8b_openthoughts_model = ExecutorStep(
    name=f"checkpoints/tulu3_sft_openthoughts{seed}",
    fn=run_levanter_sft,
    config=TrainSFTOnPodConfig(
        output_path=this_output_path(),
        tpu_type="v4-128",
        tokenizer=llama3_tokenizer,
        chat_train_urls=[output_path_of(openthoughts_dataset, "**/*.jsonl.gz")],
        supervised_data=LMSupervisedDatasetConfig(
            cache_dir=output_path_of(openthoughts_llama_tokenize_step), input_field="user", output_field="assistant"
        ),
        initialize_from_hf=False,
        model_name_or_path="meta-llama/Llama-3.1-8B",
        max_seq_len=4096,
        # Modify the nested trainer config by creating a new one
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
            initialize_from="gs://meta-llama/Llama-3.1-8B",
        ),
        model=LlamaConfig(
            seq_len=4096,  # Seq len set to reproduce Tulu SFT
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
                # Using Llama3 defaults from the code
                theta=500000,
                factor=8.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=8192,
            ),
        ),
        # TODO: tune this for a good learning rate
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
    executor_main(steps=[openthoughts_llama_tokenize_step, tulu3_sft_8b_openthoughts_model])
