from experiments.defaults import default_sft, default_tokenize
from experiments.instruction_datasets import get_instruction_dataset, llama3_instruct_trainable_chat_template
from experiments.llama import llama3_instruct_tokenizer, llama3_tokenizer, llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import executor_main

# Get instruction dataset
synthetic_instruction_dataset = get_instruction_dataset("sherryy/tulu-3-sft-personas-instruction-following-expanded")

# TODO: tune this for a good number of steps
NUM_TRAIN_STEPS = 2500

# Add tokenization step
synthetic_instruction_llama_tokenized = default_tokenize(
    name="synthetic_instruction_llama3_tokenizer",
    dataset=synthetic_instruction_dataset / "**/*.jsonl.gz",
    tokenizer=llama3_instruct_tokenizer,
    format=llama3_instruct_trainable_chat_template,
)

seed = 1
# tulu3_sft_8b_synthetic_instruction_model = ExecutorStep(
#     name=f"checkpoints/tulu3_sft_synthetic_instruction{seed}",
#     fn=run_levanter_sft,
#     config=TrainSFTOnPodConfig(
#         output_path=this_output_path(),
#         pod_config=PodConfig(tpu_type="v4-64"),
#         config=SFTConfig(
#             tokenizer=llama3_tokenizer,
#             chat_train_urls=[output_path_of(synthetic_instruction_dataset, "**/*.jsonl.gz")],
#             supervised_data=SupervisedUrlSourceConfig(
#                 cache_dir=output_path_of(synthetic_instruction_llama_tokenized),
#                 input_field="user",
#                 output_field="assistant",
#             ),
#             initialize_from_hf=False,
#             model_name_or_path="meta-llama/Llama-3.1-8B",
#             max_seq_len=4096,
#             # Modify the nested trainer config by creating a new one
#             trainer=TrainerConfig(
#                 tracker=WandbConfig(
#                     project="marin",
#                 ),
#                 mp=jmp.get_policy("p=f32,c=bfloat16"),
#                 seed=seed,
#                 train_batch_size=64,
#                 num_train_steps=NUM_TRAIN_STEPS,
#                 checkpointer=CheckpointerConfig(
#                     save_interval=timedelta(minutes=10),
#                     keep=[dict(every=25000)],
#                 ),
#                 initialize_from="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3-12305c/checkpoints/step-9980/",
#             ),
#             model=LlamaConfig(
#                 seq_len=4096,  # Seq len set to reproduce Tulu SFT
#                 hidden_dim=4096,
#                 intermediate_dim=14336,
#                 num_layers=32,
#                 num_heads=32,
#                 num_kv_heads=8,
#                 use_bias=False,
#                 use_layer_norm_weight=True,
#                 initializer_range=0.02,
#                 use_flash_attention=True,
#                 flash_attention_block_size=512,
#                 rope=Llama3RotaryEmbeddingsConfig(
#                     # Using Llama3 defaults from the code
#                     theta=500000,
#                     factor=8.0,
#                     low_freq_factor=1.0,
#                     high_freq_factor=4.0,
#                     original_max_position_embeddings=2048,
#                 ),
#             ),
#             # TODO: tune this for a good learning rate
#             optimizer=AdamConfig(
#                 learning_rate=5e-6,  #  5x10^-6
#                 weight_decay=0.0,
#                 warmup=0.03,
#                 cooldown=0.0,
#                 min_lr_ratio=0.0,
#                 lr_schedule="linear",
#                 max_grad_norm=None,
#                 haps=None,
#                 weight_decay_modules=None,
#                 default_weight_decay_mask=None,
#             ),
#             hf_save_steps=500,
#         ),
#     ),
# )

tulu3_sft_8b_synthetic_instruction_model = default_sft(
    name=f"checkpoints/tulu3_sft_synthetic_instruction{seed}",
    tokenized=synthetic_instruction_llama_tokenized,
    model_config=llama_8b,
    sft_config=SimpleSFTConfig(
        train_batch_size=64,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=5e-6,
        tpu_type="v4-64",
        tokenizer=llama3_tokenizer,
        model_name_or_path="meta-llama/Llama-3.1-8B",
        max_seq_len=4096,
        seed=seed,
        initialize_from_hf=False,
        initialize_from_checkpoint_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3-12305c/checkpoints/step-9980/",
    ),
    tags=["llama", "8b", "synthetic_instruction", "exp727"],
)


if __name__ == "__main__":
    executor_main(steps=[synthetic_instruction_llama_tokenized, tulu3_sft_8b_synthetic_instruction_model])
