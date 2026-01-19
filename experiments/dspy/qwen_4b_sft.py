"""
SFT script for Qwen-4B using DSPy format adaptation trace data.
"""

from levanter.data.text import ChatLmDatasetFormat
from experiments.defaults import default_sft, default_tokenize
from experiments.dspy.train_qwen_4b_config import QwenSFTConfig
from levanter.models.llama import LlamaConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from transformers import AutoTokenizer
from marin.execution import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

# Configuration
config = QwenSFTConfig()

# Default path for trace data
DEFAULT_CHAT_TRAIN_URLS = "experiments/dspy/format_adaptation_dataset.jsonl"

def get_qwen_tokenizer():
    return AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

# Chat format configuration
dspy_chat_format = ChatLmDatasetFormat(
    messages_field="chat",
    single_turn=False,
    pack=True,
    mask_user_turns=True,
)

# Tokenize chat JSONL data
tokenize_step = default_tokenize(
    name="qwen-4b-dspy-tokenize",
    dataset=DEFAULT_CHAT_TRAIN_URLS,
    tokenizer=get_qwen_tokenizer(), 
    format=dspy_chat_format,
)

# Create data config
tokenized_data = lm_data_config(tokenize_step, permutation_type="linear")

# Define Qwen-4B configuration (based on Qwen/Qwen1.5-4B-Chat)
qwen_4b_model_config = LlamaConfig(
    seq_len=config.max_seq_len,
    hidden_dim=2560,
    intermediate_dim=6912,
    num_heads=20,
    num_kv_heads=20,
    num_layers=40,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True, 
)

# SFT Step
sft_step = default_sft(
    name="qwen-4b-dspy-sft",
    tokenized=tokenized_data,
    model_config=qwen_4b_model_config,
    
    # Pass training parameters from config
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    num_train_steps=config.num_train_steps,
    train_batch_size=config.train_batch_size,
    
    tags=["dspy", "format-adaptation", "qwen-4b"],
).with_output_path("checkpoints/qwen-4b-dspy-sft")

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_step, sft_step],
        description="SFT for Qwen-4B using DSPy adaptation traces.",
    )
