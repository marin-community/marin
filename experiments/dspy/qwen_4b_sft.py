"""
SFT script for Qwen-4B using DSPy format adaptation trace data.
"""

from levanter.data.text import ChatLmDatasetFormat
from experiments.defaults import default_sft, default_tokenize
from experiments.dspy.train_qwen_4b_config import QwenSFTConfig
from experiments.llama import llama3_instruct_tokenizer  # We might need a Qwen tokenizer instead

# Using AutoTokenizer for Qwen
from transformers import AutoTokenizer

from marin.execution import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

# Default path for trace data (can be local or GCS)
DEFAULT_CHAT_TRAIN_URLS = "experiments/dspy/format_adaptation_dataset.jsonl"

# Chat format configuration
# Qwen typically uses ChatML or similar. 
# For simplicity, if we converted our data to standard OpenAI format, ChatLmDatasetFormat (which usually handles standard formats) might work 
# OR we rely on the tokenizer's chat template.
dspy_chat_format = ChatLmDatasetFormat(
    messages_field="chat",
    single_turn=False,
    pack=True,
    mask_user_turns=True,
)

def get_qwen_tokenizer():
    # You might need to use the actual model path if it differs
    return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat", trust_remote_code=True)

# Tokenize chat JSONL data
# Note: Using Qwen tokenizer
tokenize_step = default_tokenize(
    name="qwen-4b-dspy-tokenize",
    dataset=DEFAULT_CHAT_TRAIN_URLS,
    tokenizer=get_qwen_tokenizer(), 
    format=dspy_chat_format,
)

# Create data config
tokenized_data = lm_data_config(tokenize_step, permutation_type="linear")

from levanter.models.llama import LlamaConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

# Define Qwen-4B configuration (based on Qwen/Qwen1.5-4B-Chat)
# Hidden size: 2560, Heads: 20, Layers: 40, Intermediate: 6912
qwen_4b_config = LlamaConfig(
    seq_len=4096,  # Qwen supports up to 32k but sticking to 4k for consistency with config
    hidden_dim=2560,
    intermediate_dim=6912,
    num_heads=20,
    num_kv_heads=20,
    num_layers=40,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True, 
)

sft_step = default_sft(
    name="qwen-4b-dspy-sft",
    tokenized=tokenized_data,
    model_config=qwen_4b_config,
    sft_config=sft_config,
    tags=["dspy", "format-adaptation", "qwen-4b"],
).with_output_path("checkpoints/qwen-4b-dspy-sft")

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_step, sft_step],
        description="SFT for Qwen-4B using DSPy adaptation traces.",
    )
