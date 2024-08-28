import argparse
import yaml

from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model_path", type=lambda s: Path(s), required=True)


def calculate_model_size(
    vocab_size: int = 128256, 
    hidden_dim: int = 4096,
    intermediate_dim: int = 14336,
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    **kwargs,
):
    """Calculates number of parameters given a llama model config.
    Calculations done with the following assumptions:
    1. llama3 tokenizer (128256 vocab size)
    2. swiglu activation (3 linear layers in MLP)
    3. untied embedding and lm_head weights
    4. rotary embeddings (ignore frequency buffers)
    5. RMSNorm
    """

    token_embedding = vocab_size * hidden_dim

    head_size = hidden_dim // num_heads
    q_proj = hidden_dim * head_size * num_heads
    kv_proj = 2 * hidden_dim * head_size * num_kv_heads
    o_proj = head_size * num_heads * hidden_dim
    attn = q_proj + kv_proj + o_proj

    mlp = 3 * hidden_dim * intermediate_dim

    transformer_layer = attn + mlp + 2 * hidden_dim # plus 2 rmsnorm

    transformer = num_layers * transformer_layer + hidden_dim # plus final rmsnorm

    lm_head = hidden_dim * vocab_size

    print(f"Non-embedding params: {transformer}")
    print(f"Total params: {token_embedding + transformer + lm_head}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    assert args.model_path.exists(), f"{args.model_path} does not exist."
    
    print(f"Model config: {args.model_path}")
    with open(args.model_path, 'r') as f:
        data = yaml.safe_load(f)

    assert data['type'] == 'llama', "Model size calculator currently only supports llama models"

    calculate_model_size(**data)
