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
    token_embedding = vocab_size * hidden_dim

    head_size = hidden_dim // num_heads
    attn = hidden_dim * head_size * (2 * num_kv_heads + num_heads) + num_heads * head_size * hidden_dim

    mlp = 3 * hidden_dim * intermediate_dim

    transformer_layer = attn + mlp + 2 * hidden_dim

    transformer = num_layers * transformer_layer + hidden_dim

    lm_head = hidden_dim * vocab_size

    print(f"Non-embedding params: {transformer}")
    print(f"Total params: {token_embedding + transformer + lm_head}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    assert args.model_path.exists(), f"{args.model_path} does not exist."
    
    with open(args.model_path, 'r') as f:
        data = yaml.safe_load(f)

    assert data['type'] == 'llama', "Model size calculator currently only supports llama models"

    calculate_model_size(**data)
