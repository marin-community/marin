# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Simple script to check train-time vs inference-time logprobs match.
Uses a simple autoregressive loop with no KV cache.
"""

import numpy as np
import jax
import jax.numpy as jnp
import haliax as hax
import optax

from levanter.layers.attention import AttentionMask


def simple_autoregressive_inference(model, prompt_tokens, max_new_tokens, key):
    """
    Simple autoregressive inference with no KV cache.
    
    Args:
        model: The language model
        prompt_tokens: Initial prompt tokens as a NamedArray with ["batch", "position"] axes
        max_new_tokens: Maximum number of tokens to generate
        key: JAX random key for sampling
        
    Returns:
        Generated tokens (including prompt) as a NamedArray
    """
    current_tokens = prompt_tokens
    batch_axis = current_tokens.axes[0]
    position_axis = current_tokens.axes[1]
    
    for _ in range(max_new_tokens):
        # Run model on full sequence (no KV cache)
        logits = model(input_ids=current_tokens, attn_mask=AttentionMask.causal(), key=key)
        
        # Get logits for last position using Haliax axis indexing
        logits_position_axis = logits.axes[1]
        next_token_logits = logits[logits_position_axis, -1]  # Select last position: [batch, vocab]
        
        # Sample next token (greedy for simplicity)
        next_token = jnp.argmax(next_token_logits.array, axis=-1, keepdims=True)  # [batch, 1]
        
        # Concatenate with current tokens by working with raw arrays
        current_array = current_tokens.array  # [batch, seq_len]
        new_array = jnp.concatenate([current_array, next_token], axis=-1)  # [batch, seq_len+1]
        
        # Rewrap as NamedArray with updated position axis
        new_position_axis = position_axis.resize(new_array.shape[-1])
        current_tokens = hax.named(new_array, [batch_axis, new_position_axis])
        
    return current_tokens


def check_logprobs_match(model, tokens, expected_logprobs, prompt_length):
    """
    Check that model logprobs match expected logprobs from an API response.
    
    Args:
        model: The language model
        tokens: Full sequence of tokens (prompt + response) as NamedArray ["batch", "position"]
        expected_logprobs: Expected logprobs for response tokens (numpy array)
        prompt_length: Length of prompt tokens
        
    Returns:
        Dictionary with comparison metrics
    """
    # Run model on full sequence
    logits = model(input_ids=tokens, attn_mask=AttentionMask.causal(), key=jax.random.PRNGKey(0))
    
    # Compute logprobs: [bsz, seq_len, vocab_size]
    logits_array = logits.array.astype(jnp.float32)[:, :-1, :]  # Remove last position
    labels = tokens.array[:, 1:]  # Shift labels by 1
    
    # Compute cross-entropy loss (negative logprob)
    cross_entropy = optax.softmax_cross_entropy_with_integer_labels(logits_array, labels)
    model_logprobs = -1 * cross_entropy  # Convert to logprobs
    
    # Extract response logprobs (after prompt)
    response_logprobs = model_logprobs[0, prompt_length - 1:]
    
    # Compare with expected logprobs
    mean_diff = np.mean(np.abs(expected_logprobs - response_logprobs))
    max_diff = np.max(np.abs(expected_logprobs - response_logprobs))
    
    results = {
        "model_logprobs": response_logprobs,
        "expected_logprobs": expected_logprobs,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
    }
    
    return results


def run_logprob_check(model, prompt_tokens, response_token_ids, response_logprobs):
    """
    Main function to run logprob checking.
    
    Args:
        model: The language model
        prompt_tokens: List of prompt token IDs
        response_token_ids: Numpy array of response token IDs from API
        response_logprobs: Numpy array of response logprobs from API
    """
    # Combine prompt and response tokens
    tokens_list = prompt_tokens + response_token_ids.tolist()
    
    # Create batched tokens (replicate for batch dimension)
    batch_size = 4
    tokens = np.array([tokens_list] * batch_size, dtype=np.int32).reshape(batch_size, -1)
    tokens = hax.named(tokens, ["batch", "position"])
    
    # Check logprobs match
    results = check_logprobs_match(model, tokens, response_logprobs, len(prompt_tokens))
    
    print("Response token ids: ", response_token_ids)
    print("Response logprobs (expected): ", response_logprobs)
    print("Response logprobs (model): ", results["model_logprobs"])
    print("Response logprobs mean difference: ", results["mean_diff"])
    print("Response logprobs max difference: ", results["max_diff"])
    
    return results


if __name__ == "__main__":
    """
    Run logprob checking on an example prompt.
    
    Usage:
        python -m levanter.main.inference_repl --checkpoint meta-llama/Llama-3.2-1B-Instruct
        # Then use this script to verify logprobs match
    """
    import argparse
    import equinox as eqx
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning
    
    from levanter.checkpoint import load_checkpoint
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
    from levanter.models.lm_model import LmConfig
    from levanter.models.llama import LlamaConfig
    from levanter.trainer import TrainerConfig
    
    parser = argparse.ArgumentParser(description="Check train vs inference logprobs")
    parser.add_argument("--checkpoint", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace checkpoint to load")
    parser.add_argument("--max-tokens", type=int, default=5,
                        help="Maximum tokens to generate")
    args = parser.parse_args()
    
    # Example prompt tokens (Llama 3 chat format: "Hello, how are you?")
    prompt_tokens = [
        128000, 128006, 9125, 128007, 271,
        38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198,
        15724, 2696, 25, 220, 2705, 4723, 220, 2366, 20, 271,
        128009, 128006, 882, 128007, 271,
        9906, 11, 1268, 527, 499, 30, 128009, 128006, 78191, 128007, 271
    ]
    
    print(f"Loading model from {args.checkpoint}...")
    
    # Set up configuration
    model_config = LlamaConfig()
    trainer_config = TrainerConfig()
    
    # Load tokenizer and model
    tokenizer = load_tokenizer(args.checkpoint)
    vocab_size = len(tokenizer)
    
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer_config.compute_axis_mapping)
        
        print("Loading HF checkpoint...")
        converter = HFCheckpointConverter(
            type(model_config),
            reference_checkpoint=args.checkpoint,
            tokenizer=tokenizer,
        )
        
        model = converter.load_pretrained(
            model_config.model_type,
            ref=args.checkpoint,
            dtype=trainer_config.mp.compute_dtype,
            axis_mapping=trainer_config.parameter_axis_mapping,
        )
        
        print("Model loaded successfully!")
        
        # Run simple autoregressive inference
        print(f"\nGenerating {args.max_tokens} tokens...")
        # Replicate batch to match data parallelism (typically 4)
        batch_size = 4
        prompt_array = np.array([prompt_tokens] * batch_size, dtype=np.int32).reshape(batch_size, -1)
        prompt_named = hax.named(prompt_array, ["batch", "position"])
        
        generated = simple_autoregressive_inference(
            model, prompt_named, args.max_tokens, jax.random.PRNGKey(0)
        )
        
        print(f"Generated tokens shape: {generated.shape}")
        print(f"Generated tokens: {generated.array[0][len(prompt_tokens):].tolist()}")
        print(f"Generated string: {tokenizer.decode(generated.array[0][len(prompt_tokens):])}")
        
        # If you have API response tokens and logprobs, you can check them:
        # response_token_ids = np.array([...], dtype=np.int32)
        # response_logprobs = np.array([...], dtype=np.float32)
        # results = run_logprob_check(model, prompt_tokens, response_token_ids, response_logprobs)
