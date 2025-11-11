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

from levanter.layers.attention import AttentionMask, AttentionBackend


def simple_autoregressive_inference(model, prompt_tokens, max_new_tokens, key):
    """
    Simple autoregressive inference with no KV cache.
    
    Args:
        model: The language model
        prompt_tokens: Initial prompt tokens as a NamedArray with ["batch", "position"] axes
        max_new_tokens: Maximum number of tokens to generate
        key: JAX random key for sampling
        
    Returns:
        Tuple of (generated tokens, response logprobs)
        - Generated tokens (including prompt) as a NamedArray
        - Response logprobs as a JAX array of shape [batch, max_new_tokens]
    """
    current_tokens = prompt_tokens
    batch_axis = current_tokens.axes[0]
    position_axis = current_tokens.axes[1]
    response_logprobs_list = []
    
    for _ in range(max_new_tokens):
        # Run model on full sequence (no KV cache)
        logits = model(input_ids=current_tokens, attn_mask=AttentionMask.causal(), key=key)
        
        # Get logits for last position using Haliax axis indexing
        logits_position_axis = logits.axes[1]
        next_token_logits = logits[logits_position_axis, -1]  # Select last position: [batch, vocab]
        
        # Sample next token (greedy for simplicity)
        next_token = jnp.argmax(next_token_logits.array, axis=-1, keepdims=True)  # [batch, 1]
        
        # Compute logprob using the same method as line 142
        next_token_squeezed = next_token.squeeze(-1)  # [batch]
        logprob = optax.softmax_cross_entropy_with_integer_labels(
            next_token_logits.array.astype(jnp.float32), next_token_squeezed
        )
        logprob = -1 * logprob  # Negate to get log probability
        response_logprobs_list.append(logprob)
        
        # Concatenate with current tokens by working with raw arrays
        current_array = current_tokens.array  # [batch, seq_len]
        new_array = jnp.concatenate([current_array, next_token], axis=-1)  # [batch, seq_len+1]
        
        # Rewrap as NamedArray with updated position axis
        new_position_axis = position_axis.resize(new_array.shape[-1])
        current_tokens = hax.named(new_array, [batch_axis, new_position_axis])
    
    # Stack logprobs into a single array [batch, max_new_tokens]
    response_logprobs = jnp.stack(response_logprobs_list, axis=-1)
    
    return current_tokens, response_logprobs


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
    
    print(f"Loading model from {args.checkpoint}...")
    
    # Set up configuration
    trainer_config = TrainerConfig()
    
    # Load tokenizer first to encode the prompt
    tokenizer = load_tokenizer(args.checkpoint)

    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/discussions/14#679229feb7c3dc07f41e213b
    prompt_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        date_string="06 Nov 2025"
    )
    
    vocab_size = len(tokenizer)
    
    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer_config.compute_axis_mapping)
        
        print("Loading HF checkpoint...")
        converter = HFCheckpointConverter(
            LlamaConfig,
            reference_checkpoint=args.checkpoint,
            tokenizer=tokenizer,
        )
        
        # Load the config from checkpoint and override attn_backend to VANILLA
        hf_config = converter.hf_config_from_hf_checkpoint(args.checkpoint)
        model_config = converter.config_from_hf_config(hf_config, overrides={"attn_backend": AttentionBackend.VANILLA})
        
        model = converter.load_pretrained(
            model_config.model_type,
            ref=args.checkpoint,
            config=model_config,
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
        
        generated, response_logprobs = simple_autoregressive_inference(
            model, prompt_named, args.max_tokens, jax.random.PRNGKey(0)
        )

        # Only take the first sequence in the batch
        response_logprobs = response_logprobs[0]
        
        print(f"Generated tokens shape: {generated.shape}")
        print(f"Generated tokens: {generated.array[0][len(prompt_tokens):].tolist()}")
        print(f"Generated string: {tokenizer.decode(generated.array[0][len(prompt_tokens):])}")
        
        # Extract response token ids for comparison
        response_token_ids = generated.array[0][len(prompt_tokens):].tolist()

        logits = model(input_ids=generated, attn_mask=AttentionMask.causal(), key=jax.random.PRNGKey(0))

        # [bsz, seq_len, vocab_size]
        logits = logits.array.astype(jnp.float32)[:, :-1, :].reshape(4, -1, logits.array.shape[-1])
        labels = generated.array[:, 1:].reshape(4, -1)
        logprobs = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        logprobs = -1 * logprobs
        response_logprobs_levanter = logprobs[0, len(prompt_tokens)-1:]
        # print("Response logprobs levanter: ", logprobs[0, len(prompt_tokens)-1:])

        print("Response token ids: ", response_token_ids)
        print("Response logprobs: ", response_logprobs)
        print("Response logprobs levanter: ", response_logprobs_levanter)
        print("Response logprobs mean difference: ", np.mean(np.abs(response_logprobs - response_logprobs_levanter)))
        print("Response logprobs max difference: ", np.max(np.abs(response_logprobs - response_logprobs_levanter)))
        # print("Prompt and response tokens: ", prompt_tokens + response_token_ids)
