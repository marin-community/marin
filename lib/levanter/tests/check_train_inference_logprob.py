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

def logsumexp_batch_invariant(x: jax.Array, axis: int = -1) -> jax.Array:
    assert axis == -1 or axis == len(x.shape) - 1, "This implementation of logsumexp only supports last axis"
    x_2d = x.reshape(-1, x.shape[-1])
    logprobs_individual = []
    for i in range(x_2d.shape[0]):
        logprobs_individual.append(jax.jit(jax.nn.logsumexp)(x_2d[i]))
    return jnp.array(logprobs_individual).reshape(x.shape[:-1])

def softmax_cross_entropy_with_integer_labels_batch_invariant(logits: jax.Array, labels: jax.Array, axis: int = -1) -> jax.Array:
    label_logits = jnp.take_along_axis(
        logits, jnp.expand_dims(labels, axis), axis=axis
    ).take(0, axis=axis)
    print(f'{logits.shape=}')
    log_normalizers = logsumexp_batch_invariant(logits, axis=axis)
    out = log_normalizers - label_logits
    return out

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
    response_logits_list = []
    
    for i in range(max_new_tokens):
        # Run model on full sequence (no KV cache)
        print(f'[decode {i}] current tokens: {current_tokens}')
        logits = model(input_ids=current_tokens, attn_mask=AttentionMask.causal(), key=key)
        print(f'[decode {i}] decode logits: {logits}')
        
        # Get logits for last position using Haliax axis indexing
        logits_position_axis = logits.axes[1]
        next_token_logits = logits[logits_position_axis, -1]  # Select last position: [batch, vocab]
        
        # Sample next token (greedy for simplicity)
        next_token = jnp.argmax(next_token_logits.array, axis=-1, keepdims=True)  # [batch, 1]
        
        # Compute logprob using the same method as line 142
        next_token_squeezed = next_token.squeeze(-1)  # [batch]
        print(f"[decode {i}] {next_token_squeezed=}")
        logprob = softmax_cross_entropy_with_integer_labels_batch_invariant(
            next_token_logits.array.astype(jnp.float32), next_token_squeezed
        )
        logprob = -1 * logprob  # Negate to get log probability
        response_logprobs_list.append(logprob)
        response_logits_list.append(next_token_logits.array.astype(jnp.float32))
        
        # Concatenate with current tokens by working with raw arrays
        current_array = current_tokens.array  # [batch, seq_len]
        new_array = jnp.concatenate([current_array, next_token], axis=-1)  # [batch, seq_len+1]
        
        # Rewrap as NamedArray with updated position axis
        new_position_axis = position_axis.resize(new_array.shape[-1])
        current_tokens = hax.named(new_array, [batch_axis, new_position_axis])
    
    # Stack logprobs into a single array [batch, max_new_tokens]
    response_logprobs = jnp.stack(response_logprobs_list, axis=-1)
    response_logits = jnp.stack(response_logits_list, axis=1)

    print(f"{response_logits.shape=}")
    
    return current_tokens, response_logprobs, response_logits


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
    from jax.sharding import Mesh
    
    from levanter.checkpoint import load_checkpoint
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
    from levanter.models.lm_model import LmConfig
    from levanter.models.llama import LlamaConfig
    
    parser = argparse.ArgumentParser(description="Check train vs inference logprobs")
    parser.add_argument("--checkpoint", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace checkpoint to load")
    parser.add_argument("--max-tokens", type=int, default=5,
                        help="Maximum tokens to generate")
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    
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
    
    # Create a simple mesh with all devices as replicas (no data parallelism requirement)
    # This allows batch_size=1 to work without sharding constraints
    num_devices = jax.device_count()
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(num_devices, 1, 1), axis_names=("replica", "data", "model"))
    axis_mapping = {"batch": "data", "embed": "model"}
    
    with hax.axis_mapping(axis_mapping), mesh:
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), axis_mapping)
        
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
            dtype=jnp.float32,
            axis_mapping=axis_mapping,
        )
        
        print("Model loaded successfully!")
        
        # Run simple autoregressive inference
        print(f"\nGenerating {args.max_tokens} tokens...")
        # Use single batch for simple testing
        batch_size = 1
        prompt_array = np.array([prompt_tokens] * batch_size, dtype=np.int32).reshape(batch_size, -1)
        prompt_named = hax.named(prompt_array, ["batch", "position"])
        prompt_named = hax.shard(prompt_named)
        
        generated, response_logprobs, response_logits = simple_autoregressive_inference(
            model, prompt_named, args.max_tokens, jax.random.PRNGKey(0)
        )

        # Only take the first sequence in the batch
        response_logprobs = response_logprobs[0]
        
        print(f"Generated tokens shape: {generated.shape}")
        print(f"Generated tokens: {generated.array[0][len(prompt_tokens):].tolist()}")
        print(f"Generated string: {tokenizer.decode(generated.array[0][len(prompt_tokens):])}")
        
        # Extract response token ids for comparison
        response_token_ids = generated.array[0][len(prompt_tokens):].tolist()

        input = generated[generated.axes[1], :-1]
        print(f'[forward] input: {input}')
        logits = model(input_ids=input, attn_mask=AttentionMask.causal(), key=jax.random.PRNGKey(0))
        print(f'[forward] logits: {logits}')

        print(f"{logits.array.shape=}")

        # [bsz, seq_len, vocab_size]
        logits = logits.array.astype(jnp.float32)
        labels = generated.array[:, 1:].reshape(batch_size, -1)
        print(f"[forward] {labels=}")
        logprobs = softmax_cross_entropy_with_integer_labels_batch_invariant(logits, labels)
        logprobs = -1 * logprobs
        response_logprobs_levanter = logprobs[0, len(prompt_tokens)-1:]
        # print("Response logprobs levanter: ", logprobs[0, len(prompt_tokens)-1:])

        print("Response token ids: ", response_token_ids)
        print("Response logprobs: ", response_logprobs)
        print("Response logprobs levanter: ", response_logprobs_levanter)
        print("Response logprobs mean difference: ", np.mean(np.abs(response_logprobs - response_logprobs_levanter)))
        print("Response logprobs max difference: ", np.max(np.abs(response_logprobs - response_logprobs_levanter)))

        print("Response logits: ", response_logits[0, ])
        print("Response logits levanter: ", logits[0, len(prompt_tokens)-1:])
        print("Response logits mean difference: ", np.mean(np.abs(response_logits - logits[0, len(prompt_tokens)-1:])))
        print("Response logits max difference: ", np.max(np.abs(response_logits - logits[0, len(prompt_tokens)-1:])))
        
        # print("Prompt and response tokens: ", prompt_tokens + response_token_ids)
