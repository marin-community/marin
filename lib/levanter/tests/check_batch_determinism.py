# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax
import jax.numpy as jnp
import haliax as hax

from levanter.layers.attention import AttentionMask, AttentionBackend

if __name__ == "__main__":
    import argparse
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning
    from jax.sharding import Mesh
    
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
    from levanter.models.llama import LlamaConfig
    
    parser = argparse.ArgumentParser(description="Check train vs inference logprobs")
    parser.add_argument("--checkpoint", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace checkpoint to load")
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

        batch_size = 4
        prompt_array = np.array([prompt_tokens] * batch_size, dtype=np.int32).reshape(batch_size, -1)
        prompt_named = hax.named(prompt_array, ["batch", "position"])
        
        previous_logits = None
        for i in range(10):
            print(f'[train] input: {prompt_named}')
            logits = model(input_ids=prompt_named, attn_mask=AttentionMask.causal(), key=jax.random.PRNGKey(0))
            print(f'[train] logits: {logits}')
            if previous_logits is not None:
                print('Checking logits match...')
                assert jnp.allclose(logits.array, previous_logits.array)
            previous_logits = logits
