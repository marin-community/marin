import os

os.environ["JAX_PLATFORMS"] = "cpu"

import torch  # noqa
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed  # type:ignore


def main():
    model_id = "erfanzar/Marin-8B-Instruct-eformat"
    max_decode_length = 4096
    max_prefill_length = 4096

    max_length = max_prefill_length + max_decode_length

    processor = AutoTokenizer.from_pretrained(model_id)
    processor.pad_token_id = processor.eos_token_id
    processor.push_to_hub(model_id)

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=lax.Precision.HIGH,
        auto_shard_model=False,
        sharding_axis_dims=(1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            attn_dtype=jnp.float32,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        cache_dir="/dev/shm/marin",
    )

    model.to_torch().to(torch.bfloat16).push_to_hub(model_id)


if __name__ == "__main__":
    main()
