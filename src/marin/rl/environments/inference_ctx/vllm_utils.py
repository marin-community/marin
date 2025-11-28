
def levanter_llama_to_vllm_mapping():
    return {
        "lm_head": ("model.lm_head", (None, "model")),
        "model.embed_tokens": (
            "model.embed.embedding",
            ("model", None),
        ),
        "model.layers.*.input_layernorm": (
            "model.layers.*.input_layernorm.scale",
            (None,),
        ),
        "model.layers.*.mlp.down_proj": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", None),
        ),
        "model.layers.*.mlp.gate_proj": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.mlp.up_proj": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.post_attention_layernorm": (
            "model.layers.*.post_attention_layernorm.scale",
            (None,),
        ),
        "model.layers.*.self_attn.k_proj": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.o_proj": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", None, None),
        ),
        "model.layers.*.self_attn.q_proj": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.v_proj": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "model", None),
        ),
        "model.norm": ("model.norm.scale", (None,)),
    }


def levanter_qwen_to_vllm_mapping():
    mapping = levanter_llama_to_vllm_mapping()
    mapping.update(
        {
            "model.layers.*.self_attn.q_norm": ("model.layers.*.self_attn.q_norm.scale", (None,)),
            "model.layers.*.self_attn.k_norm": ("model.layers.*.self_attn.k_norm.scale", (None,)),
        }
    )
    return mapping


llama_transpose_keys = {
    "lm_head": (1, 0),
    "gate_proj": (1, 0),
    "up_proj": (1, 0),
    "down_proj": (1, 0),
    "q_proj": (2, 0, 1),
    "k_proj": (2, 0, 1),
    "v_proj": (2, 0, 1),
    "o_proj": (1, 2, 0),
}

MODEL_MAPPINGS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "meta-llama/Llama-3.2-1B-Instruct": levanter_llama_to_vllm_mapping(),
    "meta-llama/Llama-3.2-3B-Instruct": levanter_llama_to_vllm_mapping(),
    "Qwen/Qwen3-0.6B": levanter_qwen_to_vllm_mapping(),
    "Qwen/Qwen3-1.7B": levanter_qwen_to_vllm_mapping(),
    "meta-llama/Llama-3.1-8B-Instruct": levanter_llama_to_vllm_mapping(),
    "Qwen/Qwen3-8B": levanter_qwen_to_vllm_mapping(),
}

MODEL_TRANSPOSE_KEYS: dict[str, tuple[int, ...]] = {
    "meta-llama/Llama-3.2-1B-Instruct": llama_transpose_keys,
    "meta-llama/Llama-3.2-3B-Instruct": llama_transpose_keys,
    "Qwen/Qwen3-0.6B": llama_transpose_keys,
    "Qwen/Qwen3-1.7B": llama_transpose_keys,
    "meta-llama/Llama-3.1-8B-Instruct": llama_transpose_keys,
    "Qwen/Qwen3-8B": llama_transpose_keys,
}

def levanter_llama_to_vllm_mapping():
    return {
        "lm_head": ("model.lm_head", (None, "model")),
        "model.embed_tokens": (
            "model.embed.embedding",
            ("model", None),
        ),
        "model.layers.*.input_layernorm": (
            "model.layers.*.input_layernorm.scale",
            (None,),
        ),
        "model.layers.*.mlp.down_proj": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", None),
        ),
        "model.layers.*.mlp.gate_proj": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.mlp.up_proj": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.post_attention_layernorm": (
            "model.layers.*.post_attention_layernorm.scale",
            (None,),
        ),
        "model.layers.*.self_attn.k_proj": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.o_proj": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", None, None),
        ),
        "model.layers.*.self_attn.q_proj": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.v_proj": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "model", None),
        ),
        "model.norm": ("model.norm.scale", (None,)),
    }


def levanter_qwen_to_vllm_mapping():
    mapping = levanter_llama_to_vllm_mapping()
    mapping.update(
        {
            "model.layers.*.self_attn.q_norm": ("model.layers.*.self_attn.q_norm.scale", (None,)),
            "model.layers.*.self_attn.k_norm": ("model.layers.*.self_attn.k_norm.scale", (None,)),
        }
    )
    return mapping


llama_transpose_keys = {
    "lm_head": (1, 0),
    "gate_proj": (1, 0),
    "up_proj": (1, 0),
    "down_proj": (1, 0),
    "q_proj": (2, 0, 1),
    "k_proj": (2, 0, 1),
    "v_proj": (2, 0, 1),
    "o_proj": (1, 2, 0),
}

MODEL_MAPPINGS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "meta-llama/Llama-3.2-1B-Instruct": levanter_llama_to_vllm_mapping(),
    "meta-llama/Llama-3.2-3B-Instruct": levanter_llama_to_vllm_mapping(),
    "Qwen/Qwen3-0.6B": levanter_qwen_to_vllm_mapping(),
    "Qwen/Qwen3-1.7B": levanter_qwen_to_vllm_mapping(),
    "meta-llama/Llama-3.1-8B-Instruct": levanter_llama_to_vllm_mapping(),
    "Qwen/Qwen3-8B": levanter_qwen_to_vllm_mapping(),
}

MODEL_TRANSPOSE_KEYS: dict[str, tuple[int, ...]] = {
    "meta-llama/Llama-3.2-1B-Instruct": llama_transpose_keys,
    "meta-llama/Llama-3.2-3B-Instruct": llama_transpose_keys,
    "Qwen/Qwen3-0.6B": llama_transpose_keys,
    "Qwen/Qwen3-1.7B": llama_transpose_keys,
    "meta-llama/Llama-3.1-8B-Instruct": llama_transpose_keys,
    "Qwen/Qwen3-8B": llama_transpose_keys,
}