"""Detect Qwen3.5/3.6 multimodal shells for text-only vLLM rollouts."""


def is_qwen3_5_vlm_shell(config) -> bool:
    """True iff ``config`` is a Qwen3.5/3.6 multimodal shell wrapping a text MoE
    tower used for text-only rollouts.

    Detection (matches tmax's ``text_config`` / ``linear_conv_kernel_dim`` probe):
    a nested ``text_config`` carrying the GatedDeltaNet signature
    ``linear_conv_kernel_dim`` — i.e. the hybrid Qwen3.5/3.6 text tower — while
    the top-level config does NOT carry it (it is the shell). We additionally gate
    on the top ``model_type`` starting with ``qwen3_5`` so unrelated VLMs with a
    ``text_config`` are untouched.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return False
    # GatedDeltaNet signature lives on the *text* config for the shell.
    text_is_qwen3_5_gdn = hasattr(text_config, "linear_conv_kernel_dim")
    top_is_qwen3_5 = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    # The top config must be the multimodal shell, not the text tower
    # (i.e. it is genuinely the shell, with the GDN signature one level down).
    top_is_shell = not hasattr(config, "linear_conv_kernel_dim")
    return bool(text_is_qwen3_5_gdn and top_is_qwen3_5 and top_is_shell)
