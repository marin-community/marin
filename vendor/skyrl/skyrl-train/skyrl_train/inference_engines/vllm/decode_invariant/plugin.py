"""The ``vllm.general_plugins`` entry point that makes vLLM's CUSTOM attention backend the decode-invariant one."""

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

DECODE_INVARIANT_ATTENTION_BACKEND_CLASS = (
    "skyrl_train.inference_engines.vllm.decode_invariant.attention_backend.DecodeInvariantAttentionBackend"
)


def register_attention_backend() -> None:
    register_backend(AttentionBackendEnum.CUSTOM, DECODE_INVARIANT_ATTENTION_BACKEND_CLASS)
