"""Validate an explicitly required installed attention-helper source before startup."""

import hashlib
from importlib import metadata
from pathlib import Path
import re

from omegaconf import DictConfig

ATTENTION_HELPER_PATH = Path("vllm/v1/attention/ops/triton_attention_helpers.py")


def validate_vllm_source_config(generator: DictConfig) -> None:
    expected = generator.required_vllm_attention_helper_sha256
    if expected is None:
        return
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("generator.required_vllm_attention_helper_sha256 requires a lowercase SHA256")
    if generator.backend != "vllm" or not generator.run_engines_locally:
        raise ValueError("Required vLLM helper source validation requires local vLLM inference engines")


def verify_vllm_attention_source(expected_sha256: str | None) -> None:
    """Reject an installed helper that differs from the recipe's qualified source."""
    if expected_sha256 is None:
        return
    helper = metadata.distribution("vllm").locate_file(ATTENTION_HELPER_PATH)
    actual_sha256 = hashlib.sha256(helper.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Installed vLLM attention helper differs from the qualified recipe: "
            f"expected {expected_sha256}, actual {actual_sha256}. "
            "Adopt a published, pinned runtime containing the qualified helper; installed source is not modified."
        )
