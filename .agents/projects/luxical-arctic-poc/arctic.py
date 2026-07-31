# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Arctic teacher loading for the Luxical POC."""

from collections.abc import Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from luxical.teacher_embedder import EmbedderArctic2M
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

TEACHER_CODE_FILES = (
    "config.json",
    "configuration_hf_alibaba_nlp_gte.py",
    "modeling_hf_alibaba_nlp_gte.py",
)


def reset_arctic_nonpersistent_buffers(model: torch.nn.Module) -> None:
    """Rebuild deterministic buffers omitted by the Transformers meta loader."""
    config = model.config  # type: ignore[attr-defined]
    if config.rope_scaling is not None:
        raise ValueError(f"Unsupported Arctic RoPE scaling: {config.rope_scaling}")

    embeddings = model.embeddings  # type: ignore[attr-defined]
    maximum_positions = int(config.max_position_embeddings)
    position_ids = torch.arange(maximum_positions)
    embeddings.register_buffer("position_ids", position_ids, persistent=False)

    rotary = embeddings.rotary_emb
    head_dimension = int(config.hidden_size // config.num_attention_heads)
    if rotary.dim != head_dimension:
        raise ValueError(f"Unexpected Arctic RoPE dimension: {rotary.dim}")
    inverse_frequency = 1.0 / (
        float(config.rope_theta) ** (torch.arange(0, head_dimension, 2, dtype=torch.float32) / head_dimension)
    )
    positions = torch.arange(maximum_positions, dtype=torch.float32)
    frequencies = torch.einsum("i,j->ij", positions, inverse_frequency)
    phases = torch.cat((frequencies, frequencies), dim=-1)
    rotary.max_seq_len_cached = maximum_positions
    rotary.register_buffer("inv_freq", inverse_frequency, persistent=False)
    rotary.register_buffer("cos_cached", phases.cos(), persistent=False)
    rotary.register_buffer("sin_cached", phases.sin(), persistent=False)

    if not torch.equal(embeddings.position_ids, position_ids):
        raise ValueError("Arctic position IDs were not initialized correctly")
    for name, buffer in (
        ("inv_freq", rotary.inv_freq),
        ("cos_cached", rotary.cos_cached),
        ("sin_cached", rotary.sin_cached),
    ):
        if not torch.isfinite(buffer).all():
            raise ValueError(f"Arctic RoPE buffer {name} contains non-finite values")


class PinnedArcticEmbedder(EmbedderArctic2M):
    """Arctic embedder loaded from an exact Hugging Face revision."""

    def __init__(self, model_id: str, revision: str, max_seq_len: int) -> None:
        for filename in TEACHER_CODE_FILES:
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
            )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            attn_implementation="eager",
            unpad_inputs=False,
            use_memory_efficient_attention=False,
        )
        reset_arctic_nonpersistent_buffers(self.model)
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError(f"Arctic parameter {name} contains non-finite values")
        self.device: str | torch.device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError(f"Expected a fast tokenizer, got {type(tokenizer).__name__}")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def _tokenize(
        self,
        texts: Sequence[str],
        prefix: str,
        max_seq_len: int | None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize on the worker thread without starting CUDA work."""
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        inputs = self.tokenizer(
            prefixed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        return dict(inputs)

    @torch.inference_mode()
    def _embed_batch(
        self,
        inputs: dict[str, torch.Tensor],
        mrl: bool = False,
        scalar_quantize_with_limit: float | None = None,
    ) -> np.ndarray:
        """Move inputs to CUDA and run inference on the same thread."""
        device_inputs = {name: value.to(self.device) for name, value in inputs.items()}
        batch_size, sequence_length = device_inputs["input_ids"].shape
        device_inputs["position_ids"] = (
            torch.arange(sequence_length, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        )
        return super()._embed_batch(
            device_inputs,
            mrl=mrl,
            scalar_quantize_with_limit=scalar_quantize_with_limit,
        )
