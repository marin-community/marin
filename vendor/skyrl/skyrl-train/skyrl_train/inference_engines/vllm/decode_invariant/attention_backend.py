"""vLLM's CUSTOM attention backend for a decode-invariant engine: each query row's output equals a one-row decode
step of that row."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec

from skyrl_train.inference_engines.vllm.decode_invariant.engine import require_grug_row_invariant_gemm
from skyrl_train.inference_engines.vllm.decode_invariant.fa3 import (
    FA3_HEAD_DIM,
    FLASH_ATTN_VERSION,
    HOPPER_COMPUTE_CAPABILITY_MAJOR,
    fa3_window_requests,
)


@dataclass
class DecodeInvariantMetadata(FlashAttentionMetadata):
    """A step's FA3 metadata, plus one copy per sliding-window size in which every row past the window is its own
    request. The step's first layer with that window size fills the entry."""

    window_requests: dict[int, FlashAttentionMetadata] = field(default_factory=dict)


class DecodeInvariantMetadataBuilder(FlashAttentionMetadataBuilder):
    # Full CUDA graphs cover one-row decode steps; a step with longer requests runs attention eagerly.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ) -> None:
        require_grug_row_invariant_gemm()
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "a decode-invariant engine runs vLLM's model runner V2, whose log-probability kernel the exact "
                "trainer runs; vLLM chose the V1 runner"
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> DecodeInvariantMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # One split, planned by FA3 itself: each row reduces over its key tiles in one pass, whatever the step holds.
        values = {item.name: getattr(metadata, item.name) for item in fields(metadata)}
        return DecodeInvariantMetadata(**{**values, "max_num_splits": 1, "scheduler_metadata": None})

    def update_block_table(
        self, metadata: DecodeInvariantMetadata, blk_table: torch.Tensor, slot_mapping: torch.Tensor
    ) -> DecodeInvariantMetadata:
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping, window_requests={})

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class DecodeInvariantAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vllm_flash_attn_version != FLASH_ATTN_VERSION:
            raise RuntimeError(
                f"a decode-invariant engine runs FA{FLASH_ATTN_VERSION}; vLLM selected FA{self.vllm_flash_attn_version} "
                f"for {torch.cuda.get_device_name()}"
            )
        if self.alibi_slopes is not None:
            raise NotImplementedError("a decode-invariant engine runs attention without ALiBi")
        left, right = self.sliding_window
        self.window: int | None = left + 1 if left >= 0 and right == 0 else None

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DecodeInvariantMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            attn_metadata is not None
            and self.window is not None
            and attn_metadata.max_query_len > 1
            and attn_metadata.max_seq_len > self.window
        ):
            attn_metadata = self._window_requests(attn_metadata)
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
        )

    def _window_requests(self, metadata: DecodeInvariantMetadata) -> FlashAttentionMetadata:
        """The step's metadata with each row at a position past this layer's window as a request of its own."""
        if self.window not in metadata.window_requests:
            requests = metadata.query_start_loc.numel() - 1
            planned = fa3_window_requests(
                metadata.query_start_loc.tolist(), metadata.seq_lens[:requests].tolist(), self.window
            )
            device = metadata.query_start_loc.device
            metadata.window_requests[self.window] = replace(
                metadata,
                max_query_len=int(planned.query_start.diff().max()),
                query_start_loc=planned.query_start.to(device),
                max_seq_len=int(planned.key_lengths.max()),
                seq_lens=planned.key_lengths.to(device),
                block_table=metadata.block_table[planned.owner.to(device)],
                window_requests={},
            )
        return metadata.window_requests[self.window]


class DecodeInvariantAttentionBackend(FlashAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return AttentionBackendEnum.CUSTOM.name

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == HOPPER_COMPUTE_CAPABILITY_MAJOR

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == FA3_HEAD_DIM

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type[DecodeInvariantAttentionImpl]:
        return DecodeInvariantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[DecodeInvariantMetadataBuilder]:
        return DecodeInvariantMetadataBuilder
