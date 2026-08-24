# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.grug.attention._core import (
    AttentionMask as AttentionMask,
    GrugAttentionImplementation as GrugAttentionImplementation,
    RotaryConfig as RotaryConfig,
    ThdSegmentMetadata as ThdSegmentMetadata,
    align_kv_heads as align_kv_heads,
    apply_rotary_embedding as apply_rotary_embedding,
    attention as attention,
    reference_attention as reference_attention,
    thd_segment_metadata_from_segment_ids as thd_segment_metadata_from_segment_ids,
)
from levanter.grug.attention._fa4_cute import fa4_cute_segment_bounds as fa4_cute_segment_bounds
from levanter.grug.attention._fa4_cute import gpu_fa4_cute_attention as gpu_fa4_cute_attention
from levanter.grug.attention._fa4_thd import gpu_fa4_thd_attention as gpu_fa4_thd_attention

# ``_te_cp`` stays out of this list: ``attention()`` imports it lazily, and only its config is eager.
from levanter.grug.attention._te_cp_config import (
    ContextParallelStrategy as ContextParallelStrategy,
    TeContextParallelConfig as TeContextParallelConfig,
)
