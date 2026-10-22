"""Describe Bridge conversion tasks for bounded expert-block publication.

This adapter owns Bridge mapping names, fused layouts and HF expert schemas.
It returns live source parameters and normalized publication records at TP=ETP=1;
the schedule and storage-view layers do not interpret Bridge tasks.
"""

import re

from skyrl_train.weight_sync.expert_block.schedule import DenseSlice, WIRE_DTYPE_BYTES
from skyrl_train.weight_sync.expert_block.source_views import ExpertSlice, LocalSources, dtype_name

EXPERT_HF_NAME = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?:(?P<expert>\d+)\.)?(?P<part>gate|up|down)_proj\.weight"
)
EXPERT_SOURCE_KEY = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc[12]\.weight(\d+)")
EXPERT_MAPPINGS = ("GrugStackedExpertMapping", "GrugStackedGatedExpertMapping")
WIRE_DTYPES = frozenset(WIRE_DTYPE_BYTES)


def local_source_slices(tasks, config, *, pp: int) -> LocalSources:
    """Describe supported Bridge layouts as expert identities and dense storage runs.

    Accept whole, gated, QKV and Grug expert mappings at TP=ETP=1. Expert
    schemas are validated and reduced to layer/expert/part identities here.
    Dense runs preserve HF offsets; sources retain the original parameter
    objects so later ``param.data`` reassignment remains visible.
    """
    expert, dense, sources = [], [], {}
    for task in tasks:
        # Keep the parameter, not a detached view, so the sender's storage check sees a
        # reassigned ``param.data``.
        source = task.param_weight
        if source is None:
            continue
        key = task.global_param_name
        dtype = dtype_name(source.dtype)
        if key in sources or not source.is_contiguous() or dtype not in WIRE_DTYPES:
            raise ValueError(f"Parameter {key} must be unique, contiguous and BF16 or FP32")
        sources[key] = source
        mapping = task.mapping
        kind = type(mapping).__name__
        # For an expert mapping, tp_size is the expert-tensor-parallel size.
        if mapping.tp_size != 1:
            raise ValueError(f"Parameter {key} is tensor-parallel; expert-block sync requires trainer TP=1 and ETP=1")
        if kind in (*EXPERT_MAPPINGS, "GatedMLPMapping", "AutoMapping"):
            gated = kind in ("GrugStackedGatedExpertMapping", "GatedMLPMapping")
            names = mapping.hf_param if gated else {"down": mapping.hf_param}
            matches = {part: EXPERT_HF_NAME.fullmatch(name) for part, name in names.items()}
            stacked = kind in EXPERT_MAPPINGS
            source_match = EXPERT_SOURCE_KEY.fullmatch(key)
            # Recognize expert source keys so malformed HF names cannot become dense slices.
            if (
                stacked
                or source_match is not None
                or any(hf is not None and hf["expert"] is not None for hf in matches.values())
            ):
                expected_parts = {"gate", "up"} if gated else {"down"}
                if set(names) != expected_parts or any(
                    hf is None or hf["part"] != part or (hf["expert"] is None) != stacked
                    for part, hf in matches.items()
                ):
                    raise ValueError(
                        f"Expert mapping for {key} does not contain matching {sorted(expected_parts)} tensors"
                    )
                if stacked:
                    source_match = re.search(r"\.weight(\d+)$", key)
                if source_match is None or source.ndim != 2 or (gated and source.shape[0] % 2):
                    raise ValueError(f"Expert parameter {key} is not a complete per-expert matrix")
                # Model dimensions are checked by local_expert_sources for both schemas.
                expert_id = int(source_match.groups()[-1])
                if not stacked and (
                    any(
                        int(hf["layer"]) != int(source_match[1]) or int(hf["expert"]) != expert_id
                        for hf in matches.values()
                    )
                    or (".linear_fc1." if gated else ".linear_fc2.") not in key
                ):
                    raise ValueError(f"Split expert mapping for {key} disagrees with its HF tensors")
                expert.extend(
                    ExpertSlice(int(hf["layer"]), expert_id, part, key) for part, hf in sorted(matches.items())
                )
                continue

        def add(name, hf_offset, numel, source_offset):
            if source_offset < 0 or numel <= 0 or source_offset + numel > source.numel():
                raise ValueError(f"Slice of {key} exceeds its storage")
            dense.append(DenseSlice(name, hf_offset, numel, dtype, key, source_offset, pp))

        if kind in ("AutoMapping", "ReplicatedMapping", "RowParallelMapping"):
            # Hero's sconv_k uses a row-parallel mapping. With required TP=1,
            # its local parameter is the complete HF tensor.
            add(mapping.hf_param, 0, source.numel(), 0)
        elif kind == "GatedMLPMapping":
            if source.ndim != 2 or source.shape[0] % 2 or set(mapping.hf_param) != {"gate", "up"}:
                raise ValueError(f"Gated parameter {key} is not a complete [gate;up] matrix")
            half = source.numel() // 2
            for position, part in enumerate(("gate", "up")):
                add(mapping.hf_param[part], 0, half, position * half)
        elif kind == "QKVMapping":
            # Megatron interleaves [q..., k, v] per KV group; HF keeps q, k and v as separate tensors.
            heads, groups, head_dim = config.num_attention_heads, config.num_query_groups, config.kv_channels
            if heads % groups:
                raise ValueError(f"QKV parameter {key}: {heads} heads do not split across {groups} KV groups")
            queries = heads // groups
            expected_shape = (groups * (queries + 2) * head_dim, config.hidden_size)
            if tuple(source.shape) != expected_shape or set(mapping.hf_param) != {"q", "k", "v"}:
                raise ValueError(f"QKV parameter {key} differs from the configured interleaved layout")
            for group in range(groups):
                for part, offset, count in (("q", 0, queries), ("k", queries, 1), ("v", queries + 1, 1)):
                    numel = count * head_dim * config.hidden_size
                    source_offset = (group * (queries + 2) + offset) * head_dim * config.hidden_size
                    add(mapping.hf_param[part], group * numel, numel, source_offset)
        else:
            raise ValueError(f"Unsupported weight mapping for expert-block sync: {kind}")
    return LocalSources(expert, dense, sources)
