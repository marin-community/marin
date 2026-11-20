"""Context-parallel communication constraints for Grug's mixed attention layers."""


def grug_cp_communication(long_layers: tuple[bool, ...], *, hero: bool) -> list[str]:
    """Return the bridge's per-layer defaults before launch overrides."""
    return ["a2a" if hero and not is_long else "p2p" for is_long in long_layers]


def validate_cp_attention(
    *, query_heads: int, kv_heads: int, tensor_parallel: int,
    context_parallel: int, communication: str, sliding_window: bool,
) -> None:
    """Reject incompatible TE communication before model allocation."""
    if query_heads % tensor_parallel or kv_heads % tensor_parallel:
        raise ValueError("Attention head counts must divide tensor parallelism")
    if context_parallel == 1:
        return
    if sliding_window and communication not in {"all_gather", "a2a"}:
        raise ValueError(f"TE sliding-window CP does not support {communication}")
    if communication == "a2a" and (
        (query_heads // tensor_parallel) % context_parallel
        or (kv_heads // tensor_parallel) % context_parallel
    ):
        raise ValueError(
            f"TE a2a requires TP-local Q/KV heads divisible by CP: "
            f"Q={query_heads}, KV={kv_heads}, TP={tensor_parallel}, CP={context_parallel}"
        )
