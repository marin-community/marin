"""Well-known tokenizer identifiers and equivalence relationships."""

LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
MARIN_TOKENIZER = "marin-community/marin-tokenizer"
LLAMA3_VOCAB_SIZE = 128_256

_EQUIVALENT_TOKENIZER_GROUPS: list[frozenset[str]] = [
    frozenset({LLAMA3_TOKENIZER, MARIN_TOKENIZER}),
]


def are_tokenizers_known_equivalent(tokenizer1: str, tokenizer2: str) -> bool | None:
    """Returns True if known equivalent, None if unknown (caller should load and compare)."""
    for group in _EQUIVALENT_TOKENIZER_GROUPS:
        if tokenizer1 in group and tokenizer2 in group:
            return True
    return None
