# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical tokenizer references for midtraining preflight.

Equality is by ``(key, hf_repo, revision)`` plus an optional content
fingerprint. Tokenizer-name strings alone do not protect against the BOS-
missing cache class of bugs (a us-central1 ``4plus`` cache predated the
Llama-3 BOS fix and was silently reused). Preflight also samples BOS tokens
from a cache before trusting a tokenizer match.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerRef:
    """Pinned tokenizer identity for cross-component equality checks.

    Args:
        key: Stable short key, e.g. ``"llama3"`` or ``"qwen3"``. Used in
            manifests and base registry entries to compare without coupling
            to HF repo strings.
        hf_repo: HuggingFace repository id.
        revision: Pinned HF revision (commit sha).
        bos_token_id: Expected BOS id; preflight samples cached sequences
            and refuses launch if the prefix does not match.
        eos_token_id: Expected EOS id.
        vocab_size: Expected vocab size (post any padding).
        fingerprint: Optional sha256 over ``tokenizer.json`` or canonical
            serialization. ``None`` means "do not compare beyond revision".
    """

    key: str
    hf_repo: str
    revision: str
    bos_token_id: int
    eos_token_id: int
    vocab_size: int
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("TokenizerRef.key must be non-empty")
        if not self.hf_repo:
            raise ValueError("TokenizerRef.hf_repo must be non-empty")
        if not self.revision:
            raise ValueError("TokenizerRef.revision must be non-empty; pin a commit sha")
        if self.vocab_size <= 0:
            raise ValueError(f"TokenizerRef.vocab_size must be positive, got {self.vocab_size!r}")
        if self.bos_token_id < 0:
            raise ValueError(f"TokenizerRef.bos_token_id must be non-negative, got {self.bos_token_id!r}")
        if self.eos_token_id < 0:
            raise ValueError(f"TokenizerRef.eos_token_id must be non-negative, got {self.eos_token_id!r}")


# Llama-3.1 tokenizer — the actual tokenizer string baked into every Delphi
# .executor_info ("meta-llama/Meta-Llama-3.1-8B"). Llama-3 and Llama-3.1 share
# the same vocabulary (BOS 128000, EOS 128001, vocab 128256); only the model
# weights differ. Data caches must start with the BOS prefix [128000, ...]
# (the missing-BOS bug fixed in 2026-04 hit the older us-central1 cache).
LLAMA3_TOKENIZER = TokenizerRef(
    key="llama3",
    hf_repo="meta-llama/Meta-Llama-3.1-8B",
    revision="0e9e39f249a16976918f6564b8830bc894c89659",
    bos_token_id=128_000,
    eos_token_id=128_001,
    vocab_size=128_256,
)

# Qwen3 tokenizer for the Qwen3-architecture bases. Provided for completeness;
# Delphi cooldown/CPT bases currently use Llama-3.
QWEN3_TOKENIZER = TokenizerRef(
    key="qwen3",
    hf_repo="Qwen/Qwen3-0.6B",
    revision="main",
    bos_token_id=151_643,
    eos_token_id=151_645,
    vocab_size=151_936,
)

_CANONICAL_TOKENIZERS: dict[str, TokenizerRef] = {
    LLAMA3_TOKENIZER.key: LLAMA3_TOKENIZER,
    QWEN3_TOKENIZER.key: QWEN3_TOKENIZER,
}


def get_tokenizer(key: str) -> TokenizerRef:
    """Return a canonical :class:`TokenizerRef` by short key."""
    try:
        return _CANONICAL_TOKENIZERS[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_CANONICAL_TOKENIZERS))
        raise ValueError(f"Unknown tokenizer key {key!r}. Registered: {allowed}.") from exc


def assert_tokenizer_compatible(left: TokenizerRef, right: TokenizerRef) -> None:
    """Refuse mismatched tokenizers between two artifacts (base, cache, override).

    Equality is by ``(key, hf_repo, revision)``. ``vocab_size`` and BOS/EOS
    must also agree. If both sides carry a ``fingerprint``, those must match.
    """
    if left.key != right.key:
        raise ValueError(f"Tokenizer key mismatch: {left.key!r} vs {right.key!r}")
    if left.hf_repo != right.hf_repo:
        raise ValueError(f"Tokenizer hf_repo mismatch: {left.hf_repo!r} vs {right.hf_repo!r}")
    if left.revision != right.revision:
        raise ValueError(f"Tokenizer revision mismatch for {left.key}: {left.revision!r} vs {right.revision!r}")
    if left.vocab_size != right.vocab_size:
        raise ValueError(f"Tokenizer vocab_size mismatch for {left.key}: {left.vocab_size} vs {right.vocab_size}")
    if left.bos_token_id != right.bos_token_id:
        raise ValueError(f"Tokenizer bos_token_id mismatch for {left.key}: {left.bos_token_id} vs {right.bos_token_id}")
    if left.eos_token_id != right.eos_token_id:
        raise ValueError(f"Tokenizer eos_token_id mismatch for {left.key}: {left.eos_token_id} vs {right.eos_token_id}")
    if left.fingerprint and right.fingerprint and left.fingerprint != right.fingerprint:
        raise ValueError(f"Tokenizer fingerprint mismatch for {left.key}")
