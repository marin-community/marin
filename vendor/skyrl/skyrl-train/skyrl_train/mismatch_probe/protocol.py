"""Deterministic sample identities and seeds for mismatch archives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]
    loss_mask: tuple[bool, ...]


def request_seed(master_seed: int, prompt_id: str, repetition: int) -> int:
    """Derive a stable per-answer vLLM seed independent of Python hash salt."""
    payload = f"{master_seed}:{len(prompt_id)}:{prompt_id}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def probe_hash(records: Iterable[ProbeSample]) -> str:
    """Hash the canonical JSON representation of ordered frozen samples."""
    payload = json.dumps(
        [asdict(record) for record in records], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
