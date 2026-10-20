"""Protocols shared by the RLVR ingestion core and SkyRL verifier environments."""

from __future__ import annotations

from typing import Any, Protocol


class VerifierDataContract(Protocol):
    """Public verifier operations required while preparing a dataset."""

    env_id: str
    prompt_instruction: str | None

    def normalize_ground_truth(self, ground_truth: Any) -> str:
        """Return the canonical verifier input for one row."""

    def validate_example(self, ground_truth: Any, positive_response: str, negative_response: str) -> str:
        """Validate ground truth against known satisfying and failing responses."""
