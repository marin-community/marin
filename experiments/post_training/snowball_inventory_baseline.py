# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content identities for exact initializer evaluation artifacts."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_VERSION = "2026.10.21"


@dataclass(frozen=True)
class BaselineIdentity:
    model_uri: str
    model_manifest_identity: str
    tokenizer_uri: str
    tokenizer_revision: str
    tokenizer_template_sha256: str
    panel_sha256: str
    family_counts: dict[str, int]
    evaluation_recipe: dict
    recovery_policy: dict
    runtime_tree_hash: str
    runtime_lock_sha256: str
    scoring_source_sha256: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scoring_source_sha256() -> str:
    """Bind preparation, decoding, grading, pairing and recovery source bytes."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    sources = (
        "snowball_inventory_evaluate.py",
        "snowball_inventory_evaluate_runtime.py",
        "snowball_inventory_probe.py",
        "snowball_inventory_panel.py",
    )
    for name in sources:
        source = root / name
        digest.update(source.name.encode() + b"\0" + source.read_bytes() + b"\0")
    return digest.hexdigest()
