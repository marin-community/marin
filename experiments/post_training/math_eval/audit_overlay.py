# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind audit verdicts to immutable pool and verifier identities."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.post_training.math_eval.pool import canonical_json

VERIFIER_REVISION = "fa6aae365f70e50bca5218ed2f3d49067e6dcc26"
VERIFIER_SOURCES_SHA256 = "4b7cfd93db34c7952224cf4220e2978accd694b724438860d1f76fa830eda207"


def validated_statuses(
    manifest: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], overlay: Mapping[str, Any]
) -> tuple[Mapping[str, str], str]:
    """Refuse foreign, partial or unknown verdicts; only acceptance permits sampling."""
    if (
        overlay.get("manifest_sha256") != selection["manifest_sha256"]
        or overlay.get("verifier_revision") != VERIFIER_REVISION
        or overlay.get("verifier_sources_sha256") != VERIFIER_SOURCES_SHA256
        or not re.fullmatch(r"[0-9a-f]{64}", str(overlay.get("audit_source_sha256", "")))
    ):
        raise ValueError("Audit overlay is not bound to this pool and verifier")
    statuses = overlay.get("statuses", {})
    if set(statuses) != {row["prompt_sha256"] for row in manifest}:
        raise ValueError("Audit overlay must cover exactly the frozen pool")
    if not set(statuses.values()) <= {"accept", "reject", "flag", "pending"}:
        raise ValueError("Unknown audit status")
    return statuses, hashlib.sha256(canonical_json(overlay).encode()).hexdigest()
