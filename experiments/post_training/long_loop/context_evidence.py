# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a dry-run to CPU validation performed by the pinned vLLM runtime."""

import hashlib
import json
from pathlib import Path


def validate_context_evidence(config: dict, overrides: dict, window: int, evidence: dict) -> None:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if evidence["vllm_commit"] != "39e62869693c46402b1a95fde4fc55ca1aab9ae1":
        raise ValueError("Context proof uses a different vLLM pin")
    for case in evidence["cases"]:
        if (case["config_digest"], case["overrides"], case["requested_window"]) == (digest, overrides, window):
            if not case["accepted"] or case["max_model_len"] < window:
                raise ValueError("Pinned vLLM rejected this serving context")
            return
    raise ValueError("Exact serving inputs lack a pinned vLLM CPU context proof")


def context_evidence() -> dict:
    return json.loads(Path(__file__).with_name("context-proof.json").read_text())
