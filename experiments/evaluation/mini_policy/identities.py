# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable dataset task identities across historical archive schemas."""

import hashlib
import json
from functools import cache


@cache
def triviaqa_manifest_indices():
    # Invert Evalchemy SampleManifest schema 1 for the fixed official split.
    # source_id is the lm-eval dataset row index, not a response-derived ID.
    lookup = {}
    for index in range(17944):
        payload = {"schema_version": 1, "task": "triviaqa", "namespace": "triviaqa", "source_id": index}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        lookup[digest] = index
    return lookup


def document_identity(benchmark, document, fallback):
    if benchmark == "triviaqa":
        value = str(fallback)
        if not value.isdigit():
            value = str(triviaqa_manifest_indices()[value])
        return "dataset-row:" + value
    if benchmark in {
        "bfclparity-pi",
        "bixbench-pi",
        "tau3-pi",
        "ot-tblite-recovery",
        "tb2-recovery",
        "ds-1000-local",
        "swebench-verified",
        "SOTOPIA-hard",
    }:
        value = str(fallback)
        # Preserve repo__issue SWE IDs. Only strip an explicit HF namespace.
        value = value.split("__", 1)[-1] if "/" in value and "__" in value else value
        # Registry Git tasks expose a basename; package archives include the dataset namespace.
        if benchmark == "ds-1000-local" and "/" not in value:
            return "ds-1000/" + value
        return value
    if document:
        if benchmark in {"cruxeval", "humanevalplus"}:
            identity = document.get("task_id") or document.get("id")
            if identity:
                return str(identity)
        clean = {key: value for key, value in document.items() if key not in {"cruxeval_task"}}
        return (
            "doc:"
            + hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        )
    raise ValueError(f"no dataset document for stable identity: {benchmark} / {fallback}")
