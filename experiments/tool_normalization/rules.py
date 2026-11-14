# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identify conservative deterministic mappings implemented by native_seed."""

import hashlib
import json
import re
from pathlib import Path

from experiments.tool_normalization.native_seed import native_seed, zero_based_read_offsets

RULE_VERSION = "2026-11-05-v1"
NATIVE_TOOLS = frozenset({"bash", "read", "write", "edit", "grep", "glob"})


def rule_ids(original: list[dict], packages: list[str]) -> list[str]:
    """Return rules only when every call has a known, schema-checked mapping."""
    if native_seed(original, packages) is None:
        return []
    action_count = sum(
        len(re.findall(r"<tool_call>.*?</tool_call>", message["content"], re.DOTALL))
        for message in original
        if message["role"] == "assistant"
    )
    rules = set()
    for message in original:
        if message["role"] != "assistant":
            continue
        for body in re.findall(r"<tool_call>(.*?)</tool_call>", message["content"], re.DOTALL):
            call = json.loads(body)
            name = call["name"]
            if name in NATIVE_TOOLS:
                if name == "read" and "offset" in call["arguments"] and zero_based_read_offsets(original):
                    rules.add("read-zero-to-one-based-offset")
                else:
                    rules.add(f"native-{name}")
            elif name == "execute_bash":
                rules.add("execute-bash-to-bash")
            elif name == "str_replace_editor":
                rules.add("str-replace-editor-" + call["arguments"]["command"])
            elif name == "evaluate_code":
                if action_count != 1:
                    # Multi-call Python may depend on shared interpreter state.
                    return []
                rules.add("declared-python-stdout-to-bash")
            else:
                # Stateful kernels, custom tools and completion/todo behavior remain reviewed.
                return []
    return sorted(rules)


def rule_record(original: list[dict], packages: list[str]) -> dict:
    ids = rule_ids(original, packages)
    directory = Path(__file__).parent
    return {
        "version": RULE_VERSION,
        "rule_ids": ids,
        "implementation_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "rules.py",
                "native_seed.py",
                "historical_projection.py",
                "serialize.py",
                "candidate.py",
                "opencode-tools.json",
            )
        },
        "requires_glm_review": not bool(ids),
        "original_task_reexecuted": False,
    }


def approved(record: dict, cfg: dict) -> bool:
    approval_ref = cfg.get("rule_approval")
    if approval_ref is None or not record["rule_ids"]:
        return False
    path = Path(approval_ref["path"])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != approval_ref["sha256"]:
        raise ValueError("Rule approval artifact changed")
    approval = json.loads(content)
    if approval["version"] != record["version"] or approval["implementation_sha256"] != record["implementation_sha256"]:
        raise ValueError("Rules changed after their sample audit")
    return set(record["rule_ids"]).issubset(approval["approved_rules"])
