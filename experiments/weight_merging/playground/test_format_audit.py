# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.weight_merging.playground.format_audit import tolerant_code


def test_last_closed_fence_is_selected_without_inspecting_test_outcomes():
    text = "```python\ndef old():\n    return 1\n```\nFinal:\n```python\ndef new():\n    return 2\n```"
    assert tolerant_code(text) == ("def new():\n    return 2", "last_closed_fence")


def test_last_unfenced_complete_region_preserves_imports_and_helpers():
    text = "<think>Draft:\ndef old():\n    return 1\n\nWait, revise this.\nimport math\ndef helper():\n    return 2\n\ndef final():\n    return helper()\n\nSo that is my answer."
    code, policy = tolerant_code(text)
    assert policy == "unfenced_ast_draft"
    assert code.startswith("import math\ndef helper")
    assert code.endswith("return helper()")
    assert "old" not in code


def test_truncated_body_salvage_uses_complete_prefix_and_rejects_plain_prose():
    assert tolerant_code("<think>I have no code yet.") == (None, "no_code")
    code, _ = tolerant_code("def valid():\n    return 1\n\ndef incomplete(")
    assert code == "def valid():\n    return 1"
