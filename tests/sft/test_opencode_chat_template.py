# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opencode tools-aware chat template + its Levanter chat-format wiring."""

from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import chat_template_has_generation_block

from experiments.sft.opencode_chat_template import OPENCODE_TOOLS_CHAT_TEMPLATE, opencode_chat_lm_format


def test_template_has_generation_block():
    """Levanter's completions-only masking requires a {% generation %} block; assert it is present
    and the markers are balanced (ChatProcessor hard-errors otherwise when mask_user_turns=True)."""
    assert chat_template_has_generation_block(OPENCODE_TOOLS_CHAT_TEMPLATE)
    assert OPENCODE_TOOLS_CHAT_TEMPLATE.count("{% generation %}") == OPENCODE_TOOLS_CHAT_TEMPLATE.count(
        "{% endgeneration %}"
    )
    assert "{% generation %}" in OPENCODE_TOOLS_CHAT_TEMPLATE


def test_template_carries_tools_and_tool_call_surface():
    """The tools-aware surface (the parts whose divergence silently breaks agentic SFT) is present."""
    for marker in ("{%- if tools %}", "<tools>", "| tojson", "<tool_call>", "<tool_response>", "<think>"):
        assert marker in OPENCODE_TOOLS_CHAT_TEMPLATE, marker


def test_opencode_chat_lm_format_wiring():
    """The builder returns a ChatLmDatasetFormat wired for the opencode agentic-SFT path."""
    fmt = opencode_chat_lm_format()
    assert isinstance(fmt, ChatLmDatasetFormat)
    assert fmt.chat_template == OPENCODE_TOOLS_CHAT_TEMPLATE
    assert fmt.chat_template_kwargs == "chat_template_kwargs"
    assert fmt.mask_user_turns is True
    assert fmt.pack is True
    assert fmt.messages_field == "messages"
