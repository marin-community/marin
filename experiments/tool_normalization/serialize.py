# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render converted conversations with the Marin template and assistant masks."""

import hashlib
import json
from pathlib import Path
from threading import Lock

import numpy as np
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from transformers import PreTrainedTokenizerFast

from experiments.tool_normalization.candidate import FORBIDDEN_TOOLS

# Transformers caches one mutable assistant-span tracker per compiled Jinja template.
RENDER_LOCK = Lock()


def export(messages: list[dict], tools: list[dict], tokenizer_file: Path, directory: Path) -> dict:
    used = {call["function"]["name"] for message in messages for call in message.get("tool_calls", [])}
    tools = [tool for tool in tools if tool["function"]["name"] in used]
    if used != {tool["function"]["name"] for tool in tools}:
        raise ValueError("Missing native schema for an exported tool")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        chat_template=MARIN_CHAT_TEMPLATE,
    )
    with RENDER_LOCK:
        text = tokenizer.apply_chat_template(messages, tools=tools, enable_thinking=True, tokenize=False)
        if any(name in text for name in FORBIDDEN_TOOLS):
            raise ValueError("Legacy tool name survives final serialization")
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            enable_thinking=True,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    ids = np.asarray(encoded["input_ids"], dtype="<i4")
    mask = np.asarray(encoded["assistant_masks"], dtype="u1")
    if len(ids) > 262144:
        raise ValueError("length_deferred: complete converted trace exceeds authorized length")
    if len(ids) != len(mask) or not mask.any():
        raise ValueError("Missing assistant supervision")
    if tokenizer.decode(ids.tolist(), skip_special_tokens=False) != text:
        raise ValueError("Converted serialization does not round trip")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "text.txt").write_text(text)
    (directory / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2))
    (directory / "tools.json").write_text(json.dumps(tools, ensure_ascii=False, indent=2))
    (directory / "input_ids.i32").write_bytes(ids.tobytes())
    (directory / "loss_mask.u8").write_bytes(mask.tobytes())
    manifest = {
        "tokens": len(ids),
        "supervised_tokens": int(mask.sum()),
        "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest(),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
