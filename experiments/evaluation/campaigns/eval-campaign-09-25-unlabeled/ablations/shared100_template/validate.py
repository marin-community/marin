# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the three overlays against pinned source prompt and token behavior."""

import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from launch import OUTPUT, ROOT, SHARED_MODEL, prepare
from transformers import AutoTokenizer


def main() -> None:
    manifest = yaml.safe_load((ROOT / "variants.yaml").read_text())
    shared_revision = manifest["base_model_revision"]
    antidoom_model = manifest["antidoom_model"]
    antidoom_revision = manifest["antidoom_revision"]
    shared_tokenizer = AutoTokenizer.from_pretrained(SHARED_MODEL, revision=shared_revision)
    antidoom_tokenizer = AutoTokenizer.from_pretrained(antidoom_model, revision=antidoom_revision)
    shared_template = Path(hf_hub_download(SHARED_MODEL, "chat_template.jinja", revision=shared_revision)).read_text()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]
    history = [
        {"role": "system", "content": "Use the calculator tool for arithmetic."},
        {"role": "user", "content": "What is 19 + 23?"},
        {
            "role": "assistant",
            "reasoning_content": "Call the calculator.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_probe_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"a":19,"b":23}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_probe_1", "name": "calculator", "content": "42"},
        {"role": "user", "content": "Give only the result."},
    ]

    variants = {}
    for arm in manifest["arms"]:
        config = yaml.safe_load(prepare(arm).read_text())
        template = config["serve"]["chat_template"]
        tokenizer = antidoom_tokenizer if arm == "template_and_tokenizer" else shared_tokenizer
        rendered = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
            xml_tools=tools,
            enable_thinking=True,
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        variants[arm] = {
            "rendered_prompt": rendered,
            "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "token_count": len(token_ids),
            "token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            "tool_call_id_present": "call_probe_1" in rendered,
            "think_block_present": "<|start_think|>Call the calculator.<|end_think|>" in rendered,
        }

    no_reasoning = [{key: value for key, value in message.items() if key != "reasoning_content"} for message in history]
    original = shared_tokenizer.apply_chat_template(
        no_reasoning,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=shared_template,
        xml_tools=tools,
        enable_thinking=True,
    )
    reasoning_only = yaml.safe_load(prepare("reasoning_only").read_text())["serve"]["chat_template"]
    minimal_without_reasoning = shared_tokenizer.apply_chat_template(
        no_reasoning,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=reasoning_only,
        xml_tools=tools,
        enable_thinking=True,
    )
    if original != minimal_without_reasoning:
        raise AssertionError("reasoning-only template changed prompts without reasoning_content")
    if variants["template_only"]["rendered_prompt"] != variants["template_and_tokenizer"]["rendered_prompt"]:
        raise AssertionError("template-only and tokenizer arms rendered different text")
    if not variants["reasoning_only"]["think_block_present"]:
        raise AssertionError("reasoning-only template did not render the think block")

    output = {
        "source_revisions": {"shared100": shared_revision, "antidoom": antidoom_revision},
        "arm3_no_reasoning_prompt_identical_to_shared100": True,
        "arms": variants,
    }
    (OUTPUT / "rendered-prompt-checks.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                arm: {key: value for key, value in result.items() if key != "rendered_prompt"}
                for arm, result in variants.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
