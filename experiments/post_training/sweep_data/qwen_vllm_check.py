# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare qwen_examples.py rows with the prompt vLLM 0.25.1 itself builds for each example's last turn (task #230).

Runs in an isolated environment with vllm==0.25.1 (CPU is enough: only the chat path runs, no model is loaded):

    uv venv vllm-env && VIRTUAL_ENV=vllm-env uv pip install vllm==0.25.1
    vllm-env/bin/python -I qwen_vllm_check.py <tokenizer dir> <examples.json> <out dir>

The request is what an OpenAI client sends for the last assistant turn: every earlier message, either resending
earlier reasoning in the 'reasoning' field vLLM reads, or not resending it (what agent harnesses do). Prints, per
example and variant, whether vLLM's prompt tokens equal the training row's context before that turn, and writes both
texts for reading by eye.
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.renderers.hf import safe_apply_chat_template


def main() -> None:
    tokenizer_dir, examples_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(model=tokenizer_dir, tokenizer=tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    for example in json.loads(Path(examples_path).read_text()):
        messages = example["messages"]
        last = max(index for index, message in enumerate(messages) if message["role"] == "assistant")
        start, end = example["spans"][-1]
        row = example["ids"][:start]
        (out / f"row-{example['family']}.txt").write_text(
            tokenizer.decode(row) + "\n=== TRAINED: last turn ===\n" + tokenizer.decode(example["ids"][start:end])
        )
        for variant in ("reasoning_resent", "reasoning_not_resent"):
            request = []
            for message in messages[:last]:
                message = dict(message)
                reasoning = message.pop("reasoning_content", None)
                if message["role"] == "assistant" and variant == "reasoning_resent" and reasoning:
                    message["reasoning"] = reasoning
                request.append(message)
            conversation, *_ = parse_chat_messages(request, config, content_format="string")
            kwargs = {"tools": example["tools"], "add_generation_prompt": True}
            text = safe_apply_chat_template(config, tokenizer, conversation, tokenize=False, **kwargs)
            ids = safe_apply_chat_template(config, tokenizer, conversation, tokenize=True, **kwargs)
            (out / f"vllm-prompt-{example['family']}-{variant}.txt").write_text(text)
            difference = next((i for i, (a, b) in enumerate(zip(ids, row, strict=False)) if a != b), None)
            print(
                json.dumps(
                    {
                        "family": example["family"],
                        "trajectory": example["trajectory"],
                        "variant": variant,
                        "vllm_prompt_tokens": len(ids),
                        "row_context_tokens": len(row),
                        "identical": ids == row,
                        "first_difference": difference,
                        "prompt_ends_like_row": text.endswith("<|im_start|>assistant\n<think>\n"),
                    }
                )
            )


if __name__ == "__main__":
    main()
