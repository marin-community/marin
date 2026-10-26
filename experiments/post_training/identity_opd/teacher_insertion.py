# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the identity teacher's prompt insertion and prove it on the frozen mix.

The identity teacher sees the student's prompt with the identity system prompt added, exactly as the Snowball chat
template renders an explicit system message. The template opens with a system header (reasoning mode, and the row's
tools when present), so one rule covers every row: match the header's opening tokens and insert the system message
after the header's ``<|eot_id|><|start_header_id|>``.

For every row routed to the identity teacher, the check renders the student prompt as the task session passes it to
the template (after the chat round trip), applies the insertion, and
requires the result to equal the template's rendering with the system message as the first message. It writes the
rule (for the run config) and one decoded example per family.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from marinskyrl.distillation import TeacherPromptInsertion, conditioned_teacher_prompt
from taskcompendium.chat import chat_input
from taskcompendium.submission import conversation_messages
from transformers import AutoTokenizer

IDENTITY_SYSTEM_PROMPT = "You are Marin Snowball, an AI model created by Open Athena."
HEADER_OPEN = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
HEADER_CLOSE = "<|eot_id|><|start_header_id|>"
INSERTED = f"system<|end_header_id|>\n{IDENTITY_SYSTEM_PROMPT}<|eot_id|>\n<|start_header_id|>"


def ids(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def insertion_rule(tokenizer) -> TeacherPromptInsertion:
    return TeacherPromptInsertion(
        after_prefix=ids(tokenizer, HEADER_OPEN),
        tokens=ids(tokenizer, INSERTED),
        after_marker=ids(tokenizer, HEADER_CLOSE),
    )


def runtime_messages(prompt: list[dict]) -> list[dict]:
    """The messages the task session sends to the template: Arrow's absent fields dropped, then the chat round trip."""
    cleaned = [{key: value for key, value in message.items() if value is not None} for message in prompt]
    return list(conversation_messages(chat_input(cleaned).events))


def render(tokenizer, messages: list[dict], row: dict) -> list[int]:
    tools = json.loads(row["tools"]) if row["tools"] else None
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **json.loads(row["chat_template_kwargs"]),
    )


def check(tokenizer, rows: list[dict], teacher_routes: set[str]) -> dict:
    rule = insertion_rule(tokenizer)
    checked, examples = 0, {}
    for row in rows:
        if row["teacher_route"] not in teacher_routes:
            continue
        messages = runtime_messages(row["prompt"])
        student = render(tokenizer, messages, row)
        teacher = conditioned_teacher_prompt(student, (rule,))
        expected = render(tokenizer, [{"role": "system", "content": IDENTITY_SYSTEM_PROMPT}, *messages], row)
        if teacher != expected:
            raise ValueError(f"teacher prompt differs from the system-prompt rendering for row {row['id']}")
        checked += 1
        family = row["teacher_route"].rsplit("_", 1)[0]
        examples.setdefault(
            row["teacher_route"],
            {
                "id": row["id"],
                "family": family,
                "student": tokenizer.decode(student),
                "teacher": tokenizer.decode(teacher),
            },
        )
    return {
        "rule": {
            "after_prefix": list(rule.after_prefix),
            "tokens": list(rule.tokens),
            "after_marker": list(rule.after_marker),
        },
        "rows_checked": checked,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="local tokenizer directory (v0.1 adc0078a)")
    parser.add_argument("--mix", required=True, help="mixed.parquet")
    parser.add_argument("--routes", nargs="+", required=True, help="teacher routes scored by the identity teacher")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    result = check(tokenizer, pq.read_table(args.mix).to_pylist(), set(args.routes))
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(
        json.dumps(
            {"rows_checked": result["rows_checked"], "rule_lengths": {k: len(v) for k, v in result["rule"].items()}}
        )
    )


if __name__ == "__main__":
    main()
