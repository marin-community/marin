# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare exact conditional token sequences for structure-completion cases."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from transformers import PreTrainedTokenizerFast

from experiments.post_training.structure_closure.benchmark import ClosureCase, Family, Split, closure_format, score_case


class ToolPrompt(StrEnum):
    COMPACT = "compact"
    NATIVE_DECLARED = "native-declared"


WRITE_RESULT_TOOL = {
    "type": "function",
    "function": {
        "name": "write_result",
        "description": "Record the result for a task.",
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}, "result": {"type": "array"}},
            "required": ["task", "result"],
        },
    },
}


def native_tool_case(case: ClosureCase, tokenizer: PreTrainedTokenizerFast) -> ClosureCase:
    """Keep known tool content and control boundaries in the tokenizer's tool format."""
    body = json.loads(case.expected.removeprefix("<tool_call>").removesuffix("</tool_call>"))
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": case.instruction},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": body}]},
        ],
        tools=[WRITE_RESULT_TOOL],
        tokenize=False,
        enable_thinking=False,
    )
    expected = "<tool_call>" + rendered.rsplit("<tool_call>", 1)[1].split("</tool_call>", 1)[0] + "</tool_call>"
    if json.loads(expected.removeprefix("<tool_call>").removesuffix("</tool_call>")) != body:
        raise ValueError("Native serialization changed tool content")
    if case.control == "close_required":
        cut = len(expected) - len("</tool_call>")
    elif case.control == "opener_retention":
        cut = expected.index("{")
    elif case.control == "continue_required":
        cut = expected.index(case.task_id) + max(1, len(case.task_id) // 2)
    elif case.control == "value_continue_required":
        cut = expected.index('"task":') + len('"task":')
    else:
        raise ValueError(f"Unsupported native tool control: {case.control}")
    instruction = case.instruction
    if case.control != "close_required":
        lead, reference = instruction.rsplit("\nReference:\n", 1)
        if reference != case.expected:
            raise ValueError("Instruction reference differs from the case reference")
        instruction = lead + "\nReference:\n" + expected
    return replace(case, prefix=expected[:cut], expected=expected, instruction=instruction)


class TokenizedCaseRecord(TypedDict):
    case_id: str
    task_id: str
    split: str
    family: str
    control: str
    format: str
    instruction: str
    prefix: str
    expected: str
    prompt_token_ids: list[int]
    target_token_ids: list[int]
    prompt_sha256: str
    canonical_joint_encoding_matches: bool


def tokenized_case_record(
    case: ClosureCase, tokenizer: PreTrainedTokenizerFast, *, tool_prompt: ToolPrompt
) -> TokenizedCaseRecord:
    """Render the pinned template and verify prefix-preserving conditional targets."""
    tools = None
    if tool_prompt == ToolPrompt.NATIVE_DECLARED and case.family in (Family.TOOL, Family.REPEATED_OPENER):
        case = native_tool_case(case, tokenizer)
        tools = [WRITE_RESULT_TOOL]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": case.instruction}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=case.family == Family.REASONING,
    )
    prompt_text = rendered + case.prefix
    target_text = case.expected[len(case.prefix) :]
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prompt_text:
        raise ValueError("Prompt tokens do not preserve rendered text")
    decoded = tokenizer.decode(prompt_ids + target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != prompt_text + target_text:
        raise ValueError("Conditional target tokens change prefix or reference text")
    for stop in range(1, len(target_ids) + 1):
        partial = tokenizer.decode(
            prompt_ids + target_ids[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not partial.startswith(prompt_text):
            raise ValueError("Intermediate target token changes the conditioning prefix")
    if not score_case(case, target_text).success:
        raise ValueError("Token target does not satisfy the case contract")
    return {
        "case_id": case.case_id,
        "task_id": case.task_id,
        "split": case.split,
        "family": case.family.value,
        "control": case.control,
        "format": closure_format(case.family).value,
        "instruction": case.instruction,
        "prefix": case.prefix,
        "expected": case.expected,
        "prompt_token_ids": prompt_ids,
        "target_token_ids": target_ids,
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "canonical_joint_encoding_matches": (
            tokenizer.encode(prompt_text + target_text, add_special_tokens=False) == prompt_ids + target_ids
        ),
    }


def prepare_tokens(
    cases_directory: Path, initializer: Path, output: Path, context_tokens: int, *, tool_prompt: ToolPrompt
) -> dict:
    """Audit all branches and retain every failure in one report before launch."""
    if output.exists():
        raise FileExistsError(output)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(initializer, local_files_only=True)
    manifest_path = cases_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True)
    failures = []
    counts = Counter()
    max_prompt = max_target = noncanonical = 0
    output_hashes = {}
    for split in Split:
        source = cases_directory / f"{split}.jsonl"
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["hashes"][split]:
            raise ValueError(f"Source hash mismatch: {split}")
        destination = output / f"{split}.jsonl"
        with source.open() as inputs, destination.open("x") as outputs:
            for line in inputs:
                raw = json.loads(line)
                case = ClosureCase(**(raw | {"family": Family(raw["family"])}))
                try:
                    record = tokenized_case_record(case, tokenizer, tool_prompt=tool_prompt)
                    prompt_length = len(record["prompt_token_ids"])
                    target_length = len(record["target_token_ids"])
                    if prompt_length + target_length > context_tokens:
                        raise ValueError("Reference continuation exceeds context budget")
                except ValueError as error:
                    failures.append(
                        {"case_id": case.case_id, "family": case.family, "split": split, "error": str(error)}
                    )
                    continue
                outputs.write(json.dumps(record, separators=(",", ":")) + "\n")
                counts[f"{split}/{case.family}/{case.control}"] += 1
                max_prompt = max(max_prompt, prompt_length)
                max_target = max(max_target, target_length)
                noncanonical += not record["canonical_joint_encoding_matches"]
        output_hashes[split] = hashlib.sha256(destination.read_bytes()).hexdigest()
    report = {
        "counts": dict(counts),
        "max_prompt_tokens": max_prompt,
        "max_target_tokens": max_target,
        "noncanonical_joint_encoding_cases": noncanonical,
        "context_tokens": context_tokens,
        "tool_prompt": tool_prompt.value,
        "failures": failures,
        "hashes": output_hashes,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((initializer / "tokenizer.json").read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256((initializer / "chat_template.jinja").read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "failed" if failures else "local-token-roundtrip-passed",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--tool-prompt", type=ToolPrompt, choices=list(ToolPrompt), required=True)
    args = parser.parse_args()
    report = prepare_tokens(args.cases, args.initializer, args.output, args.context_tokens, tool_prompt=args.tool_prompt)
    print(json.dumps({key: value for key, value in report.items() if key not in ("counts", "failures")}))
    print(f"Conversion failures: {len(report['failures'])}")
    raise SystemExit(bool(report["failures"]))
