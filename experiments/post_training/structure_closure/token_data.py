# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare exact conditional token sequences for structure-completion cases."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from experiments.post_training.structure_closure.benchmark import ClosureCase, Family, closure_format, score_case


def case_tokens(case: ClosureCase, tokenizer: PreTrainedTokenizerFast) -> dict:
    """Render the pinned template and verify prefix-preserving conditional targets."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": case.instruction}],
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


def prepare_tokens(cases_directory: Path, initializer: Path, output: Path, context_tokens: int) -> dict:
    """Audit all branches and retain every failure in one report before launch."""
    if output.exists():
        raise FileExistsError(output)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(initializer, local_files_only=True)
    manifest = json.loads((cases_directory / "manifest.json").read_text())
    output.mkdir(parents=True)
    failures = []
    counts = Counter()
    max_prompt = max_target = noncanonical = 0
    output_hashes = {}
    for split in ("train", "validation"):
        source = cases_directory / f"{split}.jsonl"
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["hashes"][split]:
            raise ValueError(f"Source hash mismatch: {split}")
        destination = output / f"{split}.jsonl"
        with source.open() as inputs, destination.open("x") as outputs:
            for line in inputs:
                raw = json.loads(line)
                case = ClosureCase(**(raw | {"family": Family(raw["family"])}))
                try:
                    record = case_tokens(case, tokenizer)
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
        "failures": failures,
        "hashes": output_hashes,
        "source_manifest_sha256": hashlib.sha256((cases_directory / "manifest.json").read_bytes()).hexdigest(),
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
    args = parser.parse_args()
    report = prepare_tokens(args.cases, args.initializer, args.output, args.context_tokens)
    print(json.dumps({key: value for key, value in report.items() if key not in ("counts", "failures")}))
    print(f"Conversion failures: {len(report['failures'])}")
    raise SystemExit(bool(report["failures"]))
