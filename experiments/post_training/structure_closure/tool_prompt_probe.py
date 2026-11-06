# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether tool declarations or chat-template formatting affect tool closure.

Each case produces four prompts with the same tool-call content: declared versus
undeclared tools, crossed with compact JSON versus the supplied tokenizer's
native chat-template serialization. Declared prompts pass the tool schema to
that template. Only the closing tool delimiter is missing. The four arms let
you compare closure success across both factors and their interaction.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import ClosureFormat, score_closure_region
from verifyit.tool_call_markup import TOOL_CLOSE

from experiments.post_training.structure_closure.token_data import (
    LOCAL_TOKEN_ROUNDTRIP_PASSED,
    TokenizedCaseRecord,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import CLOSE_REQUIRED, Family
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tool_serialization import (
    paired_tool_variants,
    write_result_tool,
)


class PairedToolProbeRecord(TokenizedCaseRecord):
    probe_variant: str
    source_case_id: str


def paired_tool_probes(record: TokenizedCaseRecord, tokenizer: PreTrainedTokenizerFast) -> list[PairedToolProbeRecord]:
    """Keep task content fixed across declared/undeclared and compact/native formats."""
    messages = [{"role": "user", "content": record["instruction"]}]
    records = []
    for variant in paired_tool_variants(record["case_id"], record["prefix"], record["instruction"], tokenizer):
        tools = [write_result_tool()] if variant.declared else None
        rendered = cast(
            str,
            tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
        )
        prompt = rendered + variant.prefix
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(TOOL_CLOSE, add_special_tokens=False)
        decoded = tokenizer.decode(
            prompt_ids + target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded != prompt + TOOL_CLOSE:
            raise ValueError(f"Conditional token round trip failed: {variant.name}")
        if not score_closure_region(variant.expected, variant.expected, ClosureFormat.TOOL).success:
            raise ValueError(f"Reference tool closure failed: {variant.name}")
        if variant.name == "undeclared-compact" and prompt_ids != record["prompt_token_ids"]:
            raise ValueError("Control arm differs from its source token record")
        records.append(
            cast(
                PairedToolProbeRecord,
                {
                    **record,
                    "case_id": variant.case_id,
                    "probe_variant": variant.name,
                    "source_case_id": record["case_id"],
                    "prefix": variant.prefix,
                    "expected": variant.expected,
                    "prompt_token_ids": prompt_ids,
                    "target_token_ids": target_ids,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "canonical_joint_encoding_matches": (
                        tokenizer.encode(prompt + TOOL_CLOSE, add_special_tokens=False) == prompt_ids + target_ids
                    ),
                },
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = (args.tokens / "validation.jsonl").read_bytes()
    manifest = json.loads((args.tokens / "manifest.json").read_text())
    if manifest["status"] != LOCAL_TOKEN_ROUNDTRIP_PASSED:
        raise ValueError("Frozen validation token conversion did not pass")
    if hashlib.sha256(source).hexdigest() != manifest["hashes"]["validation"]:
        raise ValueError("Frozen validation source hash mismatch")
    cases = [json.loads(line) for line in source.splitlines()]
    cases = sorted(
        (case for case in cases if case["family"] == Family.TOOL and case["control"] == CLOSE_REQUIRED),
        key=lambda case: case["case_id"],
    )
    if not 0 < args.tasks <= len(cases):
        raise ValueError("Requested task count is outside the source cohort")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.initializer, local_files_only=True)
    records, failures = [], []
    for case in cases[: args.tasks]:
        try:
            records.extend(paired_tool_probes(case, tokenizer))
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"case_id": case["case_id"], "error": str(error)})
    args.output.mkdir(parents=True)
    payload = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records).encode()
    (args.output / "validation.jsonl").write_bytes(payload)
    report = {
        "status": "failed" if failures else LOCAL_TOKEN_ROUNDTRIP_PASSED,
        "hashes": {"validation": hashlib.sha256(payload).hexdigest()},
        "counts": dict(Counter(f"validation/{record['probe_variant']}" for record in records)),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "failures": failures,
        "purpose": "Paired diagnostic only; source validation hashes are recorded separately.",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
