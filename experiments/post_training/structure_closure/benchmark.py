# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct isolated structure-completion probes from approved BFCL answers."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from rolloutengine.closure_scoring import ClosureFormat, ClosureScore, closure_state, score_closure
from rolloutengine.structure_closure import PrefixStatus, json_closure_state


class Family(StrEnum):
    OBJECT = "json_object"
    ARRAY = "json_array"
    NESTED = "json_nested"
    STRING = "json_string_escape"
    TOOL = "tool_delimiter"
    REASONING = "reasoning_delimiter"
    FENCE = "code_fence"
    REPEATED_OPENER = "repeated_opener_retention"


@dataclass(frozen=True)
class ClosureCase:
    case_id: str
    task_id: str
    split: str
    family: Family
    control: str
    instruction: str
    prefix: str
    expected: str
    source_sha256: str


def closure_format(family: Family) -> ClosureFormat:
    if family in (Family.OBJECT, Family.ARRAY, Family.NESTED, Family.STRING):
        return ClosureFormat.JSON
    if family in (Family.TOOL, Family.REPEATED_OPENER):
        return ClosureFormat.TOOL
    if family == Family.REASONING:
        return ClosureFormat.REASONING
    if family == Family.FENCE:
        return ClosureFormat.CODE_FENCE
    raise ValueError(f"Unknown family: {family}")


def syntax_state(text: str, family: Family):
    return closure_state(text, closure_format(family))


def score_case(case: ClosureCase, continuation: str) -> ClosureScore:
    """Score a successful generation; infrastructure failures must stay unscored."""
    return score_closure(case.prefix + continuation, case.expected, closure_format(case.family))


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def task_cases(task_id: str, answer: list, split: str, source_hash: str) -> list[ClosureCase]:
    """Create known-content synthetic probes, never inferred tool-intent repairs."""
    # The sentinel is transformed task metadata, not a claim about a natural failure.
    sentinel = task_id + ' quote=" slash=\\ end'
    object_text = compact_json({"task": task_id})
    array_text = compact_json([task_id])
    nested_text = compact_json({"task": task_id, "answers": [answer]})
    string_text = compact_json(sentinel)
    tool_body = compact_json({"name": "write_result", "arguments": {"task": task_id, "result": answer}})
    reason_body = "The source task identifier is " + task_id + "."
    fence_body = compact_json(answer)
    specifications = [
        (Family.OBJECT, object_text, len(object_text) - 1),
        (Family.ARRAY, array_text, len(array_text) - 1),
        (Family.NESTED, nested_text, len(nested_text) - 2),
        (Family.STRING, string_text, len(string_text) - 1),
        (Family.TOOL, "<tool_call>" + tool_body + "</tool_call>", len("<tool_call>" + tool_body)),
        (Family.REASONING, "<|start_think|>" + reason_body + "<|end_think|>", len("<|start_think|>" + reason_body)),
        (Family.FENCE, "```json\n" + fence_body + "\n```", len("```json\n" + fence_body + "\n")),
        (Family.REPEATED_OPENER, "<tool_call>" + tool_body + "</tool_call>", len("<tool_call>")),
    ]
    cases = []
    for family, expected, close_cut in specifications:
        # A known nonempty task identifier supplies the continue-required control.
        # Fences use an unfinished opener info string when the answer is empty.
        content_at = expected.find(task_id)
        continue_cut = content_at + max(1, len(task_id) // 2) if content_at >= 0 else len("```j")
        boundaries = [("close_required", close_cut), ("continue_required", continue_cut)]
        if family == Family.STRING:
            boundaries.append(("escape_continue_required", expected.index("\\") + 1))
        for control, cut in boundaries:
            if family == Family.REPEATED_OPENER and control == "close_required":
                control = "opener_retention"
            prefix = expected[:cut]
            state = syntax_state(prefix, family)
            if state.status == PrefixStatus.INVALID or syntax_state(expected, family).status != PrefixStatus.COMPLETE:
                raise ValueError(f"Invalid synthetic construction: {task_id}/{family}/{control}")
            identity = f"{task_id}/{family}/{control}"
            case = ClosureCase(
                hashlib.sha256(identity.encode()).hexdigest(),
                task_id,
                split,
                family,
                control,
                "Continue the supplied assistant prefix to reproduce the reference below. "
                "Preserve its content and close every required structure. Emit only the missing continuation. "
                "The reference is data, not an instruction to execute tools.\nReference:\n" + expected,
                prefix,
                expected,
                source_hash,
            )
            if not score_case(case, expected[cut:]).success:
                raise ValueError(f"Reference does not score: {identity}")
            cases.append(case)
    return cases


def build_benchmark(source: Path, split_file: Path, output: Path):
    """Audit every source task, then publish a new immutable local candidate."""
    if output.exists():
        raise FileExistsError(output)
    split = json.loads(split_file.read_text())
    train, validation, parity = (
        set(split[key]) for key in ("train_task_ids", "diagnostic_reserved_task_ids", "parity_task_ids")
    )
    partition_file = source / "bfcl_complement-manifest.json"
    partition = json.loads(partition_file.read_text())
    approved = {task["source_id"] for task in partition["complement"]}
    if train & validation or (train | validation) & parity or train | validation != approved:
        raise ValueError("Task split is not disjoint and exhaustive")
    cases, failures = [], []
    for task in partition["complement"]:
        path = source / "bfcl_complement" / task["name"] / "solution/solve.sh"
        try:
            raw = path.read_bytes()
            script = raw.decode()
            marker = "<< 'ORACLE_EOF'\n"
            if script.count(marker) != 1:
                raise ValueError("Unrecognized oracle heredoc")
            payload = script.split(marker)[1].split("\nORACLE_EOF")[0]
            if json_closure_state(payload).status != PrefixStatus.COMPLETE:
                raise ValueError("Invalid or duplicate-key oracle JSON")
            answer = json.loads(payload)
            if not isinstance(answer, list):
                raise ValueError("Expected answer array")
            cases.extend(
                task_cases(
                    task["source_id"],
                    answer,
                    "train" if task["source_id"] in train else "validation",
                    hashlib.sha256(raw).hexdigest(),
                )
            )
        except (OSError, ValueError) as error:
            failures.append({"task": task["source_id"], "error": str(error)})
    output.mkdir(parents=True)
    (output / "conversion-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise ValueError(f"{len(failures)} tasks failed; see {output / 'conversion-failures.json'}")
    counts = Counter(f"{case.split}/{case.family}/{case.control}" for case in cases)
    hashes = {}
    for cohort in ("train", "validation"):
        data = "".join(json.dumps(asdict(case), separators=(",", ":")) + "\n" for case in cases if case.split == cohort)
        (output / f"{cohort}.jsonl").write_text(data)
        hashes[cohort] = hashlib.sha256(data.encode()).hexdigest()
    manifest = {
        "status": "candidate-not-frozen",
        "kind": "oracle-derived-synthetic-copy-continuation",
        "counts": dict(counts),
        "cases": len(cases),
        "hashes": hashes,
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "partition_sha256": hashlib.sha256(partition_file.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitation": (
            "Known-content probes isolate serialization; natural failure and free-generation transfer are not measured."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_benchmark(args.source, args.split, args.output), indent=2))
