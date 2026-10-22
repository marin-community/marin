# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rescore closure evaluation dumps against their validation cohort."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from verifyit.closure_scoring import (
    CLOSURE_REGION_SCORER,
    ClosureFormat,
    ContinuationOutcome,
    closure_response_content,
    continuation_outcome,
    score_closure_region,
)
from verifyit.tool_call_markup import ToolCallStatus, tool_call_regions

from experiments.post_training.structure_closure.token_data import LOCAL_TOKEN_ROUNDTRIP_PASSED, TokenizedCaseRecord
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    CONTINUE_REQUIRED,
    ESCAPE_CONTINUE_REQUIRED,
    TASK_ID_PREFIX,
    VALUE_CONTINUE_REQUIRED,
)

SNOWBALL_TURN_END_MARKERS = ("<|eot_id|>", "<|end_of_text|>")


class UidSource(StrEnum):
    ROW_INDEX = "row_index"
    CASE_ID = "case_id"


def cases_by_uid(cases: Sequence[TokenizedCaseRecord], source: UidSource) -> dict[str, TokenizedCaseRecord]:
    """Index the cohort using the ID convention recorded by its evaluation path."""
    identities = [
        str(index) if source == UidSource.ROW_INDEX else TASK_ID_PREFIX + case["case_id"]
        for index, case in enumerate(cases)
    ]
    if len(set(identities)) != len(cases):
        raise ValueError("Validation cohort contains duplicate evaluation identities")
    return dict(zip(identities, cases, strict=True))


@dataclass(frozen=True)
class CaseResult:
    uid: str
    case_id: str
    family: str
    control: str
    syntax_complete: bool
    content_preserved: bool
    success: bool
    trailing_content: bool
    whole_response_success: bool
    control_outcome: ContinuationOutcome | None
    tool_protocol_status: ToolCallStatus | None


def case_result(
    *,
    uid: str,
    case_id: str,
    family: str,
    control: str,
    prefix: str,
    expected: str,
    structure_format: ClosureFormat,
    content: str,
) -> CaseResult:
    """Classify the generated continuation under the shared text grading contract."""
    text = prefix + content
    score = score_closure_region(text, expected, structure_format)
    tool_status = None
    if structure_format == ClosureFormat.TOOL:
        regions = tool_call_regions(text, initial_thinking=False)
        tool_status = regions[0].status
    outcome = None
    if control in (CONTINUE_REQUIRED, ESCAPE_CONTINUE_REQUIRED, VALUE_CONTINUE_REQUIRED):
        outcome = continuation_outcome(text, expected, structure_format)
    return CaseResult(
        uid, case_id, family, control, **asdict(score), control_outcome=outcome, tool_protocol_status=tool_status
    )


@dataclass(frozen=True)
class AuditFailure:
    uid: str
    reason: str


@dataclass(frozen=True)
class EvaluationAudit:
    expected: int
    observed: int
    missing_uids: tuple[str, ...]
    failures: tuple[AuditFailure, ...]
    results: tuple[CaseResult, ...]

    @property
    def complete(self) -> bool:
        return self.expected > 0 and not self.missing_uids and not self.failures and len(self.results) == self.expected

    def summary(self) -> dict:
        groups: dict[str, Counter] = {}
        control_outcomes: dict[str, Counter] = {}
        tool_protocol: dict[str, Counter] = {}
        for result in self.results:
            if result.control_outcome is not None:
                control_outcomes.setdefault(f"{result.family}/{result.control}", Counter())[result.control_outcome] += 1
            if result.tool_protocol_status is not None:
                tool_protocol.setdefault(f"{result.family}/{result.control}", Counter())[
                    result.tool_protocol_status
                ] += 1
            for key in (result.family, f"{result.family}/{result.control}"):
                counts = groups.setdefault(key, Counter())
                counts.update(
                    scored=1,
                    syntax_complete=int(result.syntax_complete),
                    content_preserved=int(result.content_preserved),
                    success=int(result.success),
                    failure=int(not result.success),
                    trailing_content=int(result.trailing_content),
                    whole_response_success=int(result.whole_response_success),
                )
        return {
            "scorer": CLOSURE_REGION_SCORER,
            "complete": self.complete,
            **asdict(self),
            "groups": dict(sorted(groups.items())),
            "control_outcomes": dict(sorted(control_outcomes.items())),
            "tool_protocol": dict(sorted(tool_protocol.items())),
        }


def audit_evaluation(
    cases: Sequence[TokenizedCaseRecord], rows: Iterable[dict], *, uid_source: UidSource
) -> EvaluationAudit:
    """Rescore one response per case using the recorded prompt conditioning.

    Select row indices for batch evaluation or stable case IDs for step-wise
    evaluation. Row indices require the exact unfiltered validation cohort with
    closure rows first, as in the launch recipe.
    """
    expected = cases_by_uid(cases, uid_source)
    seen: set[str] = set()
    results = []
    failures = []
    observed = 0
    for row in rows:
        observed += 1
        uid = str(row.get("uid", ""))
        try:
            if uid not in expected or uid in seen:
                raise ValueError("Unknown or duplicate evaluation UID")
            seen.add(uid)
            case = expected[uid]
            if row["prompt_token_ids"] != case["prompt_token_ids"]:
                raise ValueError("Evaluation conditioning differs from the frozen case")
            if row.get("exception_type") or row.get("server_error"):
                raise ValueError("Infrastructure error: case cannot be counted as scored")
            verification = row.get("verification")
            if verification is None or verification["status"] != "verified":
                raise ValueError("Verifier score is unavailable")
            if not isinstance(row["output_response"], str):
                raise ValueError("Model response is missing or is not text")
            content = closure_response_content(
                row["output_response"], stop_reason=row["stop_reason"], terminal_markers=SNOWBALL_TURN_END_MARKERS
            )
            results.append(
                case_result(
                    uid=uid,
                    case_id=case["case_id"],
                    family=case["family"],
                    control=case["control"],
                    prefix=case["prefix"],
                    expected=case["expected"],
                    structure_format=ClosureFormat(case["format"]),
                    content=content,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(AuditFailure(uid, str(error)))
    return EvaluationAudit(
        len(cases), observed, tuple(uid for uid in expected if uid not in seen), tuple(failures), tuple(results)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--eval-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid-source", type=UidSource, choices=list(UidSource), required=True)
    args = parser.parse_args()
    contents = (args.tokens / "validation.jsonl").read_bytes()
    manifest = json.loads((args.tokens / "manifest.json").read_text())
    digest = hashlib.sha256(contents).hexdigest()
    if manifest["status"] != LOCAL_TOKEN_ROUNDTRIP_PASSED or digest != manifest["hashes"]["validation"]:
        raise ValueError("Validation cohort differs from the immutable token audit")
    cases = [json.loads(line) for line in contents.splitlines()]
    dump = args.eval_dump.read_bytes()
    audit = audit_evaluation(cases, (json.loads(line) for line in dump.splitlines()), uid_source=args.uid_source)
    report = {
        "uid_source": args.uid_source,
        "validation_sha256": digest,
        "eval_dump_sha256": hashlib.sha256(dump).hexdigest(),
        **audit.summary(),
    }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"complete": audit.complete, "scored": len(audit.results), "expected": audit.expected}))
    raise SystemExit(0 if audit.complete else 1)


if __name__ == "__main__":
    main()
