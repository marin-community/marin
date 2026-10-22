# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move training continuation controls near a verified premature-closure boundary.

For the reference ["first", "second"], a prefix such as ["first", "secon
can be closed into valid JSON by appending "] but would lose the final "d".
The correct continuation is d"]. This variant advances each training prefix to
the latest such point, so a model must preserve the remaining reference content
before closing. Validation data stays unchanged.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from verifyit.closure_scoring import ContinuationOutcome, continuation_outcome

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    CONTINUE_REQUIRED,
    ClosureCase,
    Family,
    Split,
    closure_format,
    score_case,
    syntax_state,
)


def near_boundary_case(case: ClosureCase) -> ClosureCase:
    """Advance only ordinary training controls, retaining known reference content."""
    if case.split != Split.TRAIN or case.control != CONTINUE_REQUIRED:
        return case
    if not case.expected.startswith(case.prefix):
        raise ValueError(f"Prefix is not part of the reference: {case.case_id}")
    for cut in range(len(case.expected) - 1, len(case.prefix) - 1, -1):
        prefix = case.expected[:cut]
        suffix = syntax_state(prefix, case.family).closing_suffix
        if not suffix:
            continue
        outcome = continuation_outcome(prefix + suffix, case.expected, closure_format(case.family))
        if outcome == ContinuationOutcome.REFERENCE_PREFIX_CLOSED_EARLY:
            result = replace(case, prefix=prefix)
            if not score_case(result, case.expected[cut:]).success:
                raise ValueError(f"Reference continuation does not pass: {case.case_id}")
            return result
    raise ValueError(f"No premature-closure boundary: {case.case_id}")


def prepare_training_prefixes(source: Path, output: Path) -> dict:
    """Write an immutable training variant with byte-identical validation data."""
    if output.exists():
        raise FileExistsError(output)
    original = (source / "train.jsonl").read_bytes()
    validation = (source / "validation.jsonl").read_bytes()
    manifest_path = source / "manifest.json"
    source_manifest = manifest_path.read_bytes()
    manifest = json.loads(source_manifest)
    for split, payload in ((Split.TRAIN, original), (Split.VALIDATION, validation)):
        if hashlib.sha256(payload).hexdigest() != manifest["hashes"][split]:
            raise ValueError(f"Source hash mismatch: {split}")
    lines = []
    failures = []
    counts = Counter()
    for number, line in enumerate(original.splitlines(keepends=True), 1):
        try:
            row = json.loads(line)
            case = ClosureCase(**{**row, "family": Family(row["family"])})
            if case.split != Split.TRAIN:
                raise ValueError("Non-training row in train.jsonl")
            updated = near_boundary_case(case)
            if updated != case:
                counts[case.family.value] += 1
                lines.append((json.dumps(asdict(updated), separators=(",", ":")) + "\n").encode())
            else:
                lines.append(line)
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"line": number, "error": str(error)})
    output.mkdir(parents=True)
    (output / "conversion-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise ValueError(f"{len(failures)} conversion failures; see {output / 'conversion-failures.json'}")
    training = b"".join(lines)
    (output / "train.jsonl").write_bytes(training)
    (output / "validation.jsonl").write_bytes(validation)
    manifest = {
        "kind": "training-near-boundary-prefixes",
        "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "source_train_sha256": hashlib.sha256(original).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hashes": {
            "train": hashlib.sha256(training).hexdigest(),
            "validation": hashlib.sha256(validation).hexdigest(),
        },
        "changed_prefixes_by_family": dict(counts),
        "training_rows": len(lines),
        "validation_unchanged": True,
        "selection": "Latest reference prefix at or after the original prefix admitting a verified premature close.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_training_prefixes(args.source, args.output), indent=2))
