# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair the laion v4 tier-scoring graders to the clean exact-text contract.

The laion v4 patch's 4-tier forgiving verifier is a reward hack (coordinator
rulings 2026-11-27): tier 3 pays the FULL reward for matching only the first
identifier token, tier 4 pays 0.25 for any identifier-shaped output under ~400
characters, and the instruction coaches both ("writing something shaped like a
completion fragment is strictly better than writing nothing"). This module
repairs the affected conversion releases to the clean contract the java family
already ships: a normalized exact-text comparison that pays 1.0 on a match and
0 otherwise, with no partial-credit tiers and no scoring hints in the
instruction. The removed tiers are recorded in the repair lineage.
"""

import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.harbor_cohort_release import release_repo as conversion_release_repo
from experiments.post_training.task_curation.license_publication import publish_licensed_population
from experiments.post_training.task_curation.native_source_conversion import (
    write_native_conversion_artifacts,
)

LOGGER = logging.getLogger(__name__)

REPAIR_VERSION = "crosscode-tier-repair-v1"
REMOVED_TIERS = (
    "tier 3 (leading-identifier match paid the full reward 1.0 for matching only the first identifier token)",
    "tier 4 (non-empty fallback paid 0.25 for any identifier-shaped output under ~400 characters)",
    "the instruction's scoring-hints section, which coached both tiers",
)
REPAIRED_TEST_SCRIPT = """#!/bin/bash
# The clean contract (the java family's verifier): a normalized exact-text
# comparison. Reward 1.0 on a match, 0 otherwise; no partial-credit tiers.
set -euo pipefail

mkdir -p /logs/verifier
EXPECTED="$(dirname "$0")/solution.txt"
OUTPUT="/app/solution.txt"
REWARD="/logs/verifier/reward.txt"

echo 0 > "$REWARD"
if [[ ! -f "$OUTPUT" ]]; then
    echo "Missing solution file $OUTPUT" >&2
    exit 1
fi

normalize() { sed -e 's/[[:space:]]*$//' "$1" | grep -v '^[[:space:]]*$' || true; }

if diff -u <(normalize "$EXPECTED") <(normalize "$OUTPUT") > /logs/verifier/diff.txt 2>&1; then
    echo 1 > "$REWARD"
    exit 0
fi
echo "Solution mismatch:" >&2
cat /logs/verifier/diff.txt >&2 || true
exit 1
"""

SCORING_SECTION_START = "The verifier byte-compares"
NEXT_SECTIONS = ("### Worked example", "## ", "### ")


def repaired_instruction(text: str) -> str:
    """Drop the scoring-hints section; state the honest exact-comparison contract."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.lstrip(" -*").startswith(SCORING_SECTION_START)), None)
    if start is None:
        raise ValueError("The instruction carries no recognizable scoring section to repair")
    end = next(
        (i for i in range(start + 1, len(lines)) if any(lines[i].startswith(head) for head in NEXT_SECTIONS)),
        len(lines),
    )
    honest = [
        "- The verifier compares your fragment to the reference fragment after whitespace",
        "  normalisation (CRLF -> LF, trailing whitespace stripped, leading and trailing",
        "  blank lines stripped) and accepts only an exact match. Anything else scores 0.",
    ]
    repaired = lines[:start] + honest + lines[end:]
    return "\n".join(repaired) + ("\n" if text.endswith("\n") else "")


def repair_task(task: TaskSpec) -> TaskSpec:
    """Replace the tier-scoring verifier, the coaching instruction AND the public
    conversation message; record the lineage.

    The scoring hints live in TWO places: the private ``instruction.md`` resource and
    the public conversation message (the task's context event — the text the actor
    actually reads). The SAMPLE QC caught the first repair repairing only the
    resource (87/100 csharp tasks flagged: the public contract misstated the rubric).
    Both surfaces get the same surgery.
    """
    verifier = {resource.path: resource for resource in task.resources.verifier}
    script = verifier.get("tests/test.sh")
    instruction = verifier.get("instruction.md")
    if script is None or instruction is None:
        raise ValueError(f"{task.id}: the package carries no tests/test.sh or instruction.md to repair")
    original_script = resource_bytes(script).decode()
    # A task needs the repair while EITHER surface carries the scoring hints: the
    # v1-repaired releases cleaned the verifier but left the public message coached.
    public_coached = any(
        isinstance(event.content, str) and SCORING_SECTION_START in event.content for event in task.context.events
    )
    if "4-tier" not in original_script and "3/leading-identifier" not in original_script and not public_coached:
        return task
    provenance = {
        "version": REPAIR_VERSION,
        "removed_tiers": list(REMOVED_TIERS),
        "contract": "normalized exact-text comparison: reward 1.0 on a match, 0 otherwise",
        "input_task_sha256": task_spec_sha256(task),
    }
    updated = []
    for resource in task.resources.verifier:
        # Each surface repairs only when IT still carries the hints: the v1-repaired
        # releases cleaned the verifier and the resource instruction but left the
        # public message coached — the pass-through must not force a full rewrite.
        if resource.path == "tests/test.sh" and (
            "4-tier" in original_script or "3/leading-identifier" in original_script
        ):
            updated.append(
                inline_resource("tests/test.sh", REPAIRED_TEST_SCRIPT.encode()).model_copy(
                    update={"mode": resource.mode, "mtime_ns": resource.mtime_ns}
                )
            )
        elif resource.path == "instruction.md" and SCORING_SECTION_START in resource_bytes(instruction).decode():
            updated.append(
                inline_resource(
                    "instruction.md", repaired_instruction(resource_bytes(instruction).decode()).encode()
                ).model_copy(update={"mode": resource.mode, "mtime_ns": resource.mtime_ns})
            )
        else:
            updated.append(resource)
    # The public conversation message: the same scoring-section surgery.
    events = []
    public_repaired = 0
    for event in task.context.events:
        content = event.content
        if isinstance(content, str) and SCORING_SECTION_START in content:
            events.append(event.model_copy(update={"content": repaired_instruction(content)}))
            public_repaired += 1
        else:
            events.append(event)
    context = task.context.model_copy(update={"events": tuple(events)})
    repaired = task.model_copy(
        update={
            "context": context,
            "resources": task.resources.model_copy(
                update={
                    "verifier": tuple(updated),
                    # The v1-repaired rows already carry a provenance resource — replace it
                    # (a path collision) with the fresh lineage.
                    "oracle": (
                        *(
                            resource
                            for resource in task.resources.oracle
                            if resource.path != "taskcompendium/crosscode-tier-repair.json"
                        ),
                        inline_resource(
                            "taskcompendium/crosscode-tier-repair.json", json.dumps(provenance, indent=1).encode()
                        ),
                    ),
                }
            ),
            "labels": {**task.labels, "tier_repair_version": REPAIR_VERSION},
        }
    )
    repaired = TaskSpec.model_validate_json(repaired.model_dump_json())
    if public_repaired == 0:
        raise ValueError(f"{task.id}: the public conversation message carries no scoring section to repair")
    return repaired


def convert_population(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Repair every row; strict — a row without the tier verifier fails the run."""
    converted, receipts = [], []
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        repaired = repair_task(task)
        serialized = repaired.model_dump_json()
        converted.append(
            {
                "path": task.source.row,
                "task_spec": serialized,
                "task_sha256": __import__("hashlib").sha256(serialized.encode()).hexdigest(),
            }
        )
        receipts.append(
            {
                "task_id": task.id,
                "protocol": REPAIR_VERSION,
                "input_task_sha256": task_spec_sha256(task),
                "output_task_sha256": task_spec_sha256(repaired),
                "changed_fields": ["resources.verifier", "resources.oracle", "labels"],
                "removed_tiers": list(REMOVED_TIERS),
            }
        )
    counts = {"input_rows": len(rows), "converted_rows": len(converted), "excluded_rows": 0}
    return converted, receipts, counts


def load_population(release_repo: str, release_revision: str) -> list[dict]:
    path = Path(hf_hub_download(release_repo, "tasks.parquet", repo_type="dataset", revision=release_revision))
    return pq.read_table(path, columns=["task_spec"]).to_pylist()


def publish(
    *,
    source_id: str,
    intended_use: str,
    input_repo: str,
    input_revision: str,
    output_root: Path,
    artifact_version: str,
    converted: list[dict],
    receipts: list[dict],
    counts: dict,
) -> dict:
    manifest = write_native_conversion_artifacts(
        converted, receipts, counts, output_root=str(output_root), source_id=source_id
    )
    manifest["converter"] = "experiments.post_training.task_curation.crosscode_tier_repair"
    manifest["mode"] = "publish"
    manifest["repair"] = {
        "version": REPAIR_VERSION,
        "removed_tiers": list(REMOVED_TIERS),
        "input_release": {"repo": input_repo, "revision": input_revision},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    body = (
        "Private research artifact. The tier-repair of a base-pinned conversion release: the 4-tier "
        "forgiving verifier is replaced with the clean normalized exact-text contract (reward 1.0 on a "
        "match, 0 otherwise; no partial-credit tiers), and the instruction's scoring hints are removed. "
        "The removed tiers are recorded in the repair lineage and the per-task conversion receipts. "
        f"Input release: {input_repo}@{input_revision}. Converted: {manifest['converted_rows']}; "
        f"input: {manifest['input_rows']}.\n"
    )
    # The repair derives from the input release's receipt; the chokepoint carries that
    # licence onto the repaired pin and registers the receipt before returning.
    publication = publish_licensed_population(
        source_id,
        conversion_release_repo(source_id, artifact_version),
        output_root,
        ("tasks.parquet", "manifest.json", "conversion-receipts.jsonl"),
        body=body,
        pretty_name="Tier-repaired conversion release",
        use=intended_use,
    )
    return {
        **manifest,
        "release_repo": publication.repo,
        "release_revision": publication.revision,
        "license_check_verdict": publication.license_check["verdict"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--release-repo", required=True)
    parser.add_argument("--release-revision", required=True)
    parser.add_argument("--intended-use", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    rows = load_population(args.release_repo, args.release_revision)
    converted, receipts, counts = convert_population(rows)
    if not args.run:
        manifest = write_native_conversion_artifacts(
            converted, receipts, counts, output_root=str(args.output_root), source_id=args.source_id
        )
        print(
            json.dumps(
                {"plan": manifest, "release_repo": conversion_release_repo(args.source_id, args.artifact_version)},
                indent=2,
            )
        )
        return
    result = publish(
        source_id=args.source_id,
        intended_use=args.intended_use,
        input_repo=args.release_repo,
        input_revision=args.release_revision,
        output_root=args.output_root,
        artifact_version=args.artifact_version,
        converted=converted,
        receipts=receipts,
        counts=counts,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
