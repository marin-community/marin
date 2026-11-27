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
from huggingface_hub import HfApi, hf_hub_download
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.excluded_population_conversion import repin_license_receipt
from experiments.post_training.task_curation.harbor_cohort_release import release_repo
from experiments.post_training.task_curation.license_binding import (
    LICENSE_RECEIPT_INDEX,
    AppendWouldRewriteSharedShard,
    append_license_receipt,
    read_license_receipts,
)
from experiments.post_training.task_curation.license_publication import (
    check_publication,
    licensed_dataset_card,
    prepare_license_publication,
)
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
    """Replace the tier-scoring verifier and the coaching instruction; record the lineage."""
    verifier = {resource.path: resource for resource in task.resources.verifier}
    script = verifier.get("tests/test.sh")
    instruction = verifier.get("instruction.md")
    if script is None or instruction is None:
        raise ValueError(f"{task.id}: the package carries no tests/test.sh or instruction.md to repair")
    original_script = resource_bytes(script).decode()
    if "4-tier" not in original_script and "3/leading-identifier" not in original_script:
        return task
    provenance = {
        "version": REPAIR_VERSION,
        "removed_tiers": list(REMOVED_TIERS),
        "contract": "normalized exact-text comparison: reward 1.0 on a match, 0 otherwise",
        "input_task_sha256": task_spec_sha256(task),
    }
    updated = []
    for resource in task.resources.verifier:
        if resource.path == "tests/test.sh":
            updated.append(
                inline_resource("tests/test.sh", REPAIRED_TEST_SCRIPT.encode()).model_copy(
                    update={"mode": resource.mode, "mtime_ns": resource.mtime_ns}
                )
            )
        elif resource.path == "instruction.md":
            updated.append(
                inline_resource(
                    "instruction.md", repaired_instruction(resource_bytes(instruction).decode()).encode()
                ).model_copy(update={"mode": resource.mode, "mtime_ns": resource.mtime_ns})
            )
        else:
            updated.append(resource)
    repaired = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={
                    "verifier": tuple(updated),
                    "oracle": (
                        *task.resources.oracle,
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
    release_repo: str,
    release_revision: str,
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
        "input_release": {"repo": release_repo, "revision": release_revision},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    previous = read_license_receipts(LICENSE_RECEIPT_INDEX).get(source_id)
    if previous is None:
        raise ValueError(f"{source_id}: no license receipt to re-pin for the repaired release")
    publication = prepare_license_publication(previous, output_root, use=intended_use)
    body = (
        "Private research artifact. The tier-repair of a base-pinned conversion release: the 4-tier "
        "forgiving verifier is replaced with the clean normalized exact-text contract (reward 1.0 on a "
        "match, 0 otherwise; no partial-credit tiers), and the instruction's scoring hints are removed. "
        "The removed tiers are recorded in the repair lineage and the per-task conversion receipts. "
        f"Input release: {release_repo}@{release_revision}. Converted: {manifest['converted_rows']}; "
        f"input: {manifest['input_rows']}.\n"
    )
    (output_root / "README.md").write_text(
        licensed_dataset_card(body, publication, pretty_name="Tier-repaired conversion release")
    )
    api = HfApi()
    repo = release_repo(source_id, artifact_version)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Conversion destination must be private")
    filenames = ["tasks.parquet", "manifest.json", "conversion-receipts.jsonl", "README.md"]
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(output_root),
        allow_patterns=filenames,
        commit_message="Publish the tier-repaired conversion release",
    )
    card_path = Path(api.hf_hub_download(repo, "README.md", repo_type="dataset", revision=commit.oid))
    bound = repin_license_receipt(previous, repo, commit.oid, card_path.read_bytes())
    try:
        index_commit: str | None = append_license_receipt(
            LICENSE_RECEIPT_INDEX,
            bound,
            message=f"Bind {source_id}'s tier-repaired release to its original upstream license",
        )
    except AppendWouldRewriteSharedShard as error:
        LOGGER.warning("receipt append refused for %s: %s", source_id, error)
        index_commit = None
    license_check = check_publication(source_id, repo, commit.oid, intended_use)
    return {
        **manifest,
        "release_repo": repo,
        "release_revision": commit.oid,
        "license_index_commit": index_commit,
        "license_check_verdict": license_check["verdict"],
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
            json.dumps({"plan": manifest, "release_repo": release_repo(args.source_id, args.artifact_version)}, indent=2)
        )
        return
    result = publish(
        source_id=args.source_id,
        intended_use=args.intended_use,
        release_repo=args.release_repo,
        release_revision=args.release_revision,
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
