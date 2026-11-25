# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the Atlas source-coverage fixture and republish its pinned reference.

The fixture pins one origin/atlas-env-audit revision (``atlas_revision``) and one
r40.json artifact version. Its bytes live in the public dataset
``open-athena/atlas-source-pins``; the repository commits only the immutable
reference (revision, sha256, byte size) in ``test_atlas_source_coverage.py``.
Regenerate whenever the Atlas admits more sources or main's declarations change:

1. Check out the audit branch at the revision the new cut serves, read-only::

       git worktree add /tmp/atlas-cut <revision>

2. Regenerate, upload a new dataset revision and print the reference to paste::

       uv run --isolated --frozen python experiments/post_training/task_curation/tests/
       generate_atlas_source_pins.py --audit-root /tmp/atlas-cut

   The audit tree's status computation reads the Atlas registry documents over the
   network (read-only metadata); the plan path and the alias table come from the
   respective trees. Paste the printed reference constants into the test module in
   the same change, then let the three tests in ``test_atlas_source_coverage.py``
   drive which sources main must declare and which waivers or blocker entries
   expire.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from finestore.rl.source_identity import canonical_source_id
from huggingface_hub import HfApi

FIXTURE_NAME = "atlas_source_pins.json"
PINS_REPO = "open-athena/atlas-source-pins"
R40_PLAN = "experiments/post_training/task_curation/release_stages/plans/r40.json"

DUMP_SCRIPT = """
import json, sys
from experiments.post_training.task_curation.sources import all_sources

rows = []
for source in all_sources().values():
    info = source.info
    dataset = source.dataset
    pipeline = source.pipeline
    rows.append(
        {
            "id": info.id,
            "status": source.status.value,
            "dataset_repo": dataset.name if dataset else None,
            "dataset_revision": dataset.revision if dataset else None,
            "pipeline_files": list(source.files) if pipeline is not None else [],
            "pipeline_name": source.name if pipeline is not None else None,
            "pipeline_version": source.version if pipeline is not None else None,
        }
    )
json.dump(rows, open(sys.argv[1], "w"), indent=1, sort_keys=True)
"""

PIN_PARITY_BLOCKER = (
    "Pending the Atlas canonical and retained prebuilt readers on main: the Atlas admitted "
    "pin is a published TaskCompendium release whose reader and receipt bindings main does "
    "not carry yet, so re-pinning main's declaration would point its converter at a payload "
    "it cannot read. The reader port lands with the ingestion PRs."
)
DECLARATION_BLOCKER = (
    "The Atlas admits this source, but the import that would run it on main is still pending: "
    "the Atlas's canonical and retained prebuilt readers and receipt bindings do not exist on "
    "main, so re-pinning or declaring it pipeline-less would either break its converter or "
    "reverses the #11903 review resolution that a record lands together with its import. The "
    "Atlas admitted pin is recorded in `pins`; an ingestion PR removes this entry when it lands "
    "the import and the declaration at that pin."
)
R40_TARGET_BLOCKER = (
    "The R40 admit set targets this requirement-backfilled variant (r40.json), whose sealed "
    "backfill cohort carries the carry-forward binding receipt; the retained reader that would "
    "run it on main is still pending, so its declaration lands with that import per the #11903 "
    "review resolution. The Atlas pin is recorded in `pins`."
)
R40_ONLY_TARGETS = frozenset(
    {
        "TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2@backfilled-20261122",
        "TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2@backfilled-20261122",
    }
)


@dataclass(frozen=True)
class PublishedReference:
    """The immutable reference a fresh upload prints for the test module."""

    repo: str
    revision: str
    sha256: str
    byte_size: int


def dump_side(root: Path, out: Path) -> None:
    subprocess.run(
        ["uv", "run", "--isolated", "--frozen", "python", "-c", DUMP_SCRIPT, str(out)],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def durable_ref(row: dict) -> dict:
    return {
        "dataset_repo": row["dataset_repo"],
        "dataset_revision": row["dataset_revision"],
        "pipeline_files": row["pipeline_files"],
        "pipeline_name": row["pipeline_name"],
        "pipeline_version": row["pipeline_version"],
    }


def build_fixture(audit_root: Path, main_root: Path, prior: dict) -> tuple[dict, int, int]:
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        dump_side(audit_root, scratch / "audit.json")
        dump_side(main_root, scratch / "main.json")
        audit = {row["id"]: row for row in json.loads((scratch / "audit.json").read_text())}
        main = {row["id"]: row for row in json.loads((scratch / "main.json").read_text())}
        r40 = json.loads((audit_root / R40_PLAN).read_text())

    required = {row["id"]: {"source": "atlas-admitted"} for row in audit.values() if row["status"] == "Available"}
    for target in r40["targets"]:
        required.setdefault(target, {"source": "r40-target"})

    main_canon = {canonical_source_id(source_id): source_id for source_id in main}
    pins, blockers, waivers = {}, {}, {}
    for source_id in sorted(required):
        atlas_ref = durable_ref(audit[source_id])
        pins[source_id] = atlas_ref
        main_id = main_canon.get(canonical_source_id(source_id))
        if main_id is None:
            blockers[source_id] = R40_TARGET_BLOCKER if source_id in R40_ONLY_TARGETS else DECLARATION_BLOCKER
        elif durable_ref(main[main_id]) != atlas_ref:
            waivers[source_id] = PIN_PARITY_BLOCKER

    updated = {
        **prior,
        "required_ids": required,
        "declaration_blockers": blockers,
        "pins": pins,
        "pin_parity_waivers": waivers,
    }
    return updated, len(blockers), len(waivers)


def publish(raw: bytes) -> PublishedReference:
    """Upload the fixture as a new dataset revision and return its pinned reference."""
    api = HfApi()
    api.create_repo(PINS_REPO, repo_type="dataset", private=False, exist_ok=True)
    info = api.upload_file(
        path_or_fileobj=raw,
        path_in_repo=FIXTURE_NAME,
        repo_id=PINS_REPO,
        repo_type="dataset",
        commit_message="Regenerate the Atlas source-coverage pins fixture",
    )
    revision = info.commit_sha
    return PublishedReference(
        repo=PINS_REPO,
        revision=revision,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True, help="read-only atlas-env-audit checkout")
    parser.add_argument("--main-root", type=Path, default=Path.cwd(), help="this repository's checkout")
    parser.add_argument(
        "--no-upload", action="store_true", help="write the fixture beside the generator without uploading"
    )
    args = parser.parse_args()
    fixture_path = Path(__file__).with_name(FIXTURE_NAME)
    prior = json.loads(fixture_path.read_text()) if fixture_path.exists() else {"atlas_revision": None}
    updated, blocker_count, waiver_count = build_fixture(args.audit_root, args.main_root, prior)
    raw = (json.dumps(updated, indent=1, sort_keys=True) + "\n").encode()
    fixture_path.write_bytes(raw)
    print(f"fixture: {len(updated['required_ids'])} required, {blocker_count} blockers, {waiver_count} waivers")

    if args.no_upload:
        print("not uploaded (--no-upload); upload it before pasting a reference")
        return 0
    reference = publish(raw)
    print("Paste into test_atlas_source_coverage.py:")
    print(f'ATLAS_SOURCE_PINS_REVISION = "{reference.revision}"')
    print(f'ATLAS_SOURCE_PINS_SHA256 = "{reference.sha256}"')
    print(f"ATLAS_SOURCE_PINS_BYTES = {reference.byte_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
