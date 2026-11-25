# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate tests/fixtures/atlas_source_pins.json from an Atlas cut.

The fixture pins one origin/atlas-env-audit revision (``atlas_revision``) and one
r40.json artifact version. Regenerate it whenever the Atlas admits more sources or
main's declarations change:

1. Check out the audit branch at the revision the new cut serves, read-only::

       git worktree add /tmp/atlas-cut <revision>

2. Regenerate the fixture from both trees::

       uv run --isolated --frozen python experiments/post_training/task_curation/tests/
       generate_atlas_source_pins.py --audit-root /tmp/atlas-cut

   The audit tree's status computation reads the Atlas registry documents over the
   network (read-only metadata); the plan path and the alias table come from the
   respective trees. Review the printed diff summary before committing the fixture:
   the three tests in ``test_atlas_source_coverage.py`` then drive which sources
   main must declare and which waivers or blocker entries expire.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from finestore.rl.source_identity import canonical_source_id

FIXTURE = Path(__file__).with_name("fixtures") / "atlas_source_pins.json"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True, help="read-only atlas-env-audit checkout")
    parser.add_argument("--main-root", type=Path, default=Path.cwd(), help="this repository's checkout")
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        dump_side(args.audit_root, scratch / "audit.json")
        dump_side(args.main_root, scratch / "main.json")
        audit = json.loads((scratch / "audit.json").read_text())
        main = json.loads((scratch / "main.json").read_text())
        r40 = json.loads((args.audit_root / R40_PLAN).read_text())

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
        **fixture,
        "required_ids": required,
        "declaration_blockers": blockers,
        "pins": pins,
        "pin_parity_waivers": waivers,
    }
    FIXTURE.write_text(json.dumps(updated, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "atlas_sources": len(audit),
                "main_sources": len(main),
                "required": len(required),
                "declaration_blockers": len(blockers),
                "pin_parity_waivers": len(waivers),
            },
            indent=1,
        )
    )
    print("Update atlas_revision (and r40_plan.artifact_version) to the cut you generated from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
