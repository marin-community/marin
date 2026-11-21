# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the forge cohort's calibration responses for its actorless tasks.

An actorless forge task has no preserved solution material, so its oracle
submits the calibration response the calibration-solutions directory supplies:
``response.json``, one OpenAI-format assistant message whose content is the
golden control's reply. This reads the staged release's task ids, extracts each
selected actorless task's golden reply from its exported forge record, writes
one response per task, and archives the tree for the calibration worker::

    uv run python -m experiments.post_training.task_curation.forge_solutions \\
        --release-root s3://.../forge-main/2026.11.21.2/release \\
        --records s3://marin-us-east-02a/marin/forge-viz/v1 \\
        --output /tmp/forge-solutions \\
        --archive s3://.../forge-main/2026.11.21.2/calibration-solutions.tar.gz
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.forge_release import (
    LOWERED_TASK_SPEC_COLUMN,
    MANIFEST_FILENAME,
    read_records,
)

RESPONSE_FILENAME = "response.json"


def actorless_without_solution(task: dict) -> bool:
    """Whether this published task needs a calibration response file."""
    requirements = task.get("environment_requirements") or {}
    if requirements.get("command_semantics") is not None:
        return False
    oracle = (task.get("resources") or {}).get("oracle") or []
    return not any(resource["path"].startswith("solution/") for resource in oracle)


def golden_response(record: dict) -> dict:
    """The calibration response document for one actorless task."""
    controls = [
        control for control in record["workpiece"]["submission"]["controls"] if control.get("kind") == "positive"
    ]
    if not controls:
        raise ValueError(f"{record['id']}: no positive control was recorded")
    reply = (controls[0].get("candidate") or {}).get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError(f"{record['id']}: its positive control carries no reply")
    return {"role": "assistant", "content": reply}


def write_responses(release_root: str, records_root: str, output: Path) -> dict:
    """Write one response directory per selected actorless task and return the counts."""
    root = StoragePath(release_root)
    manifest = json.loads((root / MANIFEST_FILENAME).read_bytes())
    staged = pq.read_table(
        io.BytesIO((root / "lowered-tasks.parquet").read_bytes()), columns=["path", LOWERED_TASK_SPEC_COLUMN]
    )
    rows = staged.to_pylist()
    selected = {row["path"] for row in rows}
    if len(selected) != manifest["selected"]:
        raise ValueError(f"Staged release holds {len(selected)} rows, manifest expects {manifest['selected']}")
    records = {record["id"]: record for record in read_records(records_root) if record["id"] in selected}
    if set(records) != selected:
        raise ValueError(f"Missing records for {sorted(selected - set(records))[:5]}")
    output.mkdir(parents=True, exist_ok=False)
    envelopes = {row["path"]: json.loads(row[LOWERED_TASK_SPEC_COLUMN])["task"] for row in rows}
    written = 0
    for task_id, record in sorted(records.items()):
        if not actorless_without_solution(envelopes[task_id]):
            continue
        directory = output / task_id
        directory.mkdir(parents=True)
        (directory / RESPONSE_FILENAME).write_text(json.dumps([golden_response(record)], ensure_ascii=False))
        written += 1
    return {"selected": len(selected), "responses": written}


def archive_responses(output: Path, archive: str) -> str:
    """Tar the response tree onto object storage and return its sha256.

    Task ids carry a slash, so each response lives one directory deep; the
    archive preserves that layout, which the calibration's per-task copy
    expects.
    """
    digest = hashlib.sha256()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for response in sorted(output.rglob(RESPONSE_FILENAME)):
            task_dir = response.parent
            name = task_dir.relative_to(output).as_posix()
            digest.update(name.encode())
            digest.update(response.read_bytes())
            bundle.add(task_dir, arcname=name)
    StoragePath(archive).write_bytes(buffer.getvalue())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", required=True, help="object-storage root holding the staged release")
    parser.add_argument("--records", required=True, help="the forge export root holding inputs/ and records/")
    parser.add_argument("--output", type=Path, required=True, help="local calibration-solutions directory")
    parser.add_argument("--archive", required=True, help="object-storage path receiving the response archive")
    args = parser.parse_args()
    configure_coreweave_s3()
    counts = write_responses(args.release_root, args.records, args.output)
    sha256 = archive_responses(args.output, args.archive)
    print(json.dumps({**counts, "archive": args.archive, "sha256": sha256}, indent=2))


if __name__ == "__main__":
    main()
