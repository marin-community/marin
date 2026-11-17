"""Convert the pinned TaskForge D01 release and retain its historical admitted cohort."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from taskcompendium.models import CommandSemantics, Source, TaskSpec
from taskcompendium.runtime.resources import inline_resource

RELEASE_REVISION = "d415c8e7ac8f4515fb9c5a79bc589c335138bdbe"
RELEASE_REPO = "open-athena/taskforge-testgen20261013"
REFERENCE_PATH = "taskforge/reference_control.json"


def upgrade_environment(environment: dict[str, Any]) -> dict[str, Any]:
    if environment["compatible_backends"]:
        raise ValueError("Cannot preserve explicit legacy compatible_backends")
    upgraded = {key: value for key, value in environment.items() if key != "compatible_backends"}
    if environment.get("docker_image") or environment.get("packages_lock"):
        upgraded["command_semantics"] = CommandSemantics.LINUX_PROCESS
    elif "shell" in environment["capabilities"]:
        upgraded["command_semantics"] = CommandSemantics.SHELL_SIMULATOR
    return upgraded


def reference_control(row: dict[str, Any]) -> dict[str, Any]:
    controls = [
        control
        for control in row["controls"]
        if control["kind"] == "positive" and control["category"] == "known_correct" and control["concern"] == "reference"
    ]
    if len(controls) != 1:
        raise ValueError("Expected exactly one positive reference control")
    return controls[0]["payload"]


def convert(row: dict[str, Any], index: int, importer_revision: str) -> TaskSpec:
    task = row["task"]
    if task["schema_version"] != "0.25":
        raise ValueError("Unexpected source schema")
    if task.get("compatible_backends"):
        raise ValueError("Cannot preserve explicit legacy compatible_backends")
    if task["output_directories"]:
        raise ValueError("Cannot preserve legacy output_directories")
    upgraded = {key: value for key, value in task.items() if key not in ("output_directories", "compatible_backends")}
    upgraded["environment_requirements"] = upgrade_environment(task["environment_requirements"])
    grader = dict(task["grader"])
    if grader.get("environment") is not None:
        grader["environment"] = upgrade_environment(grader["environment"])
    upgraded["grader"] = grader
    resources = dict(task["resources"])
    if any(resource["path"] == REFERENCE_PATH for role in resources.values() for resource in role):
        raise ValueError("Reference control resource path already exists")
    reference = json.dumps(reference_control(row), sort_keys=True).encode()
    resources["oracle"] = [
        *resources["oracle"],
        inline_resource(REFERENCE_PATH, reference).model_dump(mode="json"),
    ]
    source = Source(
        dataset=RELEASE_REPO,
        revision=RELEASE_REVISION,
        row=f"tasks/d01.jsonl:{index}",
        importer_revision=importer_revision,
    )
    return TaskSpec.model_validate(
        {
            **upgraded,
            "id": row["id"],
            "schema_version": "0.30",
            "resources": resources,
            "source": source.model_dump(mode="json"),
            "tags": [*task["tags"], row["id"], f"capability:{row['capability']}", "subject:d01", f"band:{row['band']}"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw_path = args.source_file
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    exclusions = json.loads(args.exclusions.read_text())
    excluded_indices = {int(row["locator"].rsplit(":", 1)[1]) for row in exclusions}
    if len(rows) != 245 or len(exclusions) != 8 or len(excluded_indices) != 8:
        raise ValueError("The pinned release must have 245 source rows and eight unique exclusions")
    if any(index < 0 or index >= len(rows) for index in excluded_indices):
        raise ValueError("Exclusion row lies outside the pinned release")
    converted = [convert(row, index, script_hash) for index, row in enumerate(rows)]
    admitted = [task for index, task in enumerate(converted) if index not in excluded_indices]
    if len(admitted) != 237 or len({task.id for task in admitted}) != 237:
        raise ValueError("Expected 237 unique admitted TaskForge tasks")
    (root / "d01-admitted-schema030.jsonl").write_text("\n".join(task.model_dump_json() for task in admitted) + "\n")
    (root / "conversion-provenance.json").write_text(
        json.dumps(
            {
                "source_repository": RELEASE_REPO,
                "source_revision": RELEASE_REVISION,
                "source_file": "tasks/d01.jsonl",
                "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "converter_sha256": script_hash,
                "runtime_revision": "8385245f5331b44d5ca0b44a7f558ebe4bd1d50b",
                "input_rows": len(rows),
                "validated_rows": len(converted),
                "historically_admitted_rows": len(admitted),
                "excluded_rows": exclusions,
                "admission_evidence": "https://public.applets.marina.oa.dev/a/ac96d706-d275-4785-b77d-5065081b8fc6/v/1/api/assessment?source_id=taskforge%3Ad01%3Atestgen20261013",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validated {len(converted)} rows; retained {len(admitted)} historical admitted rows")


if __name__ == "__main__":
    main()
