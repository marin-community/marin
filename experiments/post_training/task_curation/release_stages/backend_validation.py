# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The backend-validation stage: the graph's producer of the release's backend survey.

The release's snapshot-bound backend evidence is a graph output, not a hand-produced
pin (ruling 2026-11-24): this stage seals the target's census manifest
(``atlas-backend-census-inputs``, bound to the release snapshot and revision) and runs
the maintained live qualification
(:mod:`experiments.post_training.task_curation.backend_live_qualification`) over it,
producing the backend survey and the validation receipt under the release prefix. The
qualification campaign records (grading environment, integrity contract, runtime
defaults) are reviewed plan configuration; the stage proves they agree with the
graph-visible census artifact before the census runs.
"""

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

MANIFEST_KIND = "atlas-backend-census-inputs"
RECEIPT_SUFFIX = "backend-validation"
TASK_SPEC_COLUMN = "task_spec"


@dataclass(frozen=True)
class BackendValidationInputs:
    """Typed inputs for the release's backend validation; every value is a graph output."""

    release_id: str
    source_revision: str
    snapshot_id: str
    catalog_root: str
    output_root: str
    target: str
    census_inputs: ArtifactRef
    grading_environments: dict[str, dict]
    integrity_contracts: dict[str, dict]
    runtime_defaults: dict

    def validate(self) -> None:
        """Refuse malformed inputs before anything runs."""
        if not self.release_id or not self.source_revision or not self.snapshot_id:
            raise ValueError("release_id, source_revision and snapshot_id are required")
        if not self.catalog_root or not self.output_root:
            raise ValueError("catalog_root and output_root are required")
        if not self.target:
            raise ValueError("the validation target is required")
        if not self.grading_environments.get(self.target):
            raise ValueError(f"the plan carries no grading-environment record for {self.target!r}")
        if not self.integrity_contracts.get(self.target):
            raise ValueError(f"the plan carries no integrity-contract record for {self.target!r}")
        if not isinstance(self.runtime_defaults, dict) or not self.runtime_defaults:
            raise ValueError("the backend runtime defaults are required")


def _target_row(inputs: BackendValidationInputs) -> dict:
    """The target's census-inputs row: its frozen declaration and population pin."""
    from experiments.post_training.task_curation.release_stages.census import (  # noqa: PLC0415 — beside its only use
        INPUTS_KIND,
        CensusSourceRow,
    )

    payload = StoragePath(inputs.census_inputs.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != inputs.census_inputs.sha256:
        raise ValueError(f"census inputs digest mismatch: {inputs.census_inputs.uri}")
    document = json.loads(payload)
    if document.get("kind") != INPUTS_KIND:
        raise ValueError(f"census inputs kind is {document.get('kind')!r}, expected {INPUTS_KIND!r}")
    if document.get("source_revision") != inputs.source_revision:
        raise ValueError(
            f"census inputs pin revision {document.get('source_revision')!r}, expected {inputs.source_revision!r}"
        )
    rows = [CensusSourceRow(**row) for row in document["rows"]]
    matches = [row for row in rows if row.source_id == inputs.target]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} census rows for {inputs.target!r}; the target needs exactly one")
    return dataclasses.asdict(matches[0])


def _actor_record(population_path: str) -> dict:
    """The target's census artifact record: task count, specs digest, specs location."""
    document = json.loads(StoragePath(f"{population_path}/.artifact.json").read_text())
    result = document.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"the census artifact at {population_path} carries no result record")
    for key in ("source_id", "task_count", "task_specs_sha256", "input_artifact"):
        if not result.get(key):
            raise ValueError(f"the census artifact result carries no {key!r}")
    return result


def _specs_parts(population_dir: str) -> list[str]:
    """The frozen population's task-spec parquet parts, in deterministic order."""
    entries = StoragePath(f"{population_dir}/specs/").ls()
    parts = sorted(str(entry) for entry in entries if str(entry).endswith(".parquet"))
    if not parts:
        raise ValueError(f"the census population at {population_dir} carries no task-spec parquets")
    return parts


def _contract_groups(population_path: str) -> list[dict]:
    """The population's sealed integrity-contract groups, for the campaign-record check."""
    document = json.loads(StoragePath(f"{population_path}/integrity-contracts.json").read_text())
    groups = document.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"the population integrity contracts at {population_path} carry no groups")
    return groups


def build_census_manifest(inputs: BackendValidationInputs, *, task_count: int, specs_paths: list[str]) -> dict:
    """Bind the release's census manifest: reviewed campaign records over the frozen population."""
    return {
        "kind": MANIFEST_KIND,
        "release_id": inputs.release_id,
        "snapshot_id": inputs.snapshot_id,
        "source_revision": inputs.source_revision,
        "source_ids": [inputs.target],
        "catalog_root": inputs.catalog_root,
        "grading_environments": {inputs.target: inputs.grading_environments[inputs.target]},
        "integrity_contracts": {inputs.target: inputs.integrity_contracts[inputs.target]},
        "inputs": [
            {
                "source_id": inputs.target,
                "column": TASK_SPEC_COLUMN,
                "expected_tasks": task_count,
                "paths": specs_paths,
            }
        ],
        "runtime_defaults": inputs.runtime_defaults,
        "deployment_limits": {},
        "unassessed_reasons": {},
    }


def run_backend_validation(
    inputs: BackendValidationInputs,
    *,
    dry_run: bool,
    repo_root: Path,
    feeds: tuple | None = None,
) -> dict:
    """Seal the census manifest and run the live qualification; ``dry_run`` verifies bindings."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "backend_validation", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        row = _target_row(inputs)
        actor = _actor_record(row["population_path"])
        if actor["source_id"] != inputs.target:
            raise ValueError(f"the census artifact at {row['population_path']} binds {actor['source_id']!r}")
        task_count = actor["task_count"]
        groups = _contract_groups(row["population_path"])
        measured = sum(group["count"] for group in groups)
        if measured != task_count:
            raise ValueError(
                f"the population's contract groups cover {measured} tasks, the census measured {task_count}"
            )
        assessed = inputs.grading_environments[inputs.target].get("assessed_tasks")
        if assessed != task_count:
            raise ValueError(
                f"the grading-environment record assesses {assessed} tasks, the census measured {task_count}"
            )
        specs_sha = inputs.integrity_contracts[inputs.target].get("task_specs_sha256")
        if specs_sha and specs_sha != actor["task_specs_sha256"]:
            raise ValueError(
                f"the integrity-contract record pins specs {specs_sha}, the census measured {actor['task_specs_sha256']}"
            )
        configure_coreweave_s3()
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "backend_validation",
                {"mode": mode, "target": inputs.target, "task_count": task_count},
                alert_feeds,
            )
            return {
                "kind": f"atlas-{inputs.release_id}-{RECEIPT_SUFFIX}",
                "release_id": inputs.release_id,
                "source_revision": inputs.source_revision,
                "snapshot_id": inputs.snapshot_id,
                "target": inputs.target,
                "task_count": task_count,
                "dry_run": True,
                "sql_writes": 0,
            }
        from experiments.post_training.task_curation.backend_live_qualification import (  # noqa: PLC0415 — the maintained flow
            ValidationInputs,
            run_validation,
        )

        workdir = Path(f"{inputs.output_root}/work")
        workdir.mkdir(parents=True, exist_ok=True)
        local_paths: list[str] = []
        for index, part in enumerate(_specs_parts(actor["input_artifact"])):
            local = workdir / f"specs-{index:06d}.parquet"
            local.write_bytes(StoragePath(part).read_bytes())
            local_paths.append(str(local))
        manifest = build_census_manifest(inputs, task_count=task_count, specs_paths=local_paths)
        manifest_ref = seal(f"{inputs.output_root}/backend-census-manifest.json", manifest)
        receipt = run_validation(
            ValidationInputs(
                release_id=inputs.release_id,
                source_revision=inputs.source_revision,
                snapshot_id=inputs.snapshot_id,
                catalog_root=inputs.catalog_root,
                output_root=f"{inputs.output_root}/backend-validation",
                targets=(inputs.target,),
                manifest=manifest_ref,
                expected_task_count=task_count,
            ),
            dry_run=False,
            repo_root=repo_root,
        )
        receipt["manifest"] = {"uri": manifest_ref.uri, "sha256": manifest_ref.sha256}
        receipt["population"] = {"source_id": inputs.target, "census_artifact": row["population_path"]}
        alerts.stage_finished(
            inputs.release_id,
            "backend_validation",
            {"mode": mode, "task_count": receipt["task_count"], **receipt["receipt"]},
            alert_feeds,
        )
        return receipt
    except Exception as error:
        alerts.stage_failed(
            inputs.release_id, "backend_validation", {"mode": mode, "error": str(error)[:500]}, alert_feeds
        )
        raise


def seal(path: str, value: dict) -> ArtifactRef:
    """Write-once seal with byte-identical readback; returns the artifact reference."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON BackendValidationInputs document")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Checkout the validation runs from")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    inputs = BackendValidationInputs(**json.loads(args.inputs.read_text()))
    proof = run_backend_validation(inputs, dry_run=args.mode == "dry-run", repo_root=args.repo_root)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
