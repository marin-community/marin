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
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

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
    targets: tuple[str, ...]
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
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be a non-empty, duplicate-free tuple")
        if not isinstance(self.runtime_defaults, dict) or not self.runtime_defaults:
            raise ValueError("the backend runtime defaults are required")


def _target_rows(inputs: BackendValidationInputs) -> list[dict]:
    """The targets' census-inputs rows, in the plan's target order."""
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
    rows = {row.source_id: row for row in (CensusSourceRow(**row) for row in document["rows"])}
    missing = [target for target in inputs.targets if target not in rows]
    if missing:
        raise ValueError(f"{len(missing)} targets carry no census row: {missing[:3]}")
    return [dataclasses.asdict(rows[target]) for target in inputs.targets]


def _population_record(path: str) -> SourceTextPopulation:
    """The pinned population record: task count, specs dir, source binding."""
    result = dict(json.loads(StoragePath(f"{path.rstrip('/')}/.artifact.json").read_text())["result"])
    result.setdefault("path", path.rstrip("/"))
    return SourceTextPopulation.model_validate(result)


def _specs_parts(specs_dir: str) -> list[str]:
    """The frozen population's task-spec parquet parts, in deterministic order."""
    entries = StoragePath(f"{specs_dir.rstrip('/')}/").ls()
    parts = sorted(str(entry) for entry in entries if str(entry).endswith(".parquet"))
    if not parts:
        raise ValueError(f"the census population at {specs_dir} carries no task-spec parquets")
    return parts


def _contract_count(contract: dict) -> int:
    """The campaign integrity-contract record's covered task count."""
    groups = contract.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("the integrity-contract record carries no groups")
    return sum(group["count"] for group in groups)


def build_census_manifest(
    inputs: BackendValidationInputs, *, task_counts: dict[str, int], specs_paths: dict[str, list[str]]
) -> dict:
    """Bind the release's census manifest: reviewed campaign records over the frozen populations."""
    return {
        "kind": MANIFEST_KIND,
        "release_id": inputs.release_id,
        "snapshot_id": inputs.snapshot_id,
        "source_revision": inputs.source_revision,
        "source_ids": list(inputs.targets),
        "catalog_root": inputs.catalog_root,
        "grading_environments": {
            target: inputs.grading_environments[target]
            for target in inputs.targets
            if inputs.grading_environments.get(target)
        },
        "integrity_contracts": {
            target: inputs.integrity_contracts[target]
            for target in inputs.targets
            if inputs.integrity_contracts.get(target)
        },
        "inputs": [
            {
                "source_id": target,
                "column": TASK_SPEC_COLUMN,
                "expected_tasks": task_counts[target],
                "paths": specs_paths[target],
            }
            for target in inputs.targets
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
        rows = _target_rows(inputs)
        populations = {}
        task_counts = {}
        for row in rows:
            population = _population_record(row["population_path"])
            if canonical_source_id(population.source_id) != canonical_source_id(row["source_id"]):
                raise ValueError(f"the census population at {row['population_path']} binds {population.source_id!r}")
            populations[row["source_id"]] = population
            task_counts[row["source_id"]] = population.task_count
            record = inputs.integrity_contracts.get(row["source_id"])
            if record and _contract_count(record) != population.task_count:
                raise ValueError(
                    f"the integrity-contract record covers {_contract_count(record)}"
                    f" tasks, the census measured {population.task_count} for {row['source_id']}"
                )
            assessed = (inputs.grading_environments.get(row["source_id"]) or {}).get("assessed_tasks")
            if assessed and assessed != population.task_count:
                raise ValueError(
                    f"the grading-environment record assesses {assessed}"
                    f" tasks, the census measured {population.task_count} for {row['source_id']}"
                )
            specs_sha = (record or {}).get("task_specs_sha256")
            if specs_sha and specs_sha != population.task_specs_sha256:
                raise ValueError(
                    f"the integrity-contract record pins specs {specs_sha},"
                    f" the census measured {population.task_specs_sha256} for {row['source_id']}"
                )
        configure_coreweave_s3()
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "backend_validation",
                {"mode": mode, "targets": len(rows), "task_count": sum(task_counts.values())},
                alert_feeds,
            )
            return {
                "kind": f"atlas-{inputs.release_id}-{RECEIPT_SUFFIX}",
                "release_id": inputs.release_id,
                "source_revision": inputs.source_revision,
                "snapshot_id": inputs.snapshot_id,
                "targets": list(inputs.targets),
                "task_count": sum(task_counts.values()),
                "dry_run": True,
                "sql_writes": 0,
            }
        from experiments.post_training.task_curation.backend_live_qualification import (  # noqa: PLC0415 — the maintained flow
            ValidationInputs,
            run_validation,
        )

        workdir = Path(tempfile.mkdtemp(prefix="atlas-backend-validation-specs-"))
        specs_paths: dict[str, list[str]] = {}
        for row in rows:
            local_paths: list[str] = []
            for index, part in enumerate(_specs_parts(populations[row["source_id"]].specs_output_dir)):
                local = workdir / f"{row['source_id'].replace('/', '_').replace(':', '_')}-{index:06d}.parquet"
                local.write_bytes(StoragePath(part).read_bytes())
                local_paths.append(str(local))
            specs_paths[row["source_id"]] = local_paths
        manifest = build_census_manifest(inputs, task_counts=task_counts, specs_paths=specs_paths)
        manifest_ref = seal(f"{inputs.output_root}/backend-census-manifest.json", manifest)
        # The extracting checkout may sit at any descendant of the pinned revision
        # (pipeline fixes land after the plan); state the revision directly instead of
        # demanding checkout-HEAD equality. In the in-region bundle mode the submitter
        # attests the containment (the guard verifies the revision is landed).
        from experiments.post_training.task_curation.release_stages.guard import (  # noqa: PLC0415 — beside its only use
            attested_revision,
        )

        if attested_revision() is None:
            pinned = subprocess.run(
                ["git", "merge-base", "--is-ancestor", inputs.source_revision, "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if pinned.returncode != 0:
                raise ValueError(f"checkout HEAD does not contain the plan's pinned revision {inputs.source_revision}")
        receipt = run_validation(
            ValidationInputs(
                release_id=inputs.release_id,
                source_revision=inputs.source_revision,
                snapshot_id=inputs.snapshot_id,
                catalog_root=inputs.catalog_root,
                output_root=f"{inputs.output_root}/backend-validation",
                targets=tuple(inputs.targets),
                manifest=manifest_ref,
                expected_task_count=sum(task_counts.values()),
            ),
            dry_run=False,
            repo_root=repo_root,
            revision=inputs.source_revision,
        )
        receipt["manifest"] = {"uri": manifest_ref.uri, "sha256": manifest_ref.sha256}
        receipt["populations"] = {row["source_id"]: {"census_artifact": row["population_path"]} for row in rows}
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
