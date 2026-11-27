# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the OpenSWE keep-only release into the private TaskSpec release.

The admission chain's conversion stage for the OpenSWE wave (ruling 48 + the
2026-11-26 6:03 AM option-a profile): read the pinned keep-only population,
convert every row that the profile can express faithfully, and publish the
converted population as a private release with per-task receipts. The run is
strict about coverage, not about rejections: every input row must leave the
conversion either converted or typed-excluded with evidence, never silently
dropped. Three typed exclusion families are expected:

- the profile's translation rejections (rows whose Dockerfile bytes the
  profile cannot express),
- the fidelity gate's divergent rows (the gate's typed per-row exclusions),
- the shared chain's own typed rejections (base pins the registry cannot
  resolve, static blockers).
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection

from experiments.post_training.task_curation.datasets.openswe import bases, convert
from experiments.post_training.task_curation.datasets.openswe.sources import (
    KEEP_RELEASE_REPO,
    KEEP_RELEASE_REVISION,
    PINNED_POPULATION,
    keep_release_rows,
)
from experiments.post_training.task_curation.harbor_cohort_conversion import CONVERSION_PROTOCOL
from experiments.post_training.task_curation.license_publication import publish_licensed_population

LOGGER = logging.getLogger(__name__)

TASK_SCHEMA = pa.schema([("path", pa.string()), ("task_spec", pa.string()), ("task_sha256", pa.string())])
RECEIPTS_FILENAME = "conversion-receipts.jsonl"
EXCLUSIONS_FILENAME = "conversion-exclusions.jsonl"
GATE_RESULT_FILENAME = "fidelity-gate-result.json"
GATE_ROWS_FILENAME = "rows.jsonl"


@dataclass(frozen=True)
class GateEvidence:
    """The fidelity gate's per-row verdicts and its result record."""

    qualified: frozenset[str]
    excluded: dict[str, dict]
    result: dict

    @property
    def sampled(self) -> int:
        return len(self.qualified) + len(self.excluded)


def load_gate_evidence(gate_root: str) -> GateEvidence:
    """The fidelity gate's verdicts from its staged report; refuse a missing run.

    The gate qualifies the recipe per base on a sample: its qualified rows
    convert normally, its excluded rows become typed exclusions, and its
    unsampled rows convert under the qualified profile.
    """
    result = json.loads((StoragePath(gate_root) / GATE_RESULT_FILENAME).read_text())
    if "sampled" not in result or "qualified" not in result:
        raise ValueError(f"The fidelity gate report at {gate_root} is not a gate result record")
    excluded: dict[str, dict] = {}
    qualified: set[str] = set()
    with (StoragePath(gate_root) / GATE_ROWS_FILENAME).open("r") as stream:
        for line in stream:
            record = json.loads(line)
            if record["status"] == "qualified":
                qualified.add(record["instance_id"])
            else:
                excluded[record["instance_id"]] = record
    if len(qualified) + len(excluded) != result["sampled"]:
        raise ValueError(
            f"The fidelity gate report records {result['sampled']} sampled rows but carries "
            f"{len(qualified)} qualified and {len(excluded)} excluded"
        )
    return GateEvidence(frozenset(qualified), excluded, result)


def row_conversion_record(row, outcome) -> dict | None:
    """The typed exclusion record for a rejected row outcome, else ``None``.

    Covers both shapes the converter produces: the exception instance a raise
    path surfaces (caught by the caller) and the instance ``convert_row``
    returns for a row it excluded mid-conversion.
    """
    if isinstance(outcome, convert.OpenSweRowExcluded):
        return {"instance_id": row.instance_id, "reason": outcome.reason, "detail": outcome.detail, "evidence": {}}
    if isinstance(outcome, ImportRejection):
        return {
            "instance_id": row.instance_id,
            "reason": f"import_{outcome.kind.value}",
            "detail": outcome.reason,
            "evidence": {},
        }
    return None


def convert_population(
    *,
    condensed_bindings: object,
    gate_root: str,
    limit: int | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Convert the keep-only population; every row converts or carries typed evidence."""
    condensed_bases = bases.load_condensed_bases(condensed_bindings)
    rows = keep_release_rows()
    if limit is not None:
        rows = rows[:limit]
    gate = load_gate_evidence(gate_root)
    context = ConversionContext(inputs={}, grader_environment=None)
    from experiments.post_training.task_curation.native_source_conversion import (  # noqa: PLC0415
        resolve_base_digest,
    )

    converted: list[dict] = []
    receipts: list[dict] = []
    exclusions: list[dict] = []
    for row in rows:
        gate_record = gate.excluded.get(row.instance_id)
        if gate_record is not None:
            exclusions.append(
                {
                    "instance_id": row.instance_id,
                    "reason": f"fidelity_gate_{gate_record.get('exclusion_reason') or 'untyped'}",
                    "detail": gate_record.get("exclusion_detail") or "",
                    "evidence": gate_record,
                }
            )
            continue
        try:
            outcome = convert.convert_row(row, condensed_bases, context, resolve=resolve_base_digest)
        except convert.OpenSweRowExcluded as excluded:
            exclusions.append(row_conversion_record(row, excluded))
            continue
        record = row_conversion_record(row, outcome)
        if record is not None:
            exclusions.append(record)
            continue
        serialized = outcome.task.model_dump_json()
        converted.append(
            {
                "path": row.instance_id,
                "task_spec": serialized,
                "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            }
        )
        receipts.append(
            {
                "task_id": outcome.task.id,
                "protocol": CONVERSION_PROTOCOL,
                "input_task_sha256": outcome.input_task_sha256,
                "output_task_sha256": outcome.output_task_sha256,
                "changed_fields": ["environment_requirements", "execution_requirements", "grader", "tags", "labels"],
                "base_pins": [{"base": pin.base, "pinned": pin.pinned} for pin in outcome.base_pins],
            }
        )
    counts = {
        "input_rows": len(rows),
        "converted_rows": len(converted),
        "excluded_rows": len(exclusions),
        "gate_sampled": gate.sampled,
        "gate_qualified": len(gate.qualified),
    }
    if counts["input_rows"] != counts["converted_rows"] + counts["excluded_rows"]:
        raise ValueError(
            f"The conversion lost rows: {counts['input_rows']} inputs vs {counts['converted_rows']} converted "
            f"and {counts['excluded_rows']} excluded"
        )
    return converted, receipts, exclusions, counts


def write_conversion_artifacts(
    converted: list[dict],
    receipts: list[dict],
    exclusions: list[dict],
    counts: dict,
    *,
    output_root: str,
) -> dict:
    """Write the converted population, receipts, typed exclusions and the manifest."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tasks.parquet"
    with pq.ParquetWriter(destination, TASK_SCHEMA) as writer:
        for start in range(0, len(converted), 32):
            writer.write_table(pa.Table.from_pylist(converted[start : start + 32], schema=TASK_SCHEMA))
    (output / RECEIPTS_FILENAME).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in receipts))
    (output / EXCLUSIONS_FILENAME).write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in exclusions))
    reasons: dict[str, int] = {}
    for exclusion in exclusions:
        reasons[exclusion["reason"]] = reasons.get(exclusion["reason"], 0) + 1
    manifest = {
        "protocol": CONVERSION_PROTOCOL,
        "source_id": "TaskTrove:GAIR__OpenSWE__openswe_oss",
        "converter": "experiments.post_training.task_curation.datasets.openswe.convert",
        "upstream_release": {"repo": KEEP_RELEASE_REPO, "revision": KEEP_RELEASE_REVISION},
        "pinned_population": PINNED_POPULATION,
        "task_specs_sha256": file_sha256(StoragePath(str(destination))),
        "rejections": dict(sorted(reasons.items())),
        **counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def publish_conversion(
    *,
    output_root: str,
    version: str,
    manifest: dict,
    gate_root: str,
    converter_revision: str,
    use: str,
) -> dict:
    """Publish the written artifacts as a private release and verify the readback.

    The conversion's licence carries forward from the keep-only release's receipt
    through the publication chokepoint; the new pin's receipt registers before the
    manifest is returned.
    """
    output = Path(output_root)
    digest = hashlib.sha256(b"TaskTrove:GAIR__OpenSWE__openswe_oss").hexdigest()[:8]
    repo = f"open-athena/gair__openswe__openswe_oss-{digest}-taskspec-{version}"
    gate_result = json.loads((StoragePath(gate_root) / GATE_RESULT_FILENAME).read_text())
    body = f"""# OpenSWE oss TaskSpec conversion

Private research artifact. The dedicated OpenSWE converter profile (coordinator
ruling 2026-11-26 6:03 AM, option a): the upstream eval_script rides byte-verbatim
as the selection and grading authority; grading fails closed on a missing or
ambiguous OPENSWE_EXIT_CODE verdict; no test IDs are synthesized; the Dockerfile
translation binds the digest-pinned condensed bases and re-expresses the repo copy
as a build-time clone+checkout. Every row carries its license partition evidence
verbatim from the keep-only release.

- Upstream keep-only release: {KEEP_RELEASE_REPO}@{KEEP_RELEASE_REVISION} ({PINNED_POPULATION} rows, pinned
  GAIR/OpenSWE). License verification:
  s3://marin-us-east-02a/marin/rl-data/atlas/openswe-license-sweep/2026.11.18.1/base-commit-license-verification.json
  (ruling 48, user 2026-11-17; coordinator rulings 2026-11-26 5:16/5:25 AM).
- Fidelity gate: {gate_root} — qualified {gate_result.get("qualified")} of
  {gate_result.get("sampled")} sampled rows (20 per condensed python base); its typed
  exclusions are excluded rows here with per-row evidence.
- Converted {manifest["converted_rows"]} of {manifest["input_rows"]}; typed exclusions
  {manifest["excluded_rows"]} ({json.dumps(manifest["rejections"], sort_keys=True)}).
- Converter: harbor-cohort-taskspec-v1 at {converter_revision}.
- No quality result or oracle execution is claimed for the un-fidelity-gated rows;
  census, controls, lexical decontamination and SAMPLE QA run on this release before
  any admission.
"""
    publication = publish_licensed_population(
        manifest["source_id"],
        KEEP_RELEASE_REPO,
        KEEP_RELEASE_REVISION,
        repo,
        output,
        ("tasks.parquet", "manifest.json", RECEIPTS_FILENAME, EXCLUSIONS_FILENAME),
        body=body,
        pretty_name="OpenSWE oss TaskSpec conversion (ruling 48 keep-only)",
        use=use,
    )
    return {
        **manifest,
        "release_repo": publication.repo,
        "release_revision": publication.revision,
        "license_index_commit": publication.registry_commit,
        "license_check_verdict": publication.license_check["verdict"],
    }


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condensed-bindings", required=True, help="The staged condensed base bindings URI.")
    parser.add_argument("--gate-root", required=True, help="The staged fidelity gate report prefix.")
    parser.add_argument("--output-root", required=True, help="The conversion output directory (local scratch).")
    parser.add_argument("--artifact-version", required=True, help="The release version suffix.")
    parser.add_argument("--converter-revision", help="The landed revision the conversion runs from.")
    parser.add_argument("--limit", type=int, help="Convert only the first N rows (local validation slices).")
    parser.add_argument("--use", required=True, choices=["train", "eval"], help="The publication use.")
    parser.add_argument("--run", action="store_true", help="Publish the private release; omit to validate only.")
    parser.add_argument("--report", type=Path, help="Write the result JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    configure_coreweave_s3()

    revision = args.converter_revision
    if revision is None:
        import subprocess  # noqa: PLC0415

        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ValueError("the conversion cannot read its own checkout revision")
        revision = result.stdout.strip()
    converted, receipts, exclusions, counts = convert_population(
        condensed_bindings=args.condensed_bindings,
        gate_root=args.gate_root,
        limit=args.limit,
    )
    manifest = write_conversion_artifacts(converted, receipts, exclusions, counts, output_root=args.output_root)
    manifest["converter_revision"] = revision
    manifest["mode"] = "publish" if args.run else "plan"
    result = {**manifest, "output_root": args.output_root}
    if args.run:
        result.update(
            publish_conversion(
                output_root=args.output_root,
                version=args.artifact_version,
                manifest=manifest,
                gate_root=args.gate_root,
                converter_revision=revision,
                use=args.use,
            )
        )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)
    if counts["converted_rows"] == 0:
        raise SystemExit("the conversion converted nothing")


if __name__ == "__main__":
    main()
