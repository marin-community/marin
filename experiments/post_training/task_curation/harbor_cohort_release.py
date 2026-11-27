# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a native Harbor cohort at its pin into a private TaskSpec release.

The re-admission chain's conversion stage (ruling 2026-11-24): read the cohort's
pinned ``tasks.parquet``, convert every row with the general harbor→script-grader
converter, and publish the converted population as a private Hugging Face
release with the per-task conversion receipts the admission gate's evidence
expects. The run is strict: any row the converter rejects fails the run, because
a silently shrunken population would change the census contract without typed
admission evidence. ``--limit`` slices the input for local validation runs;
``--run`` gates the private upload the same way every other release module does.

The published rows carry the standard cohort carrier schema
(``path``/``task_spec``/``task_sha256``), so the census, controls and prebuilt
importers consume the release unchanged.
"""

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import source_display_name
from finestore.rl.licenses import require_license
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.harbor_cohort_conversion import (
    CONVERSION_PROTOCOL,
    CohortConversion,
    convert_cohort_task,
)
from experiments.post_training.task_curation.license_publication import publish_licensed_population
from experiments.post_training.task_curation.publication_gate import require_public_data
from experiments.post_training.task_curation.source import RlDataSource
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned

LOGGER = logging.getLogger(__name__)

TASK_SCHEMA = pa.schema([("path", pa.string()), ("task_spec", pa.string()), ("task_sha256", pa.string())])
RECEIPTS_FILENAME = "conversion-receipts.jsonl"
REPO_SLUG_LIMIT = 63


def source_slug(source_id: str) -> str:
    """The release-name slug for one cohort source."""
    return source_id.partition(":")[2].replace("/", "-").lower()


def release_repo(source_id: str, version: str) -> str:
    slug = source_slug(source_id)[:REPO_SLUG_LIMIT].rstrip("-")
    digest = hashlib.sha256(source_id.encode()).hexdigest()[:8]
    return f"open-athena/{slug}-{digest}-taskspec-{version}"


@dataclass(frozen=True)
class ConversionOutcome:
    source_id: str
    cohort_repo: str
    cohort_revision: str
    input_rows: int
    converted_rows: int
    task_specs_sha256: str
    receipts_path: str
    release_repo: str | None
    release_revision: str | None
    license_check: dict | None = None


def require_conversion_source(source: RlDataSource) -> None:
    """License gates and the shape gate before any cohort work."""
    dataset = source.dataset
    if dataset is None or not dataset.revision:
        raise ValueError(f"Source {source.info.id} requires an exact pinned cohort")
    require_unbanned(source.info.id, dataset.name, source.info.license_receipt, source.intended_use.value)
    require_license(source.info.license_receipt, source.info.id, source.intended_use.value, ban=None)
    require_public_data(source.info.data_restriction, source.info.id, dataset.name)


def convert_population(source: RlDataSource, *, limit: int | None) -> tuple[list[dict], list[dict], dict]:
    """Convert the pinned cohort's rows; strict — any rejection fails the run."""
    dataset = source.dataset
    assert dataset is not None
    parquet = hf_hub_download(dataset.name, "tasks.parquet", repo_type="dataset", revision=dataset.revision)
    rows = pq.read_table(parquet, columns=["task_spec"]).to_pylist()
    if limit is not None:
        rows = rows[:limit]
    converted: list[dict] = []
    receipts: list[dict] = []
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        outcome: CohortConversion = convert_cohort_task(task, input_task_sha256=task_spec_sha256(task))
        serialized = outcome.task.model_dump_json()
        converted.append(
            {
                "path": task.source.row,
                "task_spec": serialized,
                "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            }
        )
        receipts.append(
            {
                "task_id": task.id,
                "protocol": CONVERSION_PROTOCOL,
                "input_task_sha256": outcome.input_task_sha256,
                "output_task_sha256": outcome.output_task_sha256,
                "changed_fields": list(outcome.changed_fields),
            }
        )
    return converted, receipts, {"input_rows": len(rows), "converted_rows": len(converted)}


def write_conversion_artifacts(
    converted: list[dict], receipts: list[dict], counts: dict, *, output_root: str, source_id: str
) -> dict:
    """Write the converted population, its receipts and the manifest; refuse a shrunken population."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tasks.parquet"
    with pq.ParquetWriter(destination, TASK_SCHEMA) as writer:
        for start in range(0, len(converted), 32):
            writer.write_table(pa.Table.from_pylist(converted[start : start + 32], schema=TASK_SCHEMA))
    receipts_path = output / RECEIPTS_FILENAME
    receipts_path.write_text("".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in receipts))
    manifest = {
        "protocol": CONVERSION_PROTOCOL,
        "source_id": source_id,
        "converter": "experiments.post_training.task_curation.harbor_cohort_conversion",
        "task_specs_sha256": file_sha256(StoragePath(str(destination))),
        **counts,
        "rejections": {},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if counts["converted_rows"] != counts["input_rows"]:
        raise ValueError("A strict cohort conversion must convert every input row")
    return manifest


def conversion_release_body(source, manifest: dict) -> str:
    """The conversion release card's provenance body, shared by the cohort lanes."""
    dataset = source.dataset
    assert dataset is not None
    return (
        f"# {source_display_name(source.info.id)}\n\n"
        "Private research artifact. The TaskSpec-only conversion of a native Harbor cohort: every "
        "row keeps the original package's task.toml, instruction, environment and verifier files and "
        "its own tests/test.sh harness; the grader re-expresses Harbor's verifier steps for the "
        "rollout sandbox without rewriting the harness. Conversion receipts carry the per-task "
        "input/output SHAs. No quality result or oracle execution is claimed; census, controls, "
        "lexical decontamination and SAMPLE QA run on this release before any admission.\n\n"
        f"Upstream cohort: {dataset.name} at {dataset.revision}. {dataset.url}\n\n"
        f"Converted: {manifest['converted_rows']}; input: {manifest['input_rows']}. "
        "Mechanical exclusions: none (strict conversion).\n\n"
        f"Preserved source notes: {source.info.notes.split(' Admission blocked:')[0]}\n"
    )


def publish_conversion(
    source: RlDataSource,
    *,
    output_root: str,
    version: str,
    manifest: dict,
) -> ConversionOutcome:
    """Publish the written conversion artifacts as a private release and verify the readback.

    The release's licence carries forward from the cohort receipt through the
    publication chokepoint: the card declares the cohort's resolved upstream, the
    upload is read back, and the release pin's receipt lands in the registry before
    the outcome returns.
    """
    dataset = source.dataset
    assert dataset is not None
    output = Path(output_root)

    publication = publish_licensed_population(
        source.info.id,
        release_repo(source.info.id, version),
        output,
        ("tasks.parquet", "manifest.json", RECEIPTS_FILENAME),
        body=conversion_release_body(source, manifest),
        pretty_name="Harbor cohort TaskSpec conversion release",
        use=source.intended_use.value,
    )
    return ConversionOutcome(
        source_id=source.info.id,
        cohort_repo=dataset.name,
        cohort_revision=dataset.revision,
        input_rows=manifest["input_rows"],
        converted_rows=manifest["converted_rows"],
        task_specs_sha256=manifest["task_specs_sha256"],
        receipts_path=str(output / RECEIPTS_FILENAME),
        release_repo=publication.repo,
        release_revision=publication.revision,
        license_check=publication.license_check,
    )


def converter_revision(explicit: str | None = None) -> str:
    """The landed revision the conversion runs from: the bootstrap's archive revision, or the checkout."""
    if explicit:
        return explicit
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError("the conversion cannot read its own checkout revision")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="The Harbor cohort source to convert.")
    parser.add_argument("--output-root", required=True, help="The conversion output directory (local scratch).")
    parser.add_argument("--artifact-version", required=True, help="The release version suffix.")
    parser.add_argument("--limit", type=int, help="Convert only the first N rows (local validation slices).")
    parser.add_argument(
        "--converter-revision",
        help="The landed revision the conversion runs from; the bootstrap passes the archive revision "
        "because the extracted tree has no git checkout.",
    )
    parser.add_argument(
        "--run", action="store_true", help="Publish the private release; omit to validate and plan only."
    )
    parser.add_argument("--report", type=Path, help="Write the result JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from experiments.post_training.task_curation.sources import all_sources  # noqa: PLC0415

    source = all_sources().get(args.source_id)
    if source is None:
        raise SystemExit(f"Unknown source: {args.source_id}")
    require_conversion_source(source)
    converted, receipts, counts = convert_population(source, limit=args.limit)
    manifest = write_conversion_artifacts(
        converted, receipts, counts, output_root=args.output_root, source_id=source.info.id
    )
    manifest["converter_revision"] = converter_revision(args.converter_revision)
    manifest["mode"] = "publish" if args.run else "plan"
    manifest["cohort"] = {"repo": source.dataset.name, "revision": source.dataset.revision}
    if not args.run:
        print(
            json.dumps(
                {
                    "plan": manifest,
                    "release_repo": release_repo(source.info.id, args.artifact_version),
                    "output_root": args.output_root,
                },
                indent=2,
            )
        )
        return
    outcome = publish_conversion(source, output_root=args.output_root, version=args.artifact_version, manifest=manifest)
    result = {
        **manifest,
        "release_repo": outcome.release_repo,
        "release_revision": outcome.release_revision,
        "task_specs_sha256": outcome.task_specs_sha256,
        "receipts_path": outcome.receipts_path,
        "license_check": outcome.license_check,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
