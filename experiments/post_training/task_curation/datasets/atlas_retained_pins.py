# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind Atlas-retained release pins onto main's configured declarations.

The Atlas seals every retained release with a receipt in its pinned private
registry (``open-athena/marin-rl-data-atlas-registry``); the receipt records the
retained pin, the release bytes it read back, and the censuses that measured the
original and retained populations. The pin-level content of those receipts
(never dataset bytes) lives in the public dataset ``open-athena/atlas-source-pins``;
this module commits only the immutable reference and binds each entry onto main's
raw declaration at ``all_sources()`` time, so main's durable reference is the
Atlas admitted version and selection runs the retained reader.

The binding re-validates the lineage fail-closed: the declaration must be the
exact conversion input the receipt sealed (id, dataset pin, converter version and
pipeline reference), and the release must be the private open-athena TaskCompendium
publication the lexical receipt qualifies. A declaration that drifted from its
entry keeps its raw pin with a recorded note instead of blocking the catalog; the
pin-parity test then fails on the drifted pin. A table that cannot be fetched
(an offline run) binds nothing and leaves every declaration at its raw pin; a
fetched table that fails its digest check fails closed.

Refresh ``retained_pins.json`` from an audit checkout that carries the
receipt models (maintainer-run, reads the private registry)::

    uv run --isolated --frozen python -m \\
        experiments.post_training.task_curation.datasets.atlas_retained_pins \\
        --refresh --audit-root /tmp/atlas-cut
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from functools import cache
from pathlib import Path

from finestore.rl.curation import LexicalContaminationEvidence, LexicalUse, PipelineReference, SourceUse
from finestore.rl.licenses import LicenseReceipt
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, model_validator
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.retained_release import (
    RETAINED_IMPORT_VERSION,
    RetainedCohort,
    import_retained,
)
from experiments.post_training.task_curation.source import DatasetRelease, RlDataSource, SourceReference
from infra.marina.apps.rl_data_catalog.server.admission import require_lexical_qualification

RECEIPTS_REPO = "open-athena/marin-rl-data-atlas-registry"
PINS_REPO = "open-athena/atlas-source-pins"
PINS_REVISION = "5ab081a1db11de1f5a3fa07563b8944b86d15541"
PINS_SHA256 = "baabdcd8ef3904f6fb910f120920ba50e4f5a76f147f71de5456ddfe755ee013"
PINS_BYTES = 1539862
PINS_FILENAME = "retained_pins.json"

# Runs inside the audit checkout, where the sealed receipt models live: every
# retained-release shard reference is downloaded, digest-verified and validated,
# then reduced to the pin-level content the reader binds. The lexical evidence is
# the publication's own release proof when it admits the retained pin, else the
# frozen lexical measurement measured at that pin.
RESOLVE_SCRIPT = """
import json
from pathlib import Path

from experiments.post_training.task_curation.datasets.lexical_measurements import lexical_measurements
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.registry import RegistryReference, read_registry_document
from experiments.post_training.task_curation.remediation_binding import (
    RetainedReleaseReceipt,
    release_lexical_evidence,
)
from finestore.rl.curation import DeconReceiptStatus, LexicalUse

LICENSES = read_license_receipts(LICENSE_RECEIPT_INDEX)
LEXICAL = lexical_measurements()


def license_receipt_for(source_id, repo, revision):
    receipt = LICENSES.get(source_id)
    if receipt is None or (receipt.dataset_id, receipt.revision) != (repo, revision):
        # A retained release whose license receipt does not certify its exact pin
        # cannot pass the stage-0 gate on main; the resolution skips it and the
        # declaration stays at its raw pin.
        return None
    return receipt


LEXICAL = lexical_measurements()


def admits(evidence, repo, revision, use):
    # The admission predicate the retained import runs at the released pin.
    if use is LexicalUse.TRAIN:
        return evidence.admission_blocker(repo, revision) is None
    receipt = evidence.receipt
    return (
        receipt is not None
        and receipt.use in {LexicalUse.EVAL, LexicalUse.BENCHMARK}
        and receipt.status is DeconReceiptStatus.BENCHMARK_MEASURED
        and (evidence.input_dataset_id, evidence.input_dataset_revision) == (repo, revision)
    )


def lexical_evidence_for_release(source_id, publication, revision, use):
    release = publication.release
    evidence = release_lexical_evidence(publication)
    if admits(evidence, release.repo, revision, use):
        return evidence
    receipt = LEXICAL.get(source_id)
    if receipt is None:
        raise ValueError(f"no lexical measurement admits the retained pin: {source_id}")
    if (receipt.input.dataset_id, receipt.input.dataset_revision) != (release.repo, revision):
        raise ValueError(f"lexical measurement covers another pin: {source_id}")
    if not admits(receipt.measurement, release.repo, revision, use):
        raise ValueError(f"lexical measurement does not admit the retained pin: {source_id}")
    return receipt.measurement

rows = {}
skipped = []
for part in sorted(Path("experiments/post_training/task_curation/datasets/retained_release_receipts").glob("part-*.json")):
    reference = RegistryReference.model_validate_json(part.read_bytes())
    document = read_registry_document(reference)
    for source_id, payload in document.items():
        receipt = RetainedReleaseReceipt.model_validate(payload)
        publication = receipt.publication
        plan = publication.plan
        measurement = receipt.actor.measurement
        dedup = receipt.retained_dedup
        rows[source_id] = {
            "source_id": source_id,
            "original_repo": plan.source_dataset_id,
            "original_revision": plan.source_revision,
            "converter_version": plan.converter_version,
            "original_pipeline": receipt.original_pipeline.model_dump(mode="json"),
            "original_census": {
                "dataset_id": plan.source_dataset_id,
                "dataset_revision": plan.source_revision,
                "converter_version": plan.converter_version,
                "measured_tasks": plan.raw_count,
                "task_specs_sha256": plan.task_specs_sha256,
            },
            "retained_census": {
                "dataset_id": receipt.actor.input.dataset_id,
                "dataset_revision": receipt.actor.input.dataset_revision,
                "converter_version": receipt.actor.input.converter_version,
                "measured_tasks": receipt.actor.input.measured_tasks,
                "snapshot_count": measurement.daytona_snapshot_count,
                "count_basis": measurement.daytona_count_basis,
                "execution_failures": measurement.daytona_execution_failures,
            },
            "release_repo": publication.repo,
            "release_revision": publication.revision,
            "retained_count": plan.retained_count,
            "retained_dedup_unique_count": dedup.unique_count,
            "retained_dedup_duplicates": dedup.within_source_duplicates,
            "tasks_parquet_sha256": publication.tasks_parquet_sha256,
            "lexical_use": plan.lexical_use.value,
            "lexical": lexical_evidence_for_release(
                source_id, publication, publication.revision, plan.lexical_use
            ).model_dump(mode="json"),
            "evidence_url": publication.evidence_url,
            "registry_revision": reference.revision,
            "registry_path": reference.path,
            "registry_sha256": reference.sha256,
            "notes": "",
        }
        license_receipt = license_receipt_for(source_id, publication.repo, publication.revision)
        if license_receipt is None:
            skipped.append(source_id)
            del rows[source_id]
            continue
        rows[source_id]["license"] = license_receipt.model_dump(mode="json")
print(json.dumps({"pins": rows, "skipped_no_license": skipped}, indent=1, sort_keys=True))
"""


class RetainedCensus(BaseModel):
    """The population a census measured, pinned to its conversion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    dataset_revision: str
    converter_version: str
    measured_tasks: int = Field(gt=0)
    task_specs_sha256: str | None = None
    snapshot_count: int | None = None
    count_basis: str | None = None
    execution_failures: int | None = None


class RetainedPin(BaseModel):
    """The pin-level content of one sealed retained-release receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    original_repo: str
    original_revision: str
    converter_version: str
    original_pipeline: PipelineReference
    original_census: RetainedCensus
    retained_census: RetainedCensus
    release_repo: str
    release_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    retained_count: int = Field(gt=0)
    retained_dedup_unique_count: int
    retained_dedup_duplicates: int
    tasks_parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lexical_use: LexicalUse
    lexical: LexicalContaminationEvidence
    license: LicenseReceipt
    evidence_url: str
    registry_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    registry_path: str
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    notes: str = ""

    @property
    def intended_use(self) -> IntendedUse:
        return IntendedUse.TRAIN if self.lexical_use is LexicalUse.TRAIN else IntendedUse.EVAL

    @property
    def release(self) -> DatasetRelease:
        from finestore.rl.curation import huggingface_dataset_url

        return DatasetRelease(
            repo=self.release_repo,
            revision=self.release_revision,
            url=huggingface_dataset_url(self.release_repo, self.release_revision),
        )

    @model_validator(mode="after")
    def _sealed_lineage(self) -> "RetainedPin":
        if self.original_census.dataset_id != self.original_repo or (
            self.original_census.dataset_revision != self.original_revision
        ):
            raise ValueError(f"Original census covers another pin: {self.source_id}")
        if self.retained_census.dataset_id != self.release_repo or (
            self.retained_census.dataset_revision != self.release_revision
        ):
            raise ValueError(f"Retained census covers another pin: {self.source_id}")
        if self.retained_census.converter_version != RETAINED_IMPORT_VERSION:
            raise ValueError(f"Retained census was not measured at the retained importer: {self.source_id}")
        if self.retained_census.measured_tasks != self.retained_count or (
            self.retained_dedup_unique_count != self.retained_count
        ):
            raise ValueError(f"Retained census does not cover the retained population: {self.source_id}")
        if self.retained_dedup_duplicates != 0:
            raise ValueError(f"The retained population is not the deduplicated population: {self.source_id}")
        if not self.release_repo.startswith("open-athena/"):
            raise ValueError(f"Retained release is not a private open-athena publication: {self.source_id}")
        if (self.license.dataset_id, self.license.revision) != (self.release_repo, self.release_revision):
            raise ValueError(f"License receipt certifies another pin: {self.source_id}")
        require_lexical_qualification(
            self.lexical, self.release_repo, self.release_revision, SourceUse(self.intended_use.value)
        )
        return self

    def cohort(self, *, name: str) -> RetainedCohort:
        """The retained-reader configuration for one bound declaration."""
        from finestore.rl.curation import huggingface_dataset_url

        return RetainedCohort(
            name=name + "-retained",
            version=RETAINED_IMPORT_VERSION,
            release=self.release,
            original_dataset=SourceReference(
                self.original_repo,
                self.original_revision,
                huggingface_dataset_url(self.original_repo, self.original_revision),
            ),
            expected_rows=self.retained_count,
            tasks_parquet_sha256=self.tasks_parquet_sha256,
            intended_use=self.intended_use,
            evidence_url=self.evidence_url,
        )


def _fetch_pin_table() -> bytes | None:
    """The pinned table bytes, or None when an offline run cannot fetch them."""
    import huggingface_hub.errors

    try:
        local = Path(
            hf_hub_download(repo_id=PINS_REPO, repo_type="dataset", revision=PINS_REVISION, filename=PINS_FILENAME)
        )
    except huggingface_hub.errors.HuggingFaceHubException:
        return None
    raw = local.read_bytes()
    if len(raw) != PINS_BYTES or hashlib.sha256(raw).hexdigest() != PINS_SHA256:
        raise ValueError("Retained-pin table differs from its pinned reference")
    return raw


@cache
def atlas_retained_pins() -> dict[str, RetainedPin] | None:
    """The pinned pin table, validated entry by entry at first read.

    ``None`` when the table cannot be fetched (an offline run): the caller binds
    nothing and every declaration keeps its raw pin. A fetched table that fails
    its digest check fails closed.
    """
    raw = _fetch_pin_table()
    if raw is None:
        return None
    document = json.loads(raw)
    pins = {source_id: RetainedPin.model_validate(entry) for source_id, entry in document["pins"].items()}
    if len(pins) != len(document["pins"]):
        raise ValueError("Retained-pin table repeats a source id")
    return pins


def bind_retained_pins(sources: list[RlDataSource]) -> list[RlDataSource]:
    """Swap each pinned declaration to its Atlas-retained pin and the retained reader.

    A declaration whose lineage does not match its entry keeps the raw pin with a
    recorded note; the parity test fails on the drifted pin, so the drift stays
    visible instead of blocking every other binding. Without a fetchable table
    the declarations bind nothing.
    """
    pins = atlas_retained_pins()
    if pins is None:
        return sources
    bound = []
    for source in sources:
        pin = pins.get(source.info.id)
        if pin is None:
            bound.append(source)
            continue
        try:
            bound.append(_bind(source, pin))
        except ValueError as error:
            bound.append(source_with_note(source, f"Retained pin drift: {error}"))
    return bound


def _configured_reference(source: RlDataSource, pin: RetainedPin) -> str | None:
    """Check the declaration's pipeline against the sealed one; return a version drift note.

    The dataset pin, the pipeline name and the consumed files are the lineage
    identity: a mismatch fails the binding. The converter version is a label
    main's module may have bumped since the seal; a drift binds anyway and the
    note records it, because the reader consumes the published bytes and
    re-validates every task.
    """
    dataset = source.dataset
    configured = None
    if source.pipeline is not None:
        configured = (
            source.name,
            dataset.name if dataset else None,
            dataset.revision if dataset else None,
            tuple(source.files),
        )
    sealed = pin.original_pipeline
    observed = (sealed.name, sealed.source, sealed.revision, tuple(sealed.files))
    if configured != observed:
        raise ValueError(f"retention lineage belongs to another pipeline: {source.info.id}")
    if source.version != sealed.version:
        return (
            f"retention sealed at converter version {sealed.version}; the declaration labels its "
            f"converter {source.version} — the sealed release predates that change and its re-seal is pending"
        )
    return None


def _bind(source: RlDataSource, pin: RetainedPin) -> RlDataSource:
    """Re-validate the sealed lineage and select the retained pin and reader."""
    dataset = source.dataset
    if dataset is None:
        raise ValueError(f"retention binds to an unpinned declaration: {source.info.id}")
    expected = (source.info.id, dataset.name, dataset.revision)
    observed = (pin.source_id, pin.original_repo, pin.original_revision)
    if expected != observed:
        raise ValueError(f"retention belongs to another conversion input: expected {expected}, sealed {observed}")
    version_note = _configured_reference(source, pin)
    require_lexical_qualification(
        pin.lexical, pin.release_repo, pin.release_revision, SourceUse(pin.intended_use.value)
    )

    config = pin.cohort(name=source.name)
    release = pin.release
    inherited = None
    try:
        declared = source.current_release
        if source.declaration.status == "Available":
            inherited = declared
    except ValueError:
        inherited = None
    drift = "" if version_note is None else f"{version_note}. "
    info = replace(
        source.info,
        current_release=release,
        upstream_url=source.info.upstream_url or dataset.url,
        inherited_admission=inherited,
        dataset=config.dataset,
        count=pin.retained_count,
        lexical_contamination=pin.lexical,
        license_receipt=pin.license,
        daytona_snapshot_count=pin.retained_census.snapshot_count,
        daytona_count_basis=pin.retained_census.count_basis,
        daytona_execution_failures=pin.retained_census.execution_failures,
        notes=(
            drift
            + "remediated: dedup/decon. Retained TaskSpec and runtime bytes match the original pin; "
            f"original oversight and quality evidence apply through this subset lineage: {pin.evidence_url}. "
            + source.info.notes
        ),
    )
    bound = RlDataSource(info=info, config=config, pipeline=import_retained)
    bound.require_license()
    return bound


def source_with_note(source: RlDataSource, note: str) -> RlDataSource:
    return replace(source, info=replace(source.info, notes=f"{source.info.notes} {note}".strip()))


def refresh(audit_root: Path) -> None:
    """Resolve every sealed receipt into a freshly published pin table (maintainer-run)."""
    with tempfile.TemporaryDirectory() as scratch_name:
        result = subprocess.run(
            ["uv", "run", "--isolated", "--frozen", "python", "-c", RESOLVE_SCRIPT],
            cwd=audit_root,
            check=False,
            capture_output=True,
            text=True,
        )
        scratch = Path(scratch_name) / "resolved.json"
        if result.returncode != 0:
            raise ValueError(f"receipt resolution failed: {result.stderr[-2000:]}")
        scratch.write_text(result.stdout)
        resolved = json.loads(scratch.read_text())
    document = {
        "receipts_repo": RECEIPTS_REPO,
        "resolved": "2026-11-25",
        "pins": resolved["pins"],
        "skipped_no_license": resolved["skipped_no_license"],
    }
    raw = (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()
    api = HfApi()
    info = api.upload_file(
        path_or_fileobj=raw,
        path_in_repo=PINS_FILENAME,
        repo_id=PINS_REPO,
        repo_type="dataset",
        commit_message="Refresh the resolved retained-release pin table",
    )
    match = re.search(r"/commit/([0-9a-f]{40})", info.commit_url)
    revision = match.group(1) if match else info.commit_url
    print(f"published {len(document['pins'])} retained pins ({len(document['skipped_no_license'])} skipped without a certifying license receipt); paste into atlas_retained_pins.py:")
    print(f'PINS_REVISION = "{revision}"')
    print(f'PINS_SHA256 = "{hashlib.sha256(raw).hexdigest()}"')
    print(f"PINS_BYTES = {len(raw)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="resolve the sealed receipts and rewrite the table")
    parser.add_argument("--audit-root", type=Path, required=True, help="read-only atlas-env-audit checkout")
    args = parser.parse_args()
    if not args.refresh:
        parser.print_help()
        return 1
    refresh(args.audit_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
