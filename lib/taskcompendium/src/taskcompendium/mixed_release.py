# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a local demonstration dataset from complete accepted task records."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.models import SCHEMA_VERSION, SHA256_PATTERN, TaskSpec
from taskcompendium.path_validation import validate_relative_file_path
from taskcompendium.release_common import REPO_ID, sha256_file

NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
REGIONAL_PIN_PATTERN = re.compile(
    r"[^#]+#manifest-sha256=[0-9a-f]{64}#parquet-size=\d+#parquet-etag=([^#]*)#parquet-version-id=([^#]*)\Z"
)
WORKPLACE_CONFIG = "workplace"
TASKTROVE_CONFIG = "tasktrove_clean"
MANIFEST_FILENAME = "manifest.json"
CARD_FILENAME = "README.md"
ConfigName = Literal["workplace", "tasktrove_clean"]
SplitName = Literal["train", "validation"]
CONFIG_ORDER = (WORKPLACE_CONFIG, TASKTROVE_CONFIG)


def _valid_source_pin(pin: str) -> bool:
    if pin.startswith("sha256:") and SHA256_PATTERN.fullmatch(pin[7:]):
        return True
    match = REGIONAL_PIN_PATTERN.fullmatch(pin)
    return match is not None and bool(match[1] or match[2])


class SourceProof(BaseModel):
    """Row and source-archive proof retained with a public task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_row: str
    source_row_sha256: str | None = None
    input_file: str
    input_object_pin: str
    archive_path: str | None = None
    archive_sha256: str | None = None

    @model_validator(mode="after")
    def validate_proof(self) -> "SourceProof":
        if not all((self.source_row, self.input_file, self.input_object_pin)):
            raise ValueError("Source proof requires a row ID and pinned input file")
        if not _valid_source_pin(self.input_object_pin):
            raise ValueError("Source proof input object needs an immutable pin")
        if self.source_row_sha256 is not None and not SHA256_PATTERN.fullmatch(self.source_row_sha256):
            raise ValueError("Source row digest must be SHA256")
        if (self.archive_path is None) != (self.archive_sha256 is None):
            raise ValueError("Archive path and digest must be supplied together")
        if self.archive_sha256 is not None and not SHA256_PATTERN.fullmatch(self.archive_sha256):
            raise ValueError("Archive digest must be SHA256")
        return self


class AcceptedTaskRecord(BaseModel):
    """A complete task with reviewed source category and immutable row proof."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskSpec
    source_category: str | None = None
    source_proof: SourceProof


class PublishedRow(TaskSpec):
    """Complete demonstration task fields and source provenance on the Hub."""

    record_version: Literal[3] = 3
    source_category: str | None = None
    provenance: SourceProof


def published_task(row: PublishedRow) -> TaskSpec:
    return TaskSpec.model_validate(row.model_dump(exclude={"record_version", "source_category", "provenance"}))


class CatalogJoinEvidence(BaseModel):
    """Exact private catalog and ledger pins behind a complete-task conversion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_sha256: str
    ledger_sha256: str
    input_schema_version: Literal["0.13"] = "0.13"
    output_schema_version: str
    joined_rows: int
    verifier_matches: int
    projected_field_matches: int


class SourceRights(BaseModel):
    """Provenance and redistribution evidence for one included cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    license: str
    license_url: str
    attribution: str
    source_card_url: str
    source_card_revision: str
    change_notice: str

    @model_validator(mode="after")
    def validate_rights(self) -> "SourceRights":
        if not all(
            (
                self.license,
                self.license_url,
                self.attribution,
                self.source_card_url,
                self.source_card_revision,
                self.change_notice,
            )
        ):
            raise ValueError("Each cohort needs license, attribution, pinned card, and change notice")
        if not self.license_url.startswith("https://") or not self.source_card_url.startswith("https://"):
            raise ValueError("License and source card URLs must use HTTPS")
        if not GIT_SHA_PATTERN.fullmatch(self.source_card_revision):
            raise ValueError("Source card revision must be a full Git commit")
        return self


class HarborSample(BaseModel):
    """A bounded Harbor trial record for an included source cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_url: str
    taskcompendium_revision: str
    harbor_revision: str
    trials: int = Field(gt=0)
    coverage: str

    @model_validator(mode="after")
    def validate_sample(self) -> "HarborSample":
        if not self.evidence_url.startswith("https://") or not self.coverage:
            raise ValueError("Harbor samples need evidence URL and coverage")
        for revision in (self.taskcompendium_revision, self.harbor_revision):
            if not GIT_SHA_PATTERN.fullmatch(revision):
                raise ValueError("Harbor sample revisions must be full Git commits")
        return self


class ProviderPin(BaseModel):
    """Selected runtime binding, retained only in the release manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    locator: str
    provider_revision: str
    action_interface: str
    seed_sha256: str
    tools_sha256: str
    tools: tuple[str, ...]

    @model_validator(mode="after")
    def validate_pin(self) -> "ProviderPin":
        if not all((self.name, self.locator, self.provider_revision, self.action_interface)):
            raise ValueError("Provider pin needs its name, locator, revision, and action interface")
        if not SHA256_PATTERN.fullmatch(self.seed_sha256) or not SHA256_PATTERN.fullmatch(self.tools_sha256):
            raise ValueError("Provider seed and tool surface need SHA256 digests")
        return self


class SourceAsset(BaseModel):
    """One pinned source object that contains accepted rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    pin: str

    @model_validator(mode="after")
    def validate_asset(self) -> "SourceAsset":
        if not self.path or not self.pin:
            raise ValueError("Source assets need a path and immutable pin")
        if not _valid_source_pin(self.pin):
            raise ValueError("Source asset needs a SHA256 or pinned regional object")
        return self


class CohortInput(BaseModel):
    """Trusted local input for a rights-cleared, Harbor-sampled public cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: ConfigName
    cohort: str
    split: SplitName
    input_path: Path
    input_sha256: str
    accepted_rows: int = Field(gt=0)
    source_records: int = Field(gt=0)
    parsed_rows: int | None = Field(default=None, gt=0)
    source_dataset: str
    source_revision: str | None = None
    source_subset: str | None = None
    source_category: str | None = None
    projection_manifest_sha256: str | None = None
    source_assets: tuple[SourceAsset, ...] = Field(min_length=1)
    task_spec_schema: str
    importer_revision: str
    projection_builder_revision: str
    rights: SourceRights
    harbor_samples: tuple[HarborSample, ...] = Field(min_length=1)
    provider_pins: tuple[ProviderPin, ...] = ()
    catalog_join: CatalogJoinEvidence | None = None

    @model_validator(mode="after")
    def validate_cohort(self) -> "CohortInput":
        if not NAME_PATTERN.fullmatch(self.cohort):
            raise ValueError("Cohort name must be a path-safe slug")
        if self.accepted_rows > self.source_records:
            raise ValueError("Accepted rows cannot exceed source rows")
        if self.parsed_rows is not None and not self.accepted_rows <= self.parsed_rows <= self.source_records:
            raise ValueError("Parsed row count must lie between accepted and source counts")
        if not SHA256_PATTERN.fullmatch(self.input_sha256):
            raise ValueError("Accepted input needs a SHA256 digest")
        if not all((self.source_dataset, self.task_spec_schema)):
            raise ValueError("Cohort source and schema pins are required")
        if not self.importer_revision or not GIT_SHA_PATTERN.fullmatch(self.projection_builder_revision):
            raise ValueError("Importer and projection builder pins are required")
        if self.config == WORKPLACE_CONFIG and (self.source_revision is None or len(self.source_assets) != 1):
            raise ValueError("Workplace needs its pinned source revision and one split file")
        if self.config == TASKTROVE_CONFIG and (not self.source_subset or not self.source_category):
            raise ValueError("TaskTrove cohorts need an exact source subset and category")
        if self.config == TASKTROVE_CONFIG and (
            self.projection_manifest_sha256 is None or not SHA256_PATTERN.fullmatch(self.projection_manifest_sha256)
        ):
            raise ValueError("TaskTrove cohorts need the accepted projection manifest SHA256")
        if self.config == TASKTROVE_CONFIG:
            if self.catalog_join is None:
                raise ValueError("TaskTrove full tasks require private catalog join evidence")
            evidence = self.catalog_join
            if (
                evidence.output_schema_version != SCHEMA_VERSION
                or evidence.joined_rows != self.accepted_rows
                or evidence.verifier_matches != self.accepted_rows
                or evidence.projected_field_matches != self.accepted_rows
                or not SHA256_PATTERN.fullmatch(evidence.catalog_sha256)
                or not SHA256_PATTERN.fullmatch(evidence.ledger_sha256)
            ):
                raise ValueError("TaskTrove catalog join evidence differs from its accepted cohort")
        if self.config == WORKPLACE_CONFIG and self.projection_manifest_sha256 is not None:
            raise ValueError("Workplace cohorts do not use the regional TaskTrove projection manifest")
        if len({pin.name for pin in self.provider_pins}) != len(self.provider_pins):
            raise ValueError("Provider pin names must be unique")
        if len({(asset.path, asset.pin) for asset in self.source_assets}) != len(self.source_assets):
            raise ValueError("Source asset pins must be unique")
        return self


def _validated_record(
    raw: str,
    cohort: CohortInput,
    source_assets: set[tuple[str, str]],
    provider_pins: dict[str, ProviderPin],
) -> AcceptedTaskRecord:
    record = AcceptedTaskRecord.model_validate_json(raw)
    task, proof = record.task, record.source_proof
    if task.source.dataset != cohort.source_dataset:
        raise ValueError("Public row source does not match cohort pin")
    if cohort.source_revision is not None and task.source.revision != cohort.source_revision:
        raise ValueError("Public row source revision does not match cohort pin")
    if task.source.importer_revision != cohort.importer_revision or task.source.row != proof.source_row:
        raise ValueError("Public row does not match its importer or source proof")
    if (proof.input_file, proof.input_object_pin) not in source_assets:
        raise ValueError("Public row input object does not match cohort source assets")
    if set(task.tool_providers) != set(provider_pins):
        raise ValueError("Public provider requirements do not match cohort runtime pins")
    for name, requirement in task.tool_providers.items():
        pin = provider_pins[name]
        if requirement.action_interface != pin.action_interface or requirement.seed_sha256 != pin.seed_sha256:
            raise ValueError("Public provider interface or seed differs from cohort runtime pin")
    if cohort.config == TASKTROVE_CONFIG:
        if not task.tags:
            raise ValueError("TaskTrove rows must retain original ordered source tags")
        if record.source_category != cohort.source_category or not task.source.row.startswith(
            f"{cohort.source_subset}:"
        ):
            raise ValueError("TaskTrove row does not belong to the licensed source cohort")
        if proof.archive_path is None or proof.archive_sha256 is None:
            raise ValueError("TaskTrove rows need archive path and digest")
        if not task.source.row.endswith(f":{proof.archive_path}") or task.source.revision != proof.input_object_pin:
            raise ValueError("TaskTrove archive proof differs from row provenance")
    else:
        parts = task.source.row.split(":")
        if (
            len(parts) != 3
            or parts[0] != cohort.split
            or not parts[1].isdigit()
            or parts[2] != proof.source_row_sha256
            or proof.archive_path is not None
        ):
            raise ValueError("Workplace row must retain its split-local raw row digest")
    return record


def _public_asset_path(path: str, cohort: CohortInput) -> str:
    if cohort.config == TASKTROVE_CONFIG and cohort.source_dataset.startswith("s3://"):
        prefix = f"{cohort.source_dataset.rstrip('/')}/"
        if not path.startswith(prefix):
            raise ValueError("TaskTrove source asset lies outside the pinned release")
        path = path.removeprefix(prefix)
    validate_relative_file_path(path)
    return path


def _published_row(record: AcceptedTaskRecord, cohort: CohortInput) -> PublishedRow:
    """Convert a reviewed full task to a demonstration row without changing its grader."""
    task = record.task
    proof = record.source_proof.model_copy(
        update={"input_file": _public_asset_path(record.source_proof.input_file, cohort)}
    )
    if proof.archive_path is not None:
        validate_relative_file_path(proof.archive_path)
    source = (
        task.source.model_copy(update={"dataset": TASKTROVE_CONFIG})
        if cohort.config == TASKTROVE_CONFIG
        else task.source
    )
    row = PublishedRow(
        **task.model_dump(exclude={"source"}),
        source=source,
        source_category=record.source_category,
        provenance=proof,
    )
    if row.verifier != task.verifier:
        raise ValueError("Published row changed the source verifier")
    return row


def _write_cohort(
    cohort: CohortInput,
    destination: Path,
    seen_ids: set[str],
    seen_source_rows: set[tuple[str, str]],
) -> tuple[int, str]:
    if sha256_file(cohort.input_path) != cohort.input_sha256:
        raise ValueError(f"Accepted input digest mismatch: {cohort.cohort}")
    count = 0
    source_assets = {(asset.path, asset.pin) for asset in cohort.source_assets}
    provider_pins = {pin.name: pin for pin in cohort.provider_pins}
    with (
        cohort.input_path.open(encoding="utf-8") as source,
        destination.open("w", encoding="utf-8", newline="\n") as output,
    ):
        for line in source:
            record = _validated_record(line, cohort, source_assets, provider_pins)
            if record.task.id in seen_ids:
                raise ValueError(f"Duplicate task ID across cohorts: {record.task.id}")
            source_key = (record.task.source.dataset, record.source_proof.source_row)
            if source_key in seen_source_rows:
                raise ValueError(f"Duplicate source row across cohorts: {record.source_proof.source_row}")
            seen_ids.add(record.task.id)
            seen_source_rows.add(source_key)
            output.write(_published_row(record, cohort).model_dump_json() + "\n")
            count += 1
    if count != cohort.accepted_rows:
        raise ValueError(f"Accepted row count mismatch: {cohort.cohort}")
    return count, sha256_file(destination)


def _card(cohorts: tuple[CohortInput, ...], data_files: list[dict[str, object]]) -> str:
    licenses = {cohort.rights.license for cohort in cohorts}
    license_label = next(iter(licenses)) if len(licenses) == 1 else "other"
    lines = ["---", "pretty_name: TaskCompendium Alpha 1 Candidate", f"license: {license_label}", "configs:"]
    for config in (WORKPLACE_CONFIG, TASKTROVE_CONFIG):
        entries = [entry for entry in data_files if entry["config"] == config]
        if not entries:
            continue
        lines.extend((f"  - config_name: {config}", "    data_files:"))
        splits = sorted({str(entry["split"]) for entry in entries})
        for split in splits:
            paths = [str(entry["path"]) for entry in entries if entry["split"] == split]
            lines.append(f"      - split: {split}")
            if len(paths) == 1:
                lines.append(f"        path: {paths[0]}")
            else:
                lines.append("        path:")
                lines.extend(f"          - {path}" for path in paths)
    lines.extend(("---", "", "# TaskCompendium Alpha 1 Candidate", ""))
    lines.append(
        "This local demonstration candidate contains complete accepted TaskSpecs. The manifest records exported counts, "
        "source pins, rights, and Harbor sample evidence. Each row retains its actual verifier, "
        "including reference answers "
        "or expected state. Runtime agent projection keeps grading material out of model input. "
        "Rows place source pins in `provenance` and omit submission instructions. "
        "The builder does not upload to the Hub."
    )
    lines.extend(
        (
            "",
            "Workplace and TaskTrove Clean retain separate source splits. Each cohort below credits its source; "
            "complete pins and counts are in `manifest.json`.",
            "",
        )
    )
    for cohort in cohorts:
        lines.append(
            f"- `{cohort.config}/{cohort.cohort}` ({cohort.split}): {cohort.rights.attribution}, "
            f"[{cohort.rights.license}]({cohort.rights.license_url}); "
            f"[source card]({cohort.rights.source_card_url}) "
            f"at `{cohort.rights.source_card_revision}`. {cohort.rights.change_notice}"
        )
    lines.extend(
        (
            "",
            "Publication is pending a final rights and source audit. The manifest sets `publication_ready` to false.",
            "",
        )
    )
    return "\n".join(lines)


class ReleaseReview(BaseModel):
    """Evidence that a specific candidate passed publication review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_manifest_sha256: str
    rights_review_url: str
    harbor_evidence_urls: tuple[str, ...]

    @model_validator(mode="after")
    def validate_review(self) -> "ReleaseReview":
        if not SHA256_PATTERN.fullmatch(self.candidate_manifest_sha256):
            raise ValueError("Release review must pin the candidate manifest SHA256")
        if not self.rights_review_url.startswith("https://"):
            raise ValueError("Release review needs an HTTPS rights review reference")
        if not self.harbor_evidence_urls or any(not url.startswith("https://") for url in self.harbor_evidence_urls):
            raise ValueError("Release review needs HTTPS Harbor evidence for every source")
        return self


def finalize_mixed_candidate(candidate: Path, destination: Path, review: ReleaseReview) -> Path:
    """Create a distinct publication-ready artifact after review of the exact candidate."""
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("Ready artifact must be outside the candidate directory")
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("Candidate must be a regular directory")
    manifest_path = candidate / MANIFEST_FILENAME
    if sha256_file(manifest_path) != review.candidate_manifest_sha256:
        raise ValueError("Release review does not match the candidate manifest")
    manifest = json.loads(manifest_path.read_text())
    data_paths = _candidate_data_paths(candidate, manifest)
    _validate_publication_evidence(candidate, manifest, review)
    return _write_ready_artifact(candidate, destination, manifest, review, data_paths)


def _candidate_data_paths(candidate: Path, manifest: dict[str, Any]) -> set[str]:
    """Validate the candidate's exact file inventory and return its listed data paths."""
    data_paths: set[str] = set()
    for entry in manifest["data_files"]:
        path = entry["path"]
        if not isinstance(path, str):
            raise ValueError("Candidate data paths must be strings")
        relative_path = PurePosixPath(path)
        if (
            not path
            or path != relative_path.as_posix()
            or relative_path.is_absolute()
            or not relative_path.parts
            or relative_path.parts[0] != "data"
            or any(part in ("", ".", "..") for part in relative_path.parts)
        ):
            raise ValueError(f"Candidate data path is not a normalized relative path: {path}")
        if path in data_paths:
            raise ValueError(f"Candidate data path is duplicated: {path}")
        data_paths.add(path)
    allowed_files = {CARD_FILENAME, MANIFEST_FILENAME, *data_paths}
    allowed_directories: set[str] = set()
    for path in data_paths:
        parent = PurePosixPath(path).parent
        while parent != PurePosixPath("."):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
    seen_files: set[str] = set()
    for item in candidate.rglob("*"):
        relative_path = item.relative_to(candidate).as_posix()
        if item.is_symlink():
            raise ValueError(f"Candidate contains a symlink: {relative_path}")
        if item.is_dir():
            if relative_path not in allowed_directories:
                raise ValueError(f"Candidate contains an unlisted directory: {relative_path}")
        elif not item.is_file() or relative_path not in allowed_files:
            raise ValueError(f"Candidate contains an unlisted or unsupported file: {relative_path}")
        else:
            seen_files.add(relative_path)
    if seen_files != allowed_files:
        raise ValueError(f"Candidate file inventory differs from its manifest: {sorted(allowed_files - seen_files)}")
    return data_paths


def _validate_publication_evidence(candidate: Path, manifest: dict[str, Any], review: ReleaseReview) -> None:
    """Check reviewed Harbor evidence, export counts, and the candidate data digests."""
    if manifest["publication_ready"]:
        raise ValueError("Candidate is already publication-ready")
    expected_harbor_urls = {
        sample["evidence_url"] for entry in manifest["data_files"] for sample in entry["harbor_samples"]
    }
    if set(review.harbor_evidence_urls) != expected_harbor_urls:
        raise ValueError("Release review Harbor evidence must match every candidate cohort")
    if not manifest["data_files"] or any(
        entry["accepted_rows"] != entry["exported_rows"] for entry in manifest["data_files"]
    ):
        raise ValueError("Publication requires every accepted row to be exported")
    for entry in manifest["data_files"]:
        data_path = candidate / entry["path"]
        if not data_path.is_file() or sha256_file(data_path) != entry["sha256"]:
            raise ValueError(f"Candidate data file does not match its manifest: {entry['path']}")


def _write_ready_artifact(
    candidate: Path,
    destination: Path,
    manifest: dict[str, Any],
    review: ReleaseReview,
    data_paths: set[str],
) -> Path:
    """Copy only reviewed files and write a ready manifest and card."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copyfile(candidate / MANIFEST_FILENAME, temporary / MANIFEST_FILENAME)
        for path in sorted(data_paths):
            source_path = candidate / path
            output_path = temporary / path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, output_path)
        for entry in manifest["data_files"]:
            if sha256_file(temporary / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Candidate data file changed during finalization: {entry['path']}")
        manifest["publication_ready"] = True
        manifest["publication_review"] = {
            "candidate_manifest_sha256": review.candidate_manifest_sha256,
            "rights_review_url": review.rights_review_url,
            "harbor_evidence_urls": sorted(review.harbor_evidence_urls),
        }
        (temporary / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / CARD_FILENAME).write_text(_publication_card(manifest, review))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _publication_card(manifest: dict[str, Any], review: ReleaseReview) -> str:
    """Render the ready card from manifest data so candidate prose is not trusted."""
    data_files = manifest["data_files"]
    licenses = {entry["rights"]["license"] for entry in data_files}
    license_label = next(iter(licenses)) if len(licenses) == 1 else "other"
    lines = ["---", "pretty_name: TaskCompendium Alpha 1", f"license: {license_label}", "configs:"]
    for config in (WORKPLACE_CONFIG, TASKTROVE_CONFIG):
        entries = [entry for entry in data_files if entry["config"] == config]
        if not entries:
            continue
        lines.extend((f"  - config_name: {config}", "    data_files:"))
        for split in sorted({entry["split"] for entry in entries}):
            paths = [entry["path"] for entry in entries if entry["split"] == split]
            lines.append(f"      - split: {split}")
            if len(paths) == 1:
                lines.append(f"        path: {paths[0]}")
            else:
                lines.append("        path:")
                lines.extend(f"          - {path}" for path in paths)
    lines.extend(("---", "", "# TaskCompendium Alpha 1", ""))
    lines.append(
        "This demonstration release contains complete accepted TaskSpecs and their actual verifiers, including "
        "reference answers or expected state. Runtime agent projection keeps grading material out of model input. "
        "Rows place source pins in `provenance` and omit submission instructions."
    )
    lines.append("")
    for entry in data_files:
        rights = entry["rights"]
        lines.append(
            f"- `{entry['config']}/{entry['cohort']}` ({entry['split']}): {entry['exported_rows']} rows; "
            f"{rights['attribution']}, [{rights['license']}]({rights['license_url']}); "
            f"[source card]({rights['source_card_url']}) "
            f"at `{rights['source_card_revision']}`. {rights['change_notice']}"
        )
    lines.extend(("", f"Rights review: {review.rights_review_url}. Harbor evidence is recorded in `manifest.json`."))
    return "\n".join(lines) + "\n"


def assemble_mixed_candidate(cohorts: tuple[CohortInput, ...], destination: Path, *, builder_revision: str) -> Path:
    """Validate and atomically write accepted cohorts into separate Hub configs."""
    if not GIT_SHA_PATTERN.fullmatch(builder_revision):
        raise ValueError("Builder revision must be a full Git commit")
    if not cohorts or destination.exists():
        raise ValueError("Nonempty cohorts and a new destination are required")
    ordered = tuple(sorted(cohorts, key=lambda cohort: (CONFIG_ORDER.index(cohort.config), cohort.split, cohort.cohort)))
    if len({(cohort.config, cohort.split, cohort.cohort) for cohort in ordered}) != len(ordered):
        raise ValueError("Cohort output paths must be unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        data_files: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        seen_source_rows: set[tuple[str, str]] = set()
        for cohort in ordered:
            path = Path("data") / cohort.config / f"{cohort.cohort}.jsonl"
            output = temporary / path
            output.parent.mkdir(parents=True, exist_ok=True)
            rows, digest = _write_cohort(cohort, output, seen_ids, seen_source_rows)
            data_files.append(
                {
                    "config": cohort.config,
                    "cohort": cohort.cohort,
                    "split": cohort.split,
                    "path": path.as_posix(),
                    "sha256": digest,
                    "source_records": cohort.source_records,
                    "parsed_rows": cohort.parsed_rows,
                    "accepted_rows": cohort.accepted_rows,
                    "exported_rows": rows,
                    "source": {
                        "dataset": TASKTROVE_CONFIG if cohort.config == TASKTROVE_CONFIG else cohort.source_dataset,
                        "revision": cohort.source_revision,
                        "subset": cohort.source_subset,
                        "category": cohort.source_category,
                        "assets": [
                            {"path": _public_asset_path(asset.path, cohort), "pin": asset.pin}
                            for asset in cohort.source_assets
                        ],
                        "task_spec_schema": cohort.task_spec_schema,
                        "catalog_join": cohort.catalog_join.model_dump(mode="json") if cohort.catalog_join else None,
                        "importer_revision": cohort.importer_revision,
                        "projection_builder_revision": cohort.projection_builder_revision,
                        "projection_manifest_sha256": cohort.projection_manifest_sha256,
                    },
                    "rights": cohort.rights.model_dump(mode="json"),
                    "harbor_samples": [sample.model_dump(mode="json") for sample in cohort.harbor_samples],
                    "provider_pins": [pin.model_dump(mode="json") for pin in cohort.provider_pins],
                }
            )
        manifest = {
            "format_version": 1,
            "public_record_version": 3,
            "task_spec_schema": SCHEMA_VERSION,
            "repo_id": REPO_ID,
            "visibility": "demonstration",
            "builder_revision": builder_revision,
            "publication_ready": False,
            "data_files": data_files,
        }
        (temporary / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / CARD_FILENAME).write_text(_card(ordered, data_files))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a local complete-task demonstration candidate")
    parser.add_argument("--cohorts", type=Path, required=True, help="JSON array of accepted cohort inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder-revision", required=True)
    arguments = parser.parse_args()
    cohorts = tuple(CohortInput.model_validate(item) for item in json.loads(arguments.cohorts.read_text()))
    assemble_mixed_candidate(cohorts, arguments.output, builder_revision=arguments.builder_revision)


def finalize_main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a reviewed TaskCompendium candidate for publication")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True, help="JSON ReleaseReview bound to candidate manifest")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    review = ReleaseReview.model_validate_json(arguments.review.read_text())
    finalize_mixed_candidate(arguments.candidate, arguments.output, review)


if __name__ == "__main__":
    main()
