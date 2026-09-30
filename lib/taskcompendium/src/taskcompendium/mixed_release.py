# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a local mixed public candidate from accepted agent-visible records."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.models import SHA256_PATTERN
from taskcompendium.public_projection import PublicTask
from taskcompendium.release_common import REPO_ID, sha256_file

NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
REGIONAL_PIN_PATTERN = re.compile(
    r"[^#]+#manifest-sha256=[0-9a-f]{64}#parquet-size=\d+#parquet-etag=([^#]*)#parquet-version-id=([^#]*)\Z"
)
ConfigName = Literal["workplace", "tasktrove_clean"]
SplitName = Literal["train", "validation"]
RecordFormat = Literal["public_task", "accepted_public_record"]
CONFIG_ORDER = {"workplace": 0, "tasktrove_clean": 1}


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


class AcceptedPublicRecord(BaseModel):
    """A reviewed public task with independently pinned source-row proof."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: PublicTask
    source_proof: SourceProof


class SourceRights(BaseModel):
    """Provenance and redistribution evidence for one included cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    license: str
    attribution: str
    source_card_url: str
    source_card_revision: str
    change_notice: str

    @model_validator(mode="after")
    def validate_rights(self) -> "SourceRights":
        if not all(
            (self.license, self.attribution, self.source_card_url, self.source_card_revision, self.change_notice)
        ):
            raise ValueError("Each cohort needs license, attribution, pinned card, and change notice")
        if not self.source_card_url.startswith("https://"):
            raise ValueError("Source card URL must use HTTPS")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_card_revision):
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
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
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
    record_format: RecordFormat
    input_path: Path
    input_sha256: str
    accepted_rows: int = Field(gt=0)
    source_records: int = Field(gt=0)
    parsed_rows: int | None = Field(default=None, gt=0)
    source_dataset: str
    source_revision: str | None = None
    source_subset: str | None = None
    source_category: str | None = None
    source_assets: tuple[SourceAsset, ...] = Field(min_length=1)
    task_spec_schema: str
    importer_revision: str
    projection_builder_revision: str
    rights: SourceRights
    harbor_samples: tuple[HarborSample, ...] = Field(min_length=1)
    provider_pins: tuple[ProviderPin, ...] = ()

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
        if not self.importer_revision or not re.fullmatch(r"[0-9a-f]{40}", self.projection_builder_revision):
            raise ValueError("Importer and projection builder pins are required")
        if self.config == "workplace" and self.record_format != "public_task":
            raise ValueError("Workplace input must use the public task format")
        if self.config == "workplace" and (self.source_revision is None or len(self.source_assets) != 1):
            raise ValueError("Workplace needs its pinned source revision and one split file")
        if self.config == "tasktrove_clean" and self.record_format != "accepted_public_record":
            raise ValueError("TaskTrove input must carry accepted-row source proof")
        if self.config == "tasktrove_clean" and (not self.source_subset or not self.source_category):
            raise ValueError("TaskTrove cohorts need an exact source subset and category")
        if len({pin.name for pin in self.provider_pins}) != len(self.provider_pins):
            raise ValueError("Provider pin names must be unique")
        if len({(asset.path, asset.pin) for asset in self.source_assets}) != len(self.source_assets):
            raise ValueError("Source asset pins must be unique")
        return self


def _workplace_record(task: PublicTask, cohort: CohortInput) -> AcceptedPublicRecord:
    parts = task.source.row.split(":")
    if len(parts) != 3 or parts[0] != cohort.split or not parts[1].isdigit() or not SHA256_PATTERN.fullmatch(parts[2]):
        raise ValueError("Workplace row lacks split-local SHA256 provenance")
    asset = cohort.source_assets[0]
    return AcceptedPublicRecord(
        task=task,
        source_proof=SourceProof(
            source_row=task.source.row,
            source_row_sha256=parts[2],
            input_file=asset.path,
            input_object_pin=asset.pin,
        ),
    )


def _validated_record(
    raw: str,
    cohort: CohortInput,
    source_assets: set[tuple[str, str]],
    provider_pins: dict[str, ProviderPin],
) -> AcceptedPublicRecord:
    if cohort.record_format == "public_task":
        record = _workplace_record(PublicTask.model_validate_json(raw), cohort)
    else:
        record = AcceptedPublicRecord.model_validate_json(raw)
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
    if cohort.config == "tasktrove_clean":
        if not task.tags:
            raise ValueError("TaskTrove rows must retain original ordered source tags")
        if task.source_category != cohort.source_category or not task.source.row.startswith(f"{cohort.source_subset}:"):
            raise ValueError("TaskTrove row does not belong to the licensed source cohort")
        if proof.archive_path is None or proof.archive_sha256 is None:
            raise ValueError("TaskTrove rows need archive path and digest")
        if not task.source.row.endswith(f":{proof.archive_path}") or task.source.revision != proof.input_object_pin:
            raise ValueError("TaskTrove archive proof differs from row provenance")
    elif proof.archive_path is not None:
        raise ValueError("Workplace source rows must not claim an archive")
    return record


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
            output.write(record.model_dump_json() + "\n")
            count += 1
    if count != cohort.accepted_rows:
        raise ValueError(f"Accepted row count mismatch: {cohort.cohort}")
    return count, sha256_file(destination)


def _card(cohorts: tuple[CohortInput, ...], data_files: list[dict[str, object]]) -> str:
    licenses = {cohort.rights.license for cohort in cohorts}
    license_label = next(iter(licenses)) if len(licenses) == 1 else "other"
    lines = ["---", "pretty_name: TaskCompendium Alpha 1 Candidate", f"license: {license_label}", "configs:"]
    for config in ("workplace", "tasktrove_clean"):
        entries = [entry for entry in data_files if entry["config"] == config]
        if not entries:
            continue
        lines.extend((f"  - config_name: {config}", "    data_files:"))
        for entry in entries:
            lines.extend((f"      - split: {entry['split']}", f"        path: {entry['path']}"))
    lines.extend(("---", "", "# TaskCompendium Alpha 1 Candidate", ""))
    lines.append(
        "This local candidate contains accepted, agent-visible tasks. The manifest records exported counts, "
        "source pins, rights, and Harbor sample evidence. Verifier settings, reference answers, gold actions, "
        "and private resources are excluded. The builder does not upload to the Hub."
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
            f"{cohort.rights.license}; [source card]({cohort.rights.source_card_url}) "
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


def assemble_mixed_candidate(cohorts: tuple[CohortInput, ...], destination: Path, *, builder_revision: str) -> Path:
    """Validate and atomically write accepted cohorts into separate Hub configs."""
    if not re.fullmatch(r"[0-9a-f]{40}", builder_revision):
        raise ValueError("Builder revision must be a full Git commit")
    if not cohorts or destination.exists():
        raise ValueError("Nonempty cohorts and a new destination are required")
    ordered = tuple(sorted(cohorts, key=lambda cohort: (CONFIG_ORDER[cohort.config], cohort.split, cohort.cohort)))
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
                        "dataset": cohort.source_dataset,
                        "revision": cohort.source_revision,
                        "subset": cohort.source_subset,
                        "category": cohort.source_category,
                        "assets": [asset.model_dump(mode="json") for asset in cohort.source_assets],
                        "task_spec_schema": cohort.task_spec_schema,
                        "importer_revision": cohort.importer_revision,
                        "projection_builder_revision": cohort.projection_builder_revision,
                    },
                    "rights": cohort.rights.model_dump(mode="json"),
                    "harbor_samples": [sample.model_dump(mode="json") for sample in cohort.harbor_samples],
                    "provider_pins": [pin.model_dump(mode="json") for pin in cohort.provider_pins],
                }
            )
        manifest = {
            "format_version": 1,
            "public_record_version": 1,
            "repo_id": REPO_ID,
            "visibility": "agent",
            "builder_revision": builder_revision,
            "publication_ready": False,
            "data_files": data_files,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "README.md").write_text(_card(ordered, data_files))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a local agent-visible mixed alpha candidate")
    parser.add_argument("--cohorts", type=Path, required=True, help="JSON array of accepted cohort inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder-revision", required=True)
    arguments = parser.parse_args()
    cohorts = tuple(CohortInput.model_validate(item) for item in json.loads(arguments.cohorts.read_text()))
    assemble_mixed_candidate(cohorts, arguments.output, builder_revision=arguments.builder_revision)


if __name__ == "__main__":
    main()
