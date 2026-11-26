# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin campaign candidates to Atlas releases and partition their regional pools."""

import csv
import hashlib
import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from iris.cluster.client.job_info import get_job_info
from marin.rl.curriculum import (
    CurriculumEntry,
    PoolRow,
    RlCurriculum,
    SamplingMode,
    SourcePin,
    draw_validation_split,
    validation_payload,
)
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.long_loop.exp3_data import PreparedPools
from experiments.post_training.long_loop.rl_terminal_evaluation import native_command

CAMPAIGN = Path(__file__).with_name("campaign")
SEED = 20261020
VALIDATION_SIZE = 200
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CampaignMix:
    id: str
    sources: dict[str, dict]
    candidates_sha256: str
    catalog_sha256: str

    @property
    def weights(self) -> dict[str, float]:
        return {source: 1 / len(self.sources) for source in self.sources}


def load_mix(mix_id: str, directory: Path = CAMPAIGN) -> CampaignMix:
    """Resolve a version-prefixed CSV candidate, refusing unsealed or changed releases."""
    generation, separator, identifier = mix_id.partition("-")
    if generation not in {"v2", "v3"} or not separator:
        raise ValueError("mix-id must start with v2- or v3-")
    pins_path = directory / "source_pins.json"
    catalog = json.loads(pins_path.read_text())
    candidates = directory / f"mix_candidates_{generation}.csv"
    content = candidates.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != catalog["candidate_sha256"][generation]:
        raise ValueError("Candidate CSV changed; publish a fresh catalog binding first")
    with candidates.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["mix_id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"Expected one candidate for {mix_id}, found {len(matches)}")
    row = matches[0]
    names = row["sources"].split(";")
    if len(names) != len(set(names)) or len(names) != int(row["k"]):
        raise ValueError("Candidate sources repeat or disagree with k")
    sources = {name: catalog["sources"][name] for name in sorted(names)}
    pending = [name for name, pin in sources.items() if pin["admission"] != "catalog"]
    if pending:
        raise ValueError(f"Awaiting sealed retained publications, not raw converters: {pending}")
    return CampaignMix(mix_id, sources, digest, hashlib.sha256(pins_path.read_bytes()).hexdigest())


def partition_mix(mix: CampaignMix, rows: dict[str, list[dict]], version: str) -> PreparedPools:
    """Preserve source rows while removing a reproducible, mixture-weighted holdout."""
    identities = {
        label: tuple(
            PoolRow(json.loads(row["task_spec"])["id"], canonical_sha256(json.loads(row["task_spec"])))
            for row in records
        )
        for label, records in rows.items()
    }
    split = draw_validation_split(mix.weights, identities, size=VALIDATION_SIZE, seed=SEED)
    held = {(row.label, row.id) for row in split.validation.rows}
    # Cross-source copies of a held-out task must not leak back into training.
    held_ids = {row.id for row in split.validation.rows}
    held_hashes = {row.sha256 for row in split.validation.rows}
    training, validation = {}, []
    for label, records in rows.items():
        training[label] = []
        for row, identity in zip(records, identities[label], strict=True):
            stamped = {**row, "teacher_route": label}
            if (label, identity.id) in held:
                validation.append(stamped)
            elif identity.id in held_ids or identity.sha256 in held_hashes:
                raise ValueError(f"Cross-source holdout leakage: {label}/{identity.id}")
            else:
                training[label].append(stamped)
    entries = tuple(
        CurriculumEntry(
            label=label,
            source=pin["name"],
            version=pin["version"],
            pin=SourcePin(
                pin["repo"], pin["revision"], f"https://huggingface.co/datasets/{pin['repo']}/tree/{pin['revision']}"
            ),
            recipe="experiments.post_training.task_curation.retained_release.import_retained",
            weight=mix.weights[label],
        )
        for label, pin in mix.sources.items()
    )
    curriculum = RlCurriculum(
        name=f"rl-data-campaign/{mix.id}",
        version=version,
        mode=SamplingMode.WEIGHTED,
        seed=SEED,
        entries=entries,
        stages=(),
        validation=split.validation,
    )
    return PreparedPools(curriculum, training, validation)


def write_parquet(destination: StoragePath, name: str, rows: list[dict]) -> str:
    """Publish complete TaskSpec/runtime rows without dropping retained columns."""
    if not rows or any(row.keys() != rows[0].keys() for row in rows):
        raise ValueError(f"Empty or heterogeneous pool schema: {name}")
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "tasks.parquet"
        pq.write_table(pa.Table.from_pylist(rows), local, compression="zstd")
        expected = hashlib.sha256(local.read_bytes()).hexdigest()
        target = destination / name
        target.parent.mkdirs(exist_ok=True)
        with local.open("rb") as source, target.open("wb") as output:
            while chunk := source.read(8 * 1024 * 1024):
                output.write(chunk)
        with target.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"Pool readback changed: {target}")
    return expected


def prepare_pools(*, output: str, mix: CampaignMix, pool_version: str, launch_document: str) -> None:
    """Read the pinned retained cohorts only inside the destination region's CPU job."""
    if get_job_info() is None or not output.startswith("s3://marin-us-east-08a/"):
        raise ValueError("Campaign pools must be materialized inside an 08a Iris CPU job")
    rows = {}
    for label, pin in mix.sources.items():
        logger.info("Reading campaign cohort %s at %s@%s", label, pin["repo"], pin["revision"])
        local = Path(
            hf_hub_download(
                pin["repo"],
                "tasks.parquet",
                revision=pin["revision"],
                repo_type="dataset",
                token=require_secret("HF_TOKEN"),
            )
        )
        with local.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != pin["sha256"]:
                raise ValueError(f"Atlas release changed: {label}")
        records = pq.read_table(local).to_pylist()
        if len(records) != pin["count"]:
            raise ValueError(f"Atlas population changed: {label}")
        logger.info("Verified campaign cohort %s: %d rows and exact parquet SHA", label, len(records))
        for row in records:
            task = TaskSpec.model_validate_json(row["task_spec"])
            if canonical_sha256(json.loads(row["task_spec"])) != row["task_sha256"]:
                raise ValueError(f"Task checksum changed: {label}/{task.id}")
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "launch.yaml"
            settings.write_text(launch_document)
            lowered = Path(temporary) / "lowered.parquet"
            native_command(
                [
                    "python",
                    "-m",
                    "experiments.post_training.long_loop.campaign_runtime",
                    str(local),
                    str(lowered),
                    str(settings),
                    pin["route"],
                ]
            )
            rows[label] = pq.read_table(lowered).to_pylist()
        logger.info("Lowered campaign cohort %s through the native runtime", label)
    prepared = partition_mix(mix, rows, pool_version)
    curriculum, training, validation = prepared.curriculum, prepared.training, prepared.validation
    destination = StoragePath(output)
    files = {
        f"source-{index}/train.parquet": write_parquet(destination, f"source-{index}/train.parquet", records)
        for index, records in enumerate(training.values())
    }
    training_files = {label: name for label, name in zip(training, files, strict=True)}
    files["validation.parquet"] = write_parquet(destination, "validation.parquet", validation)
    definition = asdict(curriculum)
    definition.pop("fingerprint")
    payload = json.dumps(definition, indent=2).encode()
    (destination / "curriculum.json").write_bytes(payload)
    files["curriculum.json"] = hashlib.sha256(payload).hexdigest()
    document = {
        "curriculum": curriculum.identity,
        "curriculum_definition": definition,
        "validation": validation_payload(curriculum.validation),
        "composition": {label: len(records) for label, records in rows.items()},
        "train_composition": {label: len(records) for label, records in training.items()},
        "training_files": training_files,
        "files": files,
        "routes": {label: pin["route"] for label, pin in mix.sources.items()},
        "sources": mix.sources,
        "mix": asdict(mix),
    }
    (destination / "manifest.json").write_text(json.dumps(document, indent=2))
    logger.info(
        "Campaign pools published: %s, holdout=%d, training=%d",
        output,
        len(validation),
        sum(len(records) for records in training.values()),
    )
