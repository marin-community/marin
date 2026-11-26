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
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from iris.cluster.client.job_info import get_job_info
from marin.rl.curriculum import (
    PoolRow,
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
from experiments.post_training.task_curation.curriculum import (
    ServedCatalogSource,
    curriculum_entries,
    served_catalog_sources,
    weighted_curriculum,
)

CAMPAIGN = Path(__file__).with_name("campaign")
SEED = 20261020
VALIDATION_SIZE = 200
CATALOG_LOAD_ID = "54405be9b4db4c2d9828549469abe759"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CampaignMix:
    id: str
    sources: dict[str, dict]
    candidates_sha256: str
    catalog_sha256: str

    @property
    def catalog_divergence(self) -> dict[str, dict]:
        return {
            label: {"status": pin["catalog_status"], "repo": pin["repo"], "revision": pin["revision"]}
            for label, pin in self.sources.items()
            if pin["catalog_status"] != "Available"
        }

    @property
    def weights(self) -> dict[str, float]:
        return {source: 1 / len(self.sources) for source in self.sources}

    @property
    def curriculum_catalog(self) -> dict[str, ServedCatalogSource]:
        return {
            label: ServedCatalogSource(
                pin["name"], pin["version"], SourcePin(pin["repo"], pin["revision"], pin["url"]), pin["recipe"]
            )
            for label, pin in self.sources.items()
        }


def load_mix(mix_id: str, directory: Path = CAMPAIGN) -> CampaignMix:
    """Resolve the exact candidate pins under the campaign's recorded admission policy."""
    generation, separator, identifier = mix_id.partition("-")
    if generation not in {"v2", "v3"} or not separator:
        raise ValueError("mix-id must start with v2- or v3-")
    pins_path = directory / "source_pins.json"
    catalog = json.loads(pins_path.read_text())
    if catalog["catalog_load"]["load_id"] != CATALOG_LOAD_ID:
        raise ValueError("Campaign requires the explicitly selected R40 served catalog load")
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
    available = served_catalog_sources(
        {**catalog, "sources": {name: catalog["sources"][name] for name in names if name in catalog["sources"]}},
        admitted_statuses=tuple(catalog.get("admission_policy", {}).get("admitted_statuses", ["Available"])),
    )
    pending = {
        name: catalog["sources"].get(name, {}).get("status", "Missing") for name in names if name not in available
    }
    if pending:
        raise ValueError(f"Served R40 catalog cannot admit this mix: {pending}")
    entries = curriculum_entries(
        mix_id, CATALOG_LOAD_ID, weights={name: 1 / len(names) for name in names}, catalog=available
    )
    sources = {}
    for entry in sorted(entries, key=lambda entry: entry.label):
        source = catalog["sources"][entry.label]
        if source["pipeline"]["files"] != ["tasks.parquet"]:
            raise ValueError(f"Catalog source lacks the published TaskSpec cohort: {entry.label}")
        if not source["actor_census_input"] or source["route"] not in {"agentic", "nonagentic"}:
            raise ValueError(f"Catalog source lacks its measured population or actor route: {entry.label}")
        sources[entry.label] = {
            "catalog_status": source["status"],
            "name": entry.source,
            "version": entry.version,
            "repo": entry.pin.name,
            "revision": entry.pin.revision,
            "url": entry.pin.url,
            "recipe": entry.recipe,
            "count": source["actor_census_input"]["measured_tasks"],
            "route": source["route"],
            "catalog_load_id": CATALOG_LOAD_ID,
        }
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
    curriculum = weighted_curriculum(
        f"rl-data-campaign/{mix.id}",
        version,
        weights=mix.weights,
        seed=SEED,
        validation_size=VALIDATION_SIZE,
        pool_rows=identities,
        catalog=mix.curriculum_catalog,
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


def retained_task(row: dict, label: str) -> TaskSpec:
    """Verify the catalog's hash of the literal retained TaskSpec JSON bytes."""
    encoded = row["task_spec"]
    task = TaskSpec.model_validate_json(encoded)
    if hashlib.sha256(encoded.encode()).hexdigest() != row["task_sha256"]:
        raise ValueError(f"Task checksum changed: {label}/{task.id}")
    return task


def prepare_pools(*, output: str, mix: CampaignMix, pool_version: str, launch_document: str) -> None:
    """Read the pinned retained cohorts only inside the destination region's CPU job."""
    if get_job_info() is None or not output.startswith("s3://marin-us-east-08a/"):
        raise ValueError("Campaign pools must be materialized inside an 08a Iris CPU job")
    rows = {}
    source_readbacks = {}
    for label, pin in mix.sources.items():
        logger.info("Reading campaign cohort %s at %s@%s", label, pin["repo"], pin["revision"])
        token = require_secret("HF_TOKEN")
        metadata = get_hf_file_metadata(
            hf_hub_url(pin["repo"], "tasks.parquet", revision=pin["revision"], repo_type="dataset"), token=token
        )
        if metadata.etag is None or len(metadata.etag) != 64:
            raise ValueError(f"Published cohort lacks an immutable SHA256 blob: {label}")
        local = Path(
            hf_hub_download(
                pin["repo"],
                "tasks.parquet",
                revision=pin["revision"],
                repo_type="dataset",
                token=token,
            )
        )
        with local.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != metadata.etag:
                raise ValueError(f"Atlas release changed: {label}")
        records = pq.read_table(local).to_pylist()
        if len(records) != pin["count"]:
            raise ValueError(f"Atlas population changed: {label}")
        logger.info("Verified campaign cohort %s: %d rows and exact parquet SHA", label, len(records))
        source_readbacks[label] = {"parquet_sha256": metadata.etag, "rows": len(records), "catalog_pin": pin}
        for row in records:
            retained_task(row, label)
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
    files, training_files = {}, {}
    for index, (label, records) in enumerate(training.items()):
        name = f"source-{index}/train.parquet"
        files[name] = write_parquet(destination, name, records)
        training_files[label] = name
    files["validation.parquet"] = write_parquet(destination, "validation.parquet", validation)
    definition = asdict(curriculum)
    definition.pop("fingerprint")
    payload = json.dumps(definition, indent=2).encode()
    (destination / "curriculum.json").write_bytes(payload)
    files["curriculum.json"] = hashlib.sha256(payload).hexdigest()
    document = {
        "catalog_divergence": mix.catalog_divergence,
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
        "source_readbacks": source_readbacks,
    }
    (destination / "manifest.json").write_text(json.dumps(document, indent=2))
    logger.info(
        "Campaign pools published: %s, holdout=%d, training=%d",
        output,
        len(validation),
        sum(len(records) for records in training.values()),
    )
