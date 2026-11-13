# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse verified retained SAMPLE QC panels for the four-model difficulty campaign."""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from finestore.rl.curation import ContentEvidenceReference, SourceDeclaration
from finestore.rl.difficulty import DifficultySample, SampleTask
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.sample_qc import panel_tasks
from experiments.post_training.task_curation.sample_qc_artifacts import SampleInventory, SamplePanel

LOGGER = logging.getLogger(__name__)


class DifficultySource(BaseModel):
    """A frozen panel and sample receipt; runtime qualification remains separate."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    declaration: SourceDeclaration
    panel: SamplePanel
    sample: DifficultySample
    sample_receipt: ContentEvidenceReference
    backend: str


def prepare_sample(
    declaration: SourceDeclaration, inventory: SampleInventory, output_root: str, resume_root: str | None
) -> DifficultySource:
    """Check an existing seeded selection and preserve its pre-oracle-filter status."""
    binding = declaration.actor_census_input
    assert declaration.pipeline is not None
    if binding is None or (
        declaration.id,
        declaration.dataset_id,
        declaration.dataset_revision,
        declaration.pipeline.version,
        binding.measured_tasks,
        binding.task_specs_sha256,
    ) != (
        inventory.source_id,
        inventory.dataset_id,
        inventory.dataset_revision,
        inventory.converter_version,
        inventory.retained_tasks,
        inventory.task_specs_sha256,
    ):
        raise ValueError(f"Panel inventory differs from the current pinned population: {declaration.id}")
    panel = inventory.qc_panel()
    if resume_root is not None:
        previous = StoragePath(resume_root) / canonical_sha256(declaration.id)[:20] / "source.json"
        if previous.exists():
            source = DifficultySource.model_validate_json(previous.read_text())
            if (
                source.declaration.current_release,
                source.sample.converter_version,
                source.sample.converter_revision,
                source.sample.population_count,
                source.sample.task_specs_sha256,
                source.panel.panel_sha256,
                source.sample.pre_oracle_filter,
            ) != (
                declaration.current_release,
                declaration.pipeline.version,
                binding.converter_revision,
                inventory.retained_tasks,
                inventory.task_specs_sha256,
                panel.panel_sha256,
                inventory.oracle_tasks > 0,
            ):
                raise ValueError(f"Earlier difficulty sample belongs to another population: {declaration.id}")
            if (
                file_sha256(StoragePath(source.sample_receipt.uri)) != source.sample_receipt.sha256
                or file_sha256(StoragePath(source.panel.panel_path)) != panel.panel_sha256
                or DifficultySample.model_validate_json(StoragePath(source.sample_receipt.uri).read_text())
                != source.sample
            ):
                raise ValueError(f"Earlier difficulty evidence changed: {declaration.id}")
            LOGGER.info("Reused verified difficulty sample: %s", declaration.id)
            return source.model_copy(
                update={
                    "declaration": declaration,
                    "backend": "daytona" if declaration.daytona_safety == "safe" else "local",
                }
            )
    tasks, contracts = panel_tasks(panel)
    root = StoragePath(output_root) / canonical_sha256(declaration.id)[:20]
    receipt = root / "sample.json"
    if root.exists():
        raise FileExistsError(f"Select a new difficulty sample version: {root}")
    drop_manifest = StoragePath(declaration.within_source_dedup.drop_manifest_uri)
    sample = DifficultySample(
        source_id=declaration.id,
        release=declaration.current_release,
        converter_version=declaration.pipeline.version,
        converter_revision=binding.converter_revision,
        population_count=inventory.retained_tasks,
        task_specs_sha256=inventory.task_specs_sha256,
        drop_manifest=ContentEvidenceReference(uri=str(drop_manifest), sha256=file_sha256(drop_manifest)),
        seed=panel.sample_seed,
        tasks=[SampleTask(task_id=contract["task_id"], task_sha256=contract["task_sha256"]) for contract in contracts],
        pre_oracle_filter=inventory.oracle_tasks > 0,
        resample_after_oracle_filter=inventory.oracle_tasks > 0,
    )
    payload = (sample.model_dump_json(indent=2) + "\n").encode()
    # Preserve the original seed-zero panel bytes; no resampling or task conversion occurs here.
    original = StoragePath(panel.panel_path).read_bytes()
    if hashlib.sha256(original).hexdigest() != panel.panel_sha256:
        raise ValueError(f"Panel changed after task validation: {declaration.id}")
    copied_panel = root / "panel.parquet"
    copied_panel.write_bytes(original)
    if file_sha256(copied_panel) != panel.panel_sha256:
        raise ValueError(f"Staged panel digest differs: {declaration.id}")
    receipt.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if file_sha256(receipt) != digest:
        raise ValueError(f"Sample receipt changed on readback: {declaration.id}")
    result = DifficultySource(
        declaration=declaration,
        panel=panel.model_copy(update={"panel_path": str(copied_panel)}),
        sample=sample,
        sample_receipt=ContentEvidenceReference(uri=str(receipt), sha256=digest),
        backend="daytona" if declaration.daytona_safety == "safe" else "local",
    )
    (root / "source.json").write_text(result.model_dump_json(indent=2) + "\n")
    LOGGER.info(
        "%s: %d tasks, backend=%s, pre_oracle_filter=%s",
        declaration.id,
        len(tasks),
        result.backend,
        sample.pre_oracle_filter,
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--resume-root")
    args = parser.parse_args()
    configure_coreweave_s3()
    output = StoragePath(args.output_root)
    if output.exists():
        raise FileExistsError(f"Preserve earlier samples; choose a new output version: {output}")
    inventory_path = StoragePath(args.inventory)
    if file_sha256(inventory_path) != args.inventory_sha256:
        raise ValueError("Inventory differs from its frozen digest")
    inventory = {
        row.source_id: row
        for row in (SampleInventory.model_validate(item) for item in json.loads(inventory_path.read_text()))
    }
    declarations = [
        SourceDeclaration.model_validate(row) for row in json.loads(StoragePath(args.declarations).read_text())
    ]
    if len({row.id for row in declarations}) != len(declarations):
        raise ValueError("Difficulty sources must be unique")
    prepare = partial(prepare_sample, output_root=args.output_root, resume_root=args.resume_root)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, declaration, inventory[declaration.id]) for declaration in declarations]
        sources = sorted((future.result() for future in futures), key=lambda item: item.declaration.id)
    queue = output / "sources.json"
    queue.write_text(json.dumps([source.model_dump(mode="json") for source in sources], indent=2) + "\n")
    (output / "resampling-queue.json").write_text(
        json.dumps(
            [
                {
                    "source_id": source.declaration.id,
                    "status": "waiting_for_oracle_filter",
                    "previous_sample_receipt": source.sample_receipt.model_dump(mode="json"),
                    "previous_population_sha256": source.sample.task_specs_sha256,
                    "seed": source.sample.seed,
                    "action": (
                        "Select min(100, retained_count) from the filtered population with the same seed; "
                        "rerun all four models."
                    ),
                }
                for source in sources
                if source.sample.pre_oracle_filter
            ],
            indent=2,
        )
        + "\n"
    )
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "sources": len(sources),
                "sampled_tasks": sum(len(source.sample.tasks) for source in sources),
                "pre_oracle_filter_sources": sum(source.sample.pre_oracle_filter for source in sources),
                "inventory": args.inventory,
                "inventory_sha256": args.inventory_sha256,
                "queue": str(queue),
                "queue_sha256": file_sha256(queue),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
