# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the Atlas preview with its exact catalog models inside the application package."""

import argparse
import hashlib
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

from finestore.rl.curation import ActorCensusInput
from rigging.provenance import git_output

from experiments.post_training.task_curation.sample_qc_artifacts import SampleInventory
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

APP_SOURCE = Path("infra/marina/apps/rl_data_catalog")
PRODUCTION_APP_ID = "fb11c931-5861-4878-8bb5-a964d652b45f"
PREVIEW_APP_ID = "c9742a15-fb46-4d56-bc88-aa916351ebd3"
MODEL_SOURCES = {
    "curation.py": "lib/finestore/src/finestore/rl/curation.py",
    "difficulty.py": "lib/finestore/src/finestore/rl/difficulty.py",
    "source_fingerprints.py": "lib/taskcompendium/src/taskcompendium/source_fingerprints.py",
}
IMPORT_RELOCATIONS = {
    "from finestore.rl.curation import ": "from .bundled.curation import ",
    "from finestore.rl.difficulty import ": "from .bundled.difficulty import ",
    "from taskcompendium.source_fingerprints import ": "from .bundled.source_fingerprints import ",
}


def oracle_inventory_payload(raw: bytes) -> dict[str, Any]:
    """Translate a full-population SAMPLE inventory ledger into pinned browser observations."""
    ledger = json.loads(raw)
    if (ledger["schema_version"], ledger["kind"]) != (2, "full_population_oracle_inventory"):
        raise ValueError("Expected a version-2 full-population oracle inventory ledger")
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in ledger["rows"]:
        binding = ActorCensusInput.model_validate(row["current_input"])
        inventory = SampleInventory.model_validate(row["inventory"])
        identifier = canonical_source_id(row["source_id"])
        actual = (
            inventory.dataset_id,
            inventory.dataset_revision,
            inventory.converter_version,
            inventory.task_specs_sha256,
            inventory.retained_tasks,
        )
        expected = (
            binding.dataset_id,
            binding.dataset_revision,
            binding.converter_version,
            binding.task_specs_sha256,
            binding.measured_tasks,
        )
        if actual != expected or canonical_source_id(inventory.source_id) != identifier:
            raise ValueError(f"Oracle inventory differs from its actor input: {identifier}")
        if identifier in identifiers or inventory.oracle_tasks > inventory.retained_tasks:
            raise ValueError(f"Oracle inventory has duplicate sources or invalid oracle counts: {identifier}")
        identifiers.add(identifier)
        rows.append(
            {
                "source_id": identifier,
                "input": binding.model_dump(mode="json"),
                "oracle_tasks": inventory.oracle_tasks,
                "execution_qualification": inventory.execution_qualification,
                "evidence_url": inventory.evidence_url,
                "inventory_manifest_uri": row["inventory_manifest_uri"],
                "inventory_manifest_sha256": row["inventory_manifest_sha256"],
                "scanned_task_specs_sha256": inventory.scanned_task_specs_sha256,
                "panel_path": inventory.panel_path,
                "panel_sha256": inventory.panel_sha256,
            }
        )
    if len(rows) != ledger["sources"] or sum(row["input"]["measured_tasks"] for row in rows) != ledger["retained_tasks"]:
        raise ValueError("Oracle inventory totals do not cover its declared population")
    return {"sources": rows, "source_ledger_sha256": hashlib.sha256(raw).hexdigest()}


def prepare_preview(checkout: Path, output: Path, oracle_inventory: Path | None = None) -> None:
    """Write a new preview-only package and record hashes of its canonical model inputs."""
    observations = oracle_inventory_payload(oracle_inventory.read_bytes()) if oracle_inventory is not None else None
    source = checkout / APP_SOURCE
    if output.resolve().is_relative_to(checkout.resolve()):
        raise ValueError("Stage the preview outside the source checkout")
    manifest_text = (source / "applet.toml").read_text()
    if tomllib.loads(manifest_text)["applet_id"] != PRODUCTION_APP_ID:
        raise ValueError("The source manifest does not identify the maintained Atlas")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (output / "applet.toml").write_text(manifest_text.replace(PRODUCTION_APP_ID, PREVIEW_APP_ID))
    bundled = output / "server/bundled"
    bundled.mkdir()
    (bundled / "__init__.py").write_text("# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n")
    modules = []
    for name, relative in MODEL_SOURCES.items():
        raw = (checkout / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        packaged = raw.replace(b"from finestore.rl.curation import ", b"from .curation import ")
        (bundled / name).write_bytes(packaged)
        modules.append(
            {
                "source": relative,
                "source_sha256": digest,
                "packaged": f"server/bundled/{name}",
                "packaged_sha256": hashlib.sha256(packaged).hexdigest(),
                "transformation": "Unchanged" if packaged == raw else "Relocated curation model import",
            }
        )
    for relative in (
        "server/app.py",
        "server/curriculum.py",
        "server/quality.py",
        "server/contamination.py",
        "server/difficulty.py",
    ):
        path = output / relative
        content = path.read_text()
        for before, after in IMPORT_RELOCATIONS.items():
            content = content.replace(before, after)
        path.write_text(content)
    metadata: dict[str, Any] = {
        "source_revision": git_output(["rev-parse", "HEAD"], checkout),
        "modules": modules,
        "preview_only": True,
    }
    if observations is not None:
        payload = (json.dumps(observations, indent=2) + "\n").encode()
        (output / "dist/measurements/oracle-inventory.json").write_bytes(payload)
        metadata["oracle_inventory"] = {
            "source_sha256": observations["source_ledger_sha256"],
            "packaged_sha256": hashlib.sha256(payload).hexdigest(),
            "sources": len(observations["sources"]),
        }
    (output / "dist/bundled-model-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-inventory", type=Path, help="Verified full-population SAMPLE inventory ledger")
    args = parser.parse_args()
    prepare_preview(args.checkout, args.output, args.oracle_inventory)


if __name__ == "__main__":
    main()
