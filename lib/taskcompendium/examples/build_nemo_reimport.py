# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the NeMo environment rows whose source contracts are reproducible."""

import argparse
import hashlib
import json
from pathlib import Path

from coverage_labels import label

from taskcompendium.importers.nemo_workplace import import_hub_row, provider_binding
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import AssistantFinal, Rejected, Rendering
from taskcompendium.serialization import read_parquet, specification_hash, to_json, write_parquet

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/nemo/reimport"
WORKPLACE_OFFSETS = (536, 1163)


def build(output: Path) -> None:
    """Export the reproducible Workplace Hub rows and their Harbor bindings."""
    output.mkdir(parents=True, exist_ok=False)
    specifications = []
    inputs = []
    for offset in WORKPLACE_OFFSETS:
        path = FIXTURES / f"workplace-{offset}.json"
        provenance = json.loads(path.with_suffix(".provenance.json").read_text())
        data = path.read_bytes()
        specification = import_hub_row(data, split=provenance["split"], offset=provenance["offset"])
        if isinstance(specification, Rejected):
            raise ValueError(f"Pinned Workplace row {offset} was rejected: {specification.detail}")
        specifications.append(label(specification))
        inputs.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "source": provenance,
            }
        )
    (output / "specifications").mkdir()
    exports = []
    binding = provider_binding()
    rendering = Rendering("provider-chat", AssistantFinal())
    for specification in specifications:
        task_id = specification.id.replace("/", "-")
        (output / "specifications" / f"{task_id}.json").write_bytes(to_json(specification))
        destination = output / "harbor" / f"{task_id}-{rendering.id}"
        lower_to_harbor(specification, (rendering,), binding, destination)
        exports.append(
            {"task": specification.id, "rendering": rendering.id, "path": str(destination.relative_to(output))}
        )
    parquet = output / "specifications.parquet"
    write_parquet(specifications, str(parquet))
    if [specification_hash(item) for item in read_parquet(str(parquet))] != [
        specification_hash(item) for item in specifications
    ]:
        raise RuntimeError("Re-imported Workplace specifications did not round-trip through Parquet")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "specifications": len(specifications),
                "exports": exports,
                "inputs": inputs,
                "shared_environment": "nemo_workplace_v1 seeded provider",
                "scope": "Only source rows with a complete stateful provider and authoritative final-state contract.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
