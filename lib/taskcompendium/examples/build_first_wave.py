# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the bounded NeMo environment-coverage examples from pinned local fixtures."""

import argparse
import hashlib
import json
from pathlib import Path

from coverage_labels import label

from taskcompendium.execution import HarborTaskBinding, NoEnvironment
from taskcompendium.importers import nemo, nemo_predicted_action, nemo_workplace, nemo_workplace_multistep
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    SCHEMA_VERSION,
    AssistantFinal,
    JsonPath,
    PlainText,
    Rejected,
    Rendering,
    TaskSpec,
    XmlPath,
)
from taskcompendium.serialization import read_parquet, specification_hash, to_json, write_parquet

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/nemo"


def accepted(result: TaskSpec | Rejected) -> TaskSpec:
    if isinstance(result, Rejected):
        raise ValueError(f"Pinned first-wave fixture was rejected: {result}")
    return result


def build(output: Path, runtime_image: str) -> None:
    """Export six semantic instances and their supported rendering variants."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "specifications").mkdir()
    answer_binding = HarborTaskBinding(NoEnvironment())
    variants: list[tuple[TaskSpec, tuple[Rendering, ...], HarborTaskBinding]] = []
    for aggregation in ("binary", "fraction"):
        spec = accepted(
            nemo.load_instruction_sample(FIXTURES / "instruction-following-17616.json", aggregation=aggregation)
        )
        for name, extractor in (("plain", PlainText()), ("json", JsonPath()), ("xml", XmlPath())):
            variants.append((spec, (Rendering(name, AssistantFinal(extractor)),), answer_binding))
    code = accepted(
        nemo.load_code_sample(
            FIXTURES / "code-answer-c69268d8bdb4da0685d7b187c88296c1.json", verifier_image=runtime_image
        )
    )
    for name, extractor in (("raw-code", PlainText()), ("json", JsonPath()), ("xml", XmlPath())):
        variants.append((code, (Rendering(name, AssistantFinal(extractor)),), answer_binding))
    provenance = json.loads((FIXTURES / "predicted-action.provenance.json").read_text())
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    digest = provenance["canonical_json_sha256"]
    prediction = accepted(nemo_predicted_action.import_row(row, digest))
    variants.append((prediction, (nemo_predicted_action.rendering(row, digest),), answer_binding))
    workplace = nemo_workplace.build_sample(FIXTURES)
    variants.append((workplace.specification, (workplace.rendering,), workplace.binding))
    sequence = nemo_workplace_multistep.build_multistep_sample(FIXTURES)
    variants.append((sequence.specification, sequence.renderings, sequence.binding))

    variants = [(label(spec), renderings, binding) for spec, renderings, binding in variants]
    specifications = {spec.id: spec for spec, _, _ in variants}
    for spec in specifications.values():
        (output / "specifications" / f"{spec.id.replace('/', '-')}.json").write_bytes(to_json(spec))
    exports = []
    for spec, renderings, binding in variants:
        rendering = renderings[0]
        destination = output / "harbor" / f"{spec.id.replace('/', '-')}-{rendering.id}"
        lower_to_harbor(spec, renderings, binding, destination)
        exports.append(
            {
                "task": spec.id,
                "rendering": rendering.id,
                "path": str(destination.relative_to(output)),
                "specification_sha256": specification_hash(spec),
            }
        )
    parquet = output / "specifications.parquet"
    write_parquet(list(specifications.values()), str(parquet))
    if [specification_hash(spec) for spec in read_parquet(str(parquet))] != [
        specification_hash(spec) for spec in specifications.values()
    ]:
        raise RuntimeError("First-wave specifications did not round-trip through Parquet")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "specifications": len(specifications),
        "exports": exports,
        "runtime_image": runtime_image,
        "inputs": [
            {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(FIXTURES.glob("*.json"))
            if "attempt" not in path.name
        ],
        "validation": "Import, serialization, and export only. Execution evidence is recorded separately.",
        "execution": "Configure a model endpoint for chat/provider_chat, or use the private scripted test attempts.",
        "scope": (
            "Four pinned source cases plus a derived Workplace sequence. "
            "Instruction scoring has binary/fractional variants."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "specifications": len(specifications), "exports": len(exports)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True, help="Immutable compatible private code-verifier image")
    args = parser.parse_args()
    build(args.output, args.runtime_image)
