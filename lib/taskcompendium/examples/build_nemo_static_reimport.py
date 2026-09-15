# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the reviewed answer-only NeMo rows with shared static verifiers."""

import argparse
import hashlib
import json
from pathlib import Path

from coverage_labels import label

from taskcompendium.execution import HarborTaskBinding, NoEnvironment
from taskcompendium.importers.nemo_static import StaticCorpus, import_hub_row
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AssistantFinal,
    BoxedLatex,
    JsonPath,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    Rejected,
    Rendering,
    TaskSpec,
    XmlPath,
)
from taskcompendium.serialization import read_parquet, specification_hash, to_json, write_parquet

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/nemo/static"
ROWS = (
    ("mcqa-132738.json", StaticCorpus.MCQA, "train", 132738),
    ("mcqa-441248.json", StaticCorpus.MCQA, "train", 441248),
    ("open-math-0.json", StaticCorpus.OPEN_MATH, "train", 0),
    ("open-math-1.json", StaticCorpus.OPEN_MATH, "train", 1),
    ("stack-math-0.json", StaticCorpus.STACK_MATH, "train", 0),
    ("stack-math-1.json", StaticCorpus.STACK_MATH, "train", 1),
    ("open-qa-25704.json", StaticCorpus.OPEN_QA, "train", 25704),
    ("open-qa-86898.json", StaticCorpus.OPEN_QA, "train", 86898),
    ("science-144924.json", StaticCorpus.SCIENCE, "so_openq", 144924),
    ("reasoning-gym-0.json", StaticCorpus.REASONING_GYM, "train", 0),
    ("reasoning-gym-1.json", StaticCorpus.REASONING_GYM, "train", 1),
)
RENDERINGS = (
    Rendering("plain", AssistantFinal()),
    Rendering("boxed", AssistantFinal(BoxedLatex())),
    Rendering("json", AssistantFinal(JsonPath())),
    Rendering("xml", AssistantFinal(XmlPath())),
)


def _accepted(result: TaskSpec | Rejected) -> TaskSpec:
    if isinstance(result, Rejected):
        raise ValueError(f"Pinned static row was rejected: {result.detail}")
    return result


def build(output: Path, judge: JudgeConfig) -> None:
    """Export reviewed rows and formatting lowerings to native Harbor packages."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "specifications").mkdir()
    binding = HarborTaskBinding(NoEnvironment())
    specifications = []
    inputs = []
    for name, corpus, split, offset in ROWS:
        data = (FIXTURES / name).read_bytes()
        specification = _accepted(import_hub_row(data, corpus=corpus, split=split, offset=offset, judge=judge))
        specification = label(specification)
        specifications.append(specification)
        (output / "specifications" / f"{specification.id.replace('/', '-')}.json").write_bytes(to_json(specification))
        inputs.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "split": split, "offset": offset})
    exports = []
    for specification in specifications:
        for rendering in RENDERINGS:
            destination = output / "harbor" / f"{specification.id.replace('/', '-')}-{rendering.id}"
            lower_to_harbor(specification, (rendering,), binding, destination)
            exports.append(
                {
                    "task": specification.id,
                    "rendering": rendering.id,
                    "path": str(destination.relative_to(output)),
                    "specification_sha256": specification_hash(specification),
                }
            )
    parquet = output / "specifications.parquet"
    write_parquet(specifications, str(parquet))
    if [specification_hash(item) for item in read_parquet(str(parquet))] != [
        specification_hash(item) for item in specifications
    ]:
        raise RuntimeError("Static NeMo specifications did not round-trip through Parquet")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "specifications": len(specifications),
                "exports": exports,
                "inputs": inputs,
                "binding": "chat with no environment",
                "scope": (
                    "Reviewed answer-only source rows with shared MCQA, math, and reference-answer verifier contracts."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-size-class", choices=("small", "medium", "large"), required=True)
    parser.add_argument("--judge-provider", required=True)
    parser.add_argument("--judge-base-url", required=True)
    args = parser.parse_args()
    build(
        args.output,
        JudgeConfig(
            JudgeModelPolicy(args.judge_model, args.judge_size_class, args.judge_provider, args.judge_base_url),
            JudgeView(),
        ),
    )
