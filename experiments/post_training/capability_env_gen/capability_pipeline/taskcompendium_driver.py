"""Small subprocess boundary around the pinned upstream TaskCompendium package.

This module is executed with ``uv run --project <pinned source>``.  All semantic
decoding, verifier-ontology checks, grading and Harbor lowering therefore come
from that source checkout rather than a second local interpretation of schema
0.9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import msgspec
from taskcompendium.execution import HarborTaskBinding
from taskcompendium.grading import grade_attempt, source_verifier
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import FinalState, TaskTroveVerifier, tasktrove_verifier
from taskcompendium.serialization import (
    from_json,
    renderings_from_json,
    specification_hash,
)
from tasktrove_verify.spec import Mode


def _load(bundle: Path):
    specification = from_json((bundle / "specification.json").read_bytes())
    renderings = renderings_from_json((bundle / "renderings.json").read_bytes())
    binding = msgspec.json.decode(
        (bundle / "binding.json").read_bytes(), type=HarborTaskBinding
    )
    for step in specification.steps:
        verifier = tasktrove_verifier(step.verifier)
        if verifier is not None:
            source_verifier(verifier)
    return specification, renderings, binding


def validate_and_lower(bundle: Path, output: Path) -> dict:
    specification, renderings, binding = _load(bundle)
    composite = bundle / "composite-verifier.json"
    allow_composite_judge_final_state = False
    if composite.is_file():
        from capability_pipeline.composite_extension import (
            validate_composite_lowering_authorization,
        )
        validate_composite_lowering_authorization(
            specification, bundle / "specification.json", composite,
        )
        allow_composite_judge_final_state = any(
            isinstance(rendering.submission, FinalState)
            and isinstance(step.verifier, TaskTroveVerifier)
            and step.verifier.mode == Mode.JUDGE
            for step, rendering in zip(specification.steps, renderings, strict=True)
        )
    if output.exists():
        shutil.rmtree(output)
    if allow_composite_judge_final_state:
        lower_to_harbor(
            specification, renderings, binding, output,
            composite_specification_path=bundle / "specification.json",
            composite_config_path=composite,
        )
    else:
        lower_to_harbor(specification, renderings, binding, output)
    return {
        "id": specification.id,
        "specification_sha256": (
            hashlib.sha256((bundle / "specification.json").read_bytes()).hexdigest()
            if allow_composite_judge_final_state
            else specification_hash(specification)
        ),
        "steps": len(specification.steps),
        "harbor_package": str(output),
    }


def run_direct_controls(bundle: Path, controls_path: Path) -> dict:
    specification, renderings, _ = _load(bundle)
    document = json.loads(controls_path.read_text())
    if document.get("schema_version") != "1" or not isinstance(
        document.get("cases"), list
    ):
        raise ValueError("controls.json must use schema_version 1 and contain cases")
    results = []
    for case in document["cases"]:
        case_id = case["id"]
        step_index = case.get("step_index", 0)
        if type(step_index) is not int or not 0 <= step_index < len(
            specification.steps
        ):
            raise ValueError(f"invalid step_index for {case_id}")
        workspace_name = case.get("workspace")
        if workspace_name:
            source = (bundle / workspace_name).resolve()
            if bundle.resolve() not in source.parents or not source.is_dir():
                raise ValueError(f"invalid control workspace for {case_id}")
            temporary = tempfile.TemporaryDirectory(prefix="task-control-")
            workspace = Path(temporary.name) / "workspace"
            shutil.copytree(source, workspace)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="task-control-")
            workspace = Path(temporary.name) / "workspace"
            workspace.mkdir()
        try:
            result = grade_attempt(
                specification,
                renderings[step_index],
                case.get("response"),
                workspace,
                tuple(case.get("transcript", [])),
                step_index=step_index,
            )
            built = msgspec.to_builtins(result)
        except Exception as error:  # noqa: BLE001 - upstream grader faults are runtime evidence
            built = {
                "status": "infra_error",
                "reward": None,
                "detail": {"error": f"{type(error).__name__}: {error}"},
            }
        finally:
            temporary.cleanup()
        results.append({"id": case_id, "class": case.get("class"), "result": built})
    return {"cases": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-and-lower")
    validate.add_argument("--bundle", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    controls = sub.add_parser("direct-controls")
    controls.add_argument("--bundle", type=Path, required=True)
    controls.add_argument("--controls", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-and-lower":
        result = validate_and_lower(args.bundle, args.output)
    else:
        result = run_direct_controls(args.bundle, args.controls)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
