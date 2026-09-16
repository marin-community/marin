# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate existing Wikipedia proxies with the frozen FineMath math protocol."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from marin.execution.remote import remote

from experiments.domain_phase_mix import evaluate_tpp10_finemath_math as math_eval
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

DIRECTORY = math_eval.DIRECTORY.parent / "wikipedia_math_eval_20260912"
REFERENCE = math_eval.DIRECTORY / "spec.json"


def validate_spec(spec: dict) -> None:
    """Require the identical scoring protocol and a complete Wikipedia proxy grid."""
    math_eval.validate_spec(spec)
    reference = spec["reference_spec"]
    math_eval.validate_spec(reference)
    if spec["wrapper_code_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Wikipedia evaluation launcher changed")
    changed = {
        "purpose",
        "endpoints",
        "spec_sha256",
        "contamination_status",
        "reference_spec",
        "wrapper_code_sha256",
        "baseline_result_canonical_sha256",
    }
    if {k: v for k, v in spec.items() if k not in changed} != {k: v for k, v in reference.items() if k not in changed}:
        raise ValueError("Wikipedia and FineMath scoring protocols differ")
    if spec["endpoints"][0] != reference["endpoints"][0]:
        raise ValueError("The shared zero-fraction checkpoint changed")
    for endpoint in spec["endpoints"][1:]:
        request = endpoint["request"]
        if (
            request["domain"] != "wikipedia"
            or request["arm"] != "matched"
            or request["run_name"] != f"tpp10_wikipedia_matched_p{request['percent']:03d}_s20260910"
        ):
            raise ValueError("Only the seven frozen Wikipedia matched proxies are released")


def build_spec(reference_path: Path) -> dict:
    reference = math_eval.repair.read_json(str(reference_path))
    math_eval.validate_spec(reference)
    reference_results = [math_eval.verified_result(reference, e) for e in reference["endpoints"]]
    if any(result is None for result in reference_results):
        raise ValueError("The reference FineMath math curve is incomplete")
    survey = math_eval.repair.read_json(str(math_eval.SURVEY))
    requests = sorted(
        (r for r in survey["runs"] if r["domain"] == "wikipedia" and r["arm"] == "matched"),
        key=lambda r: r["percent"],
    )
    if tuple(r["percent"] for r in requests) != math_eval.GRID[1:]:
        raise ValueError("Expected all seven nonzero Wikipedia proxies")
    endpoints = [reference["endpoints"][0]]
    for request in requests:
        evidence = math_eval.repair.training_evidence(request, survey)
        endpoints.append(
            {
                "request": request,
                "checkpoint": math_eval.repair.checkpoint_snapshot(request),
                "paloma_loss": evidence["metrics"][f"eval/{math_eval.repair.PALOMA}/loss"],
            }
        )
    spec = {
        **reference,
        "purpose": "Math likelihood on the complete existing Wikipedia proxy grid; shared baseline reused; no training",
        "endpoints": endpoints,
        "reference_spec": reference,
        "wrapper_code_sha256": file_sha256(Path(__file__)),
        "baseline_result_canonical_sha256": canonical_sha256(reference_results[0]),
        "contamination_status": (
            "No benchmark decontamination audit of the Wikipedia training subset; exploratory likelihood diagnostic"
        ),
    }
    spec.pop("spec_sha256")
    spec["spec_sha256"] = canonical_sha256(spec)
    validate_spec(spec)
    return spec


def reuse_baseline(spec: dict) -> None:
    """Copy the verified shared checkpoint measurement with explicit source lineage."""
    reference = spec["reference_spec"]
    endpoint = spec["endpoints"][0]
    source = math_eval.verified_result(reference, endpoint)
    if source is None or source["population_sha256"] != canonical_sha256(spec["population_counts"]):
        raise ValueError("Shared baseline measurement is absent or has a different population")
    if canonical_sha256(source) != spec["baseline_result_canonical_sha256"]:
        raise ValueError("The pinned shared baseline measurement changed")
    result = {
        **source,
        "spec_sha256": spec["spec_sha256"],
        "reused_from": {
            "uri": f"{math_eval.output_root(reference)}/{endpoint['request']['run_name']}.json",
            "source_spec_sha256": reference["spec_sha256"],
            "source_result_canonical_sha256": canonical_sha256(source),
        },
    }
    existing = math_eval.verified_result(spec, endpoint)
    if existing is not None:
        if existing != result:
            raise ValueError("Stored baseline reuse has different source lineage or metrics")
        return
    math_eval.repair.write_json(f"{math_eval.output_root(spec)}/{endpoint['request']['run_name']}.json", result)
    if math_eval.verified_result(spec, endpoint) is None:
        raise ValueError("Failed to preserve the shared baseline receipt")


async def submit(spec: dict) -> None:
    math_eval.experiment.require_central1()
    validate_spec(spec)
    math_eval.repair.write_json(math_eval.output_root(spec) + "/spec.json", spec)
    reuse_baseline(spec)
    if all(math_eval.verified_result(spec, e) is not None for e in spec["endpoints"]):
        return
    worker = remote(
        math_eval.evaluate,
        name="wikipedia-math-proxies",
        resources=math_eval.original.TPU,
        env_vars={"MARIN_PREFIX": math_eval.experiment.PREFIX},
    )
    await asyncio.to_thread(worker, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DIRECTORY / "spec.json")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.build:
        spec = build_spec(args.reference)
        args.spec.parent.mkdir(parents=True, exist_ok=True)
        args.spec.write_text(json.dumps(spec, indent=2) + "\n")
        print(json.dumps({"spec_sha256": spec["spec_sha256"], "new_evaluations": 7, "reused_baselines": 1}))
    else:
        asyncio.run(submit(math_eval.repair.read_json(str(args.spec))))


if __name__ == "__main__":
    main()
