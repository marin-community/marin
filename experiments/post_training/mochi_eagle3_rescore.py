# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify saved Mochi autoregressive outputs against a fresh target-only teacher."""

import hashlib
import json
import logging

from marin.inference.vllm_server import VllmEnvironment
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.throughput import write_json
from experiments.post_training.mochi_eagle3 import worker_spec
from experiments.post_training.mochi_eagle3_evaluate import (
    AUTOREGRESSIVE_ARTIFACT,
    prefix_scores,
    run,
    score_difference,
)


async def qualify_saved_outputs(environment: VllmEnvironment, spec: dict, arm: dict) -> None:
    """Apply frozen bounds to independent target controls before candidate scores."""
    if arm["draft"] is not None:
        raise ValueError("Saved outputs require a target-only teacher")
    source = StoragePath(spec["recorded_evaluation"])
    calibration_payload = (source / "correctness-calibration.json").read_bytes()
    if hashlib.sha256(calibration_payload).hexdigest() != spec["calibration_sha256"]:
        raise ValueError("The frozen correctness calibration changed")
    calibration = json.loads(calibration_payload)
    if not calibration["sensitivity_passed"] or not calibration["negative_controls_passed"]:
        raise ValueError("The recorded calibration did not qualify its target-only controls")
    target_payload = (source / spec["recorded_target_arm"] / AUTOREGRESSIVE_ARTIFACT).read_bytes()
    if hashlib.sha256(target_payload).hexdigest() != spec["recorded_outputs_sha256"][spec["recorded_target_arm"]]:
        raise ValueError("The recorded target-only outputs changed")
    target_scores = json.loads(target_payload)["scores"]
    controls = (
        (
            "fixed-target-control",
            calibration["reference"],
            calibration["probes"],
            calibration["fixed_bounds"],
        ),
        (
            "autoregressive-target-control",
            target_scores,
            list(target_scores.values()),
            calibration["decode_bounds"],
        ),
    )
    for name, reference, probes, bounds in controls:
        teacher = await prefix_scores(environment.server_url, probes, 16)
        difference = score_difference(reference, teacher)
        passed = all(difference[key] <= bound for key, bound in bounds.items())
        write_json(
            spec["output"],
            f"{name}.json",
            {"passed": passed, "bounds": bounds, "difference": difference, "teacher_scores": teacher},
        )
        if not passed:
            raise ValueError(f"Fresh target-only teacher failed the frozen {name} bounds")
    for name in spec["recorded_candidate_arms"]:
        payload = (source / name / AUTOREGRESSIVE_ARTIFACT).read_bytes()
        payload_digest = hashlib.sha256(payload).hexdigest()
        if payload_digest != spec["recorded_outputs_sha256"][name]:
            raise ValueError(f"The recorded candidate outputs changed: {name}")
        recorded = json.loads(payload)
        teacher = await prefix_scores(environment.server_url, list(recorded["scores"].values()), 16)
        difference = score_difference(recorded["scores"], teacher)
        passed = all(difference[key] <= bound for key, bound in calibration["decode_bounds"].items())
        write_json(
            spec["output"],
            f"{name}/autoregressive-correctness.json",
            {
                "passed": passed,
                "recorded_evaluation": str(source),
                "recorded_arm": name,
                "recorded_outputs_sha256": payload_digest,
                "calibration_sha256": spec["calibration_sha256"],
                "bounds": calibration["decode_bounds"],
                "difference": difference,
                "teacher_scores": teacher,
            },
        )
        if not passed:
            raise ValueError(f"Saved candidate {name} exceeded the frozen autoregressive correctness bounds")


def main() -> None:
    spec = worker_spec(__doc__)
    if len(spec["arms"]) != 1 or not spec["recorded_candidate_arms"]:
        raise ValueError("Saved-output qualification requires one teacher and at least one recorded candidate")
    run(spec, qualify_saved_outputs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
