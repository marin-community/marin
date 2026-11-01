# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from experiments.evaluation.serving_parity.compare import (
    IneligibleStack,
    compare_arms,
    frozen_reference,
    validate_arms,
    verify_reference,
)
from experiments.evaluation.serving_parity.settings import TASKS, HardwareKind, frozen_items


def artifact(root, hardware, **extra):
    root.mkdir()
    (root / "complete.json").write_text("{}")
    provenance = dict(
        model="model",
        revision="revision",
        manifest_sha256="manifest",
        evaluator_source="client",
        arm=dict(hardware=hardware),
        shipped_stack_eligible=True,
        runtime_patches=[],
        effective_tpu_pin="shipped",
    )
    provenance.update(extra)
    (root / "provenance.json").write_text(json.dumps(provenance))
    return str(root)


def test_duplicate_prefix_and_mislabeled_hardware_cannot_pass(tmp_path):
    h100 = artifact(tmp_path / "h100", "h100")
    v5p = artifact(tmp_path / "v5p", "v5p-8")
    with pytest.raises(ValueError, match="distinct artifact"):
        validate_arms({"h100": h100, "v6e-8": h100 + "/", "v5p-8": v5p})
    copied = artifact(tmp_path / "copy", "h100")
    with pytest.raises(ValueError, match="label differs"):
        validate_arms({"h100": h100, "v6e-8": copied, "v5p-8": v5p})


def test_evaluator_change_and_experimental_kernel_cannot_pass(tmp_path):
    reference = artifact(tmp_path / "reference", "h100")
    changed = artifact(tmp_path / "changed", "v6e-8", evaluator_source="different-client")
    with pytest.raises(ValueError, match="evaluator_source"):
        compare_arms(changed, reference)
    experimental = artifact(tmp_path / "experimental", "v4-32", runtime_patches=[dict(arithmetic_changed=True)])
    with pytest.raises(IneligibleStack):
        compare_arms(experimental, reference)


def test_three_distinct_shipped_arms_produce_complete_comparison(tmp_path):
    arms = {
        kind.value: artifact(tmp_path / kind.value, kind.value)
        for kind in (HardwareKind.H100, HardwareKind.V6E, HardwareKind.V5P)
    }
    tokens = [
        dict(token_id=i, logprob=None if i == 0 else -0.1, top1_token_id=None if i == 0 else i) for i in range(5279)
    ]
    for prefix in arms.values():
        root = tmp_path / prefix.split("/")[-1]
        for concurrency in (1, 4):
            directory = root / f"teacher-{concurrency}"
            directory.mkdir()
            for i in range(concurrency):
                (directory / f"{i}.json").write_text(
                    json.dumps(dict(case_id="303", input_sha256="input", tokens=tokens))
                )
        for task in TASKS:
            (root / f"{task}-items.json").write_text(
                json.dumps(
                    [
                        dict(
                            id=item["id"],
                            correct=True,
                            finish_reason="stop" if task == "gsm8k-0shot" else "loglikelihood",
                        )
                        for item in frozen_items(task)
                    ]
                )
            )
    validate_arms(arms)
    for candidate in arms.values():
        report = compare_arms(candidate, arms["h100"])
        assert report["status"] == "PASS"
        assert len(report["teacher"]) == 5
        assert set(report["accuracy"]) == set(TASKS)

    reference_root = Path(arms["h100"])
    manifest = tmp_path / "approved.json"
    manifest.write_text(
        json.dumps(
            dict(
                reference_prefix=arms["h100"],
                files={
                    str(p.relative_to(reference_root)): dict(
                        bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
                    )
                    for p in reference_root.rglob("*")
                    if p.is_file()
                },
            )
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with frozen_reference(arms["h100"], str(manifest), digest) as snapshot:
        (reference_root / "teacher-1/0.json").write_text("overwritten after verification")
        assert compare_arms(arms["v5p-8"], snapshot)["status"] == "PASS"


def test_three_tpu_arms_cannot_replace_h100(tmp_path):
    arms = {
        kind.value: artifact(tmp_path / kind.value, kind.value)
        for kind in (HardwareKind.V4, HardwareKind.V5P, HardwareKind.V6E)
    }
    with pytest.raises(ValueError, match="Require H100"):
        validate_arms(arms)


def test_reference_hash_manifest_rejects_changed_artifacts(tmp_path):
    reference = artifact(tmp_path / "reference", "h100")
    root = tmp_path / "reference"
    for name in ("teacher-1/0.json", *(f"{task}-items.json" for task in TASKS)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                reference_prefix=reference,
                files={
                    str(p.relative_to(root)): dict(
                        bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
                    )
                    for p in root.rglob("*")
                    if p.is_file()
                },
            )
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    verify_reference(reference, str(manifest), digest)
    (root / "teacher-1/0.json").write_text("changed")
    with pytest.raises(ValueError, match="Reference artifact changed"):
        verify_reference(reference, str(manifest), digest)
    with pytest.raises(ValueError, match="approved digest"):
        verify_reference(reference, str(manifest), "0" * 64)
