import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "c32_probe", Path(__file__).resolve().parents[1] / "scripts/run_c32_semantic_probe.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def staged(tmp_path):
    for name in MOD.EXPECTED_HASHES:
        (tmp_path / name).write_text("fixture")
    image = MOD.VERIFIER_IMAGE
    (tmp_path / "specification.json").write_text(json.dumps({
        "steps": [{"verifier": {"runtime": {"image": image}}}],
    }))
    contract = {
        "input_hashes": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in MOD.EXPECTED_HASHES
        },
        "private_image": image,
        "snapshot": MOD.VERIFIER_SNAPSHOT,
        "expected_state": "counterexample_rejected",
    }
    (tmp_path / "probe-contract.json").write_text(json.dumps(contract))
    return contract


def test_repaired_probe_validates_all_seven_inputs(tmp_path):
    contract = staged(tmp_path)
    actual, observed = MOD.verify_inputs(tmp_path)
    assert actual == contract["input_hashes"]
    assert observed == contract


def test_repaired_probe_rejects_stale_embedded_specification(tmp_path):
    staged(tmp_path)
    (tmp_path / "specification.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        MOD.verify_inputs(tmp_path)


@pytest.mark.parametrize("change,match", [
    ({"input_hashes": {}}, "seven"),
    ({"private_image": "another"}, "image"),
    ({"snapshot": "another"}, "snapshot"),
    ({"expected_state": "false_positive_confirmed"}, "rejection"),
])
def test_repaired_probe_rejects_contract_mismatch(tmp_path, change, match):
    contract = staged(tmp_path)
    contract.update(change)
    (tmp_path / "probe-contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match=match):
        MOD.verify_inputs(tmp_path)
