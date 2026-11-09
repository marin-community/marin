# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import yaml

from experiments.evaluation.task334 import bfcl_five


def test_profile_is_the_m51_bfcl_profile_without_name_and_location():
    m51 = yaml.safe_load((bfcl_five.HERE / "m51-profile-reference.yaml").read_text())
    assert {k: v for k, v in m51.items() if k not in ("name", "location")} == bfcl_five.PROFILE


def test_model_and_command_bind_the_admitted_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl_five, "HERE", bfcl_five.HERE)
    admission = tmp_path / "merge-admission.json"
    admission.write_text(json.dumps({"admitted": True, "output": "s3://marin-rno2a/tmp/ttl=14d/x/hf"}))
    path = bfcl_five.model("fixture-t0", str(admission))
    try:
        value = yaml.safe_load(path.read_text())
        assert value["location"] == "s3://marin-rno2a/tmp/ttl=14d/x/hf"
        argv = bfcl_five.command("fixture-t0")
        assert argv[argv.index("--priority-band") + 1] == "interactive"
        assert argv[argv.index("--records-prefix") + 1].startswith(bfcl_five.SCREEN_ROOT + "/fixture-t0/")
    finally:
        path.unlink()
