import re

import pytest

from scripts.preflight_stage_bundle import check_stage


def test_stage_preflight_requires_payload_and_excludes_other_leaves(tmp_path):
    stage = tmp_path / "submissions/current"
    stage.mkdir(parents=True)
    worker = stage / "worker.sh"
    worker.write_text("# trusted worker fixture\n")
    payload = stage / "payload.json"
    payload.write_text("{}")
    cache = stage / "cache.pyc"
    cache.write_bytes(b"ignored")
    default = re.compile(r"\.pyc$")
    with pytest.raises(ValueError, match="omits required"):
        check_stage(tmp_path, stage, [worker], default)
    assert check_stage(tmp_path, stage, [worker, payload], default)["stage_files"] == 2
    other = tmp_path / "submissions/other/worker.sh"
    with pytest.raises(ValueError, match="another submission"):
        check_stage(tmp_path, stage, [worker, payload, other], default)
    link = stage / "linked.json"
    link.symlink_to(payload)
    with pytest.raises(ValueError, match="symlink"):
        check_stage(tmp_path, stage, [worker, payload, link], default)
