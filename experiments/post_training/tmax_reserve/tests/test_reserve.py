# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import zipfile

from experiments.post_training.tmax_reserve.reserve import TMUX_BAKE, draw, package, screen

TOML = '[metadata]\ndomain = "{domain}"\n'


def _task(z: zipfile.ZipFile, name: str, instruction: str, tests: str, *, domain: str = "debugging", extra=()) -> None:
    z.writestr(f"{name}/instruction.md", instruction)
    z.writestr(f"{name}/task.toml", TOML.format(domain=domain))
    z.writestr(f"{name}/tests/test_final_state.py", tests)
    z.writestr(f"{name}/environment/Dockerfile", "FROM ubuntu:22.04\nRUN bash /tmp/post_install.sh\n")
    z.writestr(f"{name}/environment/post_install.sh", "mkdir -p /app/verifier && cp ref /app/verifier/truth.txt\n")
    for path in extra:
        z.writestr(f"{name}/{path}", "x")


def _archive(build) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        build(z)
    return zipfile.ZipFile(buf)


def test_screen_excludes_hidden_assets_oracle_fixtures_and_sft_overlap():
    def reserve(z):
        _task(z, "hidden", "Fix the parser.", "open('/app/verifier/truth.txt')")
        _task(z, "named", "Match /app/verifier/truth.txt exactly.", "open('/app/verifier/truth.txt')")
        _task(z, "oracle", "Write a decoder.", "assert True", extra=["environment/_fixtures/app/fixtures/oracle"])
        _task(z, "copied", "Sort the CSV by date.", "assert True")
        _task(z, "clean", "Count the lines.", "assert True")

    def sft(z):
        z.writestr("t0/task.json", json.dumps({"description": "Sort the  CSV by date."}))
        z.writestr("t0/test_final_state.py", "different")

    rows = {row["task"]: row["excluded"] for row in screen(_archive(reserve), _archive(sft))}
    assert rows["hidden"] == ["image_path", "answer_name"]
    assert rows["named"] == []
    assert rows["oracle"] == ["oracle_fixture"]
    assert rows["copied"] == ["sft_overlap"]
    assert rows["clean"] == []


def test_draw_is_seeded_stratified_and_skips_excluded_or_contaminated():
    rows = [{"task": f"a{i}", "domain": "a", "excluded": []} for i in range(30)]
    rows += [{"task": f"b{i}", "domain": "b", "excluded": []} for i in range(10)]
    rows += [
        {"task": "bad", "domain": "a", "excluded": ["image_path"]},
        {"task": "dirty", "domain": "b", "excluded": []},
    ]
    decon = {row["task"]: {"contaminated": row["task"] == "dirty"} for row in rows}
    picked = draw(rows, decon, size=8, seed=1)
    assert picked == draw(rows, decon, size=8, seed=1)
    assert sum(t.startswith("a") for t in picked) == 6 and sum(t.startswith("b") for t in picked) == 2
    assert {"bad", "dirty"}.isdisjoint(picked)


def test_package_bakes_tmux_and_keeps_every_other_byte(tmp_path):
    def reserve(z):
        _task(z, "t1", "Count the lines.", "assert True")

    archive = _archive(reserve)
    manifest = package(archive, ["t1"], tmp_path / "tasks")
    dockerfile = (tmp_path / "tasks" / "t1" / "environment" / "Dockerfile").read_text()
    assert dockerfile.endswith(TMUX_BAKE + "\n")
    assert dockerfile.startswith(archive.read("t1/environment/Dockerfile").decode().rstrip("\n"))
    for rel in ("instruction.md", "tests/test_final_state.py", "environment/post_install.sh", "task.toml"):
        assert (tmp_path / "tasks" / "t1" / rel).read_bytes() == archive.read(f"t1/{rel}")
    assert package(archive, ["t1"], tmp_path / "again")["sha256"] == manifest["sha256"]
