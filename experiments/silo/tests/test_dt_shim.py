# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The generated dt.py, run as the CLI builder agents use, against a live stack."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from silo.dt_shim import generate_dt
from test_end_to_end import stack  # noqa: F401 - fixture

HERE = Path(__file__).parent
ORIGINAL = (HERE / "pipeline_contract" / "dt.py.orig").read_text()


def test_vendored_dt_py_is_byte_identical_to_upstream():
    # Formatters never touch a .orig file, but check anyway: the whole point of
    # the fixture is that it is the real, unmodified CLI.
    provenance = (HERE / "pipeline_contract" / "dt.py.orig.PROVENANCE").read_text()
    pinned = next(line.split()[1] for line in provenance.splitlines() if line.startswith("sha256:"))
    assert hashlib.sha256((HERE / "pipeline_contract" / "dt.py.orig").read_bytes()).hexdigest() == pinned


DIGEST = "docker.io/library/alpine@sha256:" + "a" * 64


@pytest.fixture
def dt(tmp_path, stack):  # noqa: F811
    script = tmp_path / "dt.py"
    script.write_text(generate_dt(ORIGINAL))
    env = {
        "PATH": os.environ["PATH"],
        "SILO_API_TOKEN": "api-token",
        "SILO_BROKER_URL": stack.broker_server.url,
        "DT_LOG": str(tmp_path / "dt_calls.jsonl"),
        "DT_KEY": "k1",
        # Stubs first: a fake `daytona` for the param classes, a poisoned `silo`.
        "PYTHONPATH": str(HERE / "stubs"),
    }

    def run(*args: str, expect_ok: bool = True) -> dict:
        proc = subprocess.run(
            [sys.executable, "-s", str(script), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.stdout, proc.stderr
        result = json.loads(proc.stdout)
        assert result["ok"] is expect_ok, (result, proc.stderr[-2000:])
        return result

    run.tmp = tmp_path
    return run


def test_generate_dt_leaves_every_other_function_untouched():
    generated = generate_dt(ORIGINAL)
    for fn in ("def snapshot_get(", "def run_in_sandbox(", "def upload_path(", "def validate("):
        start = ORIGINAL.index(fn)
        body = ORIGINAL[start : ORIGINAL.index("\ndef ", start + 1)]
        assert body in generated, fn


def test_generation_fails_loudly_if_dt_py_changed_upstream():
    with pytest.raises(RuntimeError, match=r"upstream dt\.py changed"):
        generate_dt(ORIGINAL.replace('key = os.environ.get("DAYTONA_API_KEY", "")', "key = 1"))


def test_builder_workflow_through_the_real_cli(dt):
    dockerfile = dt.tmp / "Dockerfile"
    dockerfile.write_text(f"FROM {DIGEST}\n")

    built = dt("snapshot", "build", "--dockerfile", str(dockerfile), "--name", "eg-run-abc")
    assert built["name"] == "eg-run-abc" and "ACTIVE" in built["state"].upper()
    assert built["quota_stalls"] == 0

    got = dt("snapshot", "get", "eg-run-abc")
    assert got["dockerfile_in_build_info"] is True

    sandbox = dt("sandbox", "create", "--snapshot", "eg-run-abc", "--no-network")
    assert sandbox["network_blocked"] is True
    sid = sandbox["id"]

    ran = dt("exec", sid, "--json", "--", "echo hello-from-silo")
    assert ran["exit"] == 0 and ran["stdout"].strip() == "hello-from-silo"

    (dt.tmp / "in.txt").write_text("payload")
    dt("upload", sid, str(dt.tmp / "in.txt"), "/work/in.txt")
    dt("download", sid, "/work/in.txt", str(dt.tmp / "out.txt"))
    assert (dt.tmp / "out.txt").read_text() == "payload"

    listed = dt("sandbox", "list")
    assert [row["id"] for row in listed["sandboxes"]] == [sid]
    dt("sandbox", "delete-mine")

    # The call log the build report is checked against is still written.
    calls = [json.loads(line)["cmd"] for line in (dt.tmp / "dt_calls.jsonl").read_text().splitlines()]
    assert calls[:3] == ["snapshot.build", "snapshot.get", "sandbox.create"]


def test_networked_sandbox_request_gets_an_instructive_refusal(dt):
    dockerfile = dt.tmp / "Dockerfile"
    dockerfile.write_text(f"FROM {DIGEST}\n")
    dt("snapshot", "build", "--dockerfile", str(dockerfile), "--name", "eg-net")
    refused = dt("sandbox", "create", "--snapshot", "eg-net", expect_ok=False)
    # The refusal names the flag the agent should pass instead.
    assert "--no-network" in refused["error"]


def test_copy_is_still_refused_by_dt_itself(dt):
    dockerfile = dt.tmp / "Dockerfile"
    dockerfile.write_text(f"FROM {DIGEST}\nCOPY . /app\n")
    refused = dt("snapshot", "build", "--dockerfile", str(dockerfile), "--name", "eg-copy", expect_ok=False)
    assert "COPY" in refused["error"]
