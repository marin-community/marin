"""Remote-only complete ShellSim VFS snapshot protocol fixture."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from capability_pipeline.shellsim_snapshot_extension import (
    apply_overlay,
    extension_record,
)

LARGE = 17 * 1024 * 1024

def rpc(p, value, *, ok=True):
    p.stdin.write(json.dumps(value, separators=(",", ":")) + "\n"); p.stdin.flush()
    answer = json.loads(p.stdout.readline())
    if ok and answer.get("ok") is not True: raise RuntimeError(answer.get("error"))
    if not ok and answer.get("ok") is not False: raise RuntimeError("bounded snapshot unexpectedly succeeded")
    return answer.get("result") if ok else answer.get("error")

def run(out: Path, source: Path) -> dict:
    if os.environ.get("CAPABILITY_REMOTE_RESET_PROBE") != "1": raise RuntimeError("remote-only")
    with tempfile.TemporaryDirectory() as t:
        overlay = apply_overlay(source, Path(t) / "overlay"); target = Path(t) / "target"
        for args, name in ((["test"], "cargo-test.log"), (["build", "--release"], "cargo-build.log")):
            r = subprocess.run(["cargo", *args, "--locked", "--manifest-path", str(overlay / "shellsim-bridge/Cargo.toml"), "--target-dir", str(target)], text=True, capture_output=True, timeout=900, check=False)
            (out / name).write_text(r.stdout + r.stderr)
            if r.returncode:
                excerpt = (r.stdout + r.stderr)[-4000:].replace(str(overlay), "<overlay>")
                raise RuntimeError(f"{name} failed: {excerpt}")
        p = subprocess.Popen([str(target / "release/taskcompendium-shellsim")], text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        rpc(p, {"op":"init","limits":{"cpu":10000000,"memory":67108864,"disk":67108864,"output":4194304}})
        for cmd in ("mkdir -p /app/empty",):
            if rpc(p, {"op":"exec","command":cmd})["return_code"] != 0: raise RuntimeError(cmd)
        large_bytes = b"x" * LARGE
        rpc(p, {"op":"write","path":"/app/large.bin","data":base64.b64encode(large_bytes).decode("ascii")})
        for cmd in ("chmod 640 /app/large.bin", "ln -s large.bin /app/link"):
            if rpc(p, {"op":"exec","command":cmd})["return_code"] != 0: raise RuntimeError(cmd)
        limits={"max_entries":16,"max_file_bytes":40*1024*1024,"max_response_bytes":1024*1024}
        result=rpc(p,{"op":"snapshot","path":"/app","limits":limits}); snap=result["snapshot"]
        (out / "shellsim-snapshot-raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        canonical=json.dumps(snap,separators=(",",":"),ensure_ascii=True).encode()
        entries={x["path"]:x for x in snap["entries"]}
        expected=hashlib.sha256(large_bytes).hexdigest()
        checks={"snapshot_hash": result["snapshot_sha256"] == hashlib.sha256(canonical).hexdigest(),
                "large_hash": entries["large.bin"]["sha256"] == expected,
                "large_mode": entries["large.bin"]["mode"] == 0o640,
                "empty_dir": entries["empty"]["kind"] == "directory",
                "symlink_target": entries["link"]["target"] == "large.bin"}
        (out / "shellsim-snapshot-checks.json").write_text(json.dumps({"checks":checks,"expected_large_sha256":expected,"python_snapshot_sha256":hashlib.sha256(canonical).hexdigest()}, indent=2,sort_keys=True)+"\n")
        if not all(checks.values()): raise ValueError(f"snapshot evidence differs: {checks}")
        for bound in ({**limits,"max_entries":1},{**limits,"max_file_bytes":1},{**limits,"max_response_bytes":1}): rpc(p,{"op":"snapshot","path":"/app","limits":bound},ok=False)
        rpc(p,{"op":"close"}); p.wait(timeout=30)
    evidence={"schema_version":"capability-shellsim-snapshot-probe-v1","state":"ready","extension":extension_record(),"snapshot":result,"large_file_bytes":LARGE}
    (out / "shellsim-snapshot.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
    return {"state":"ready","summary":{"snapshot_sha256":result["snapshot_sha256"],"large_file_bytes":LARGE}}
