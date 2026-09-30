#!/usr/bin/env python3
"""envgen verifier: the one script every generated environment ships as its grader.

Runs under the tasktrove `script` mode contract (tasktrove_verify.modes.grade_script at
revision b76d0313): the environment exports TASKTROVE_TESTS_DIR (this file's bundle),
TASKTROVE_WORKSPACE (the candidate tree) and TASKTROVE_LOGS_DIR; the reward is reported
through $TASKTROVE_LOGS_DIR/reward.json AND as the last non-empty line of stdout.

The bundle it reads, laid out under $TASKTROVE_TESTS_DIR/envgen/:

    manifest.json   {"workdir": "/app/repo",              # informational; cwd is TASKTROVE_WORKSPACE
                     "env": {"GOFLAGS": "-mod=mod", ...},  # exported for every command
                     "overlay": [{"from": "tests/pkg/foo_test.go", "to": "pkg/foo_test.go"}, ...],
                     "p2p": {"cmd": "go test ./pkg/... -run 'TestA|TestB' -count=1", "timeout": 600} | null,
                     "f2p": {"cmd": "go test ./pkg/ -run 'TestNew' -count=1 -v", "timeout": 600},
                     "p2p_reason": "why p2p is null, if it is",
                     "mutant": {"patch": "mutant/revert_fix.patch", "cmd": "go test ./pkg/... -count=1",
                                "timeout": 600} | null,
                     "adapter": {"required": false, ...}}          # see adapter.py; host-side, before this runs
    tests/...       the test files named by `overlay`, as they exist at the reference commit
                    (or written by the environment author when the task is test_authoring)

Order matters and is the whole design: P2P runs FIRST, on the candidate tree as submitted,
so a fail-to-pass test file that does not even compile at base cannot drag the P2P set
down with it (Go and Java compile a package's tests together).  Then the overlay is copied
in and the F2P command runs.  Reward is 1.0 iff both commands exit 0; anything else is 0.0.
The per-command record (exit code, seconds, output tails) goes to
$TASKTROVE_LOGS_DIR/detail.json so a validator can say WHY a run failed, not just that it did.

`mutant` exists for test-authoring tasks, where the solver WRITES the tests: after F2P, the
mutant patch (the fix reverted) is applied and `mutant.cmd` is run; the reward requires
that command to FAIL, which is what proves the solver's tests detect the defect.  Without
new tests, the existing suite still passes on the mutant and the reward is 0.

No network is assumed or allowed at grade time: commands run with whatever the image
already holds.  A command that reaches for the network fails here, which is the point.

Python 3.5+ compatible on purpose: it runs under whatever `python3` the image has, and
2017-era repositories pin interpreters that predate `capture_output` and postponed
annotations (a 3.6 image made the first version die with SyntaxError -> reward None).
"""
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

TAIL = 6000


def _tail(s, n=TAIL):
    return s if len(s) <= n else s[-n:]


def _text(b):
    if b is None:
        return ""
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def run_cmd(cmd, cwd, env, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        return {"cmd": cmd, "exit": p.returncode, "seconds": round(time.time() - t0, 2),
                "timed_out": False, "stdout_tail": _tail(_text(p.stdout)), "stderr_tail": _tail(_text(p.stderr))}
    except subprocess.TimeoutExpired as e:
        return {"cmd": cmd, "exit": None, "seconds": round(time.time() - t0, 2), "timed_out": True,
                "stdout_tail": _tail(_text(e.stdout)), "stderr_tail": _tail(_text(e.stderr))}


def main():
    tests = pathlib.Path(os.environ["TASKTROVE_TESTS_DIR"])
    workspace = pathlib.Path(os.environ.get("TASKTROVE_WORKSPACE") or os.getcwd())
    logs = pathlib.Path(os.environ.get("TASKTROVE_LOGS_DIR") or "/tmp/envgen-logs")
    logs.mkdir(parents=True, exist_ok=True)
    bundle = tests / "envgen"
    detail = {"workspace": str(workspace), "runs": [], "overlay": [], "reward": 0.0, "error": None}

    def finish(reward):
        detail["reward"] = reward
        (logs / "detail.json").write_text(json.dumps(detail, indent=1))
        (logs / "reward.json").write_text(json.dumps({"reward": reward}) + "\n")
        print("%.1f" % reward)
        return 0

    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
    except Exception as e:  # noqa: BLE001
        detail["error"] = "manifest unreadable: %s" % e
        return finish(0.0)

    env = dict(os.environ)
    for k, v in (manifest.get("env") or {}).items():
        env[str(k)] = str(v)
    env.setdefault("CI", "1")

    p2p = manifest.get("p2p")
    f2p = manifest.get("f2p") or {}
    if not isinstance(f2p, dict) or not f2p.get("cmd"):
        detail["error"] = "manifest has no f2p.cmd"
        return finish(0.0)

    # 1. pass-to-pass, on the tree exactly as submitted.
    if isinstance(p2p, dict) and p2p.get("cmd"):
        r = run_cmd(p2p["cmd"], workspace, env, float(p2p.get("timeout") or 900))
        r["name"] = "p2p"
        detail["runs"].append(r)
    else:
        detail["runs"].append({"name": "p2p", "skipped": True, "reason": manifest.get("p2p_reason") or "no p2p in manifest", "exit": 0})

    # 2. overlay the fail-to-pass test files, then run them.
    for item in manifest.get("overlay") or []:
        src = bundle / item["from"]
        dst = workspace / item["to"]
        rec = {"from": item["from"], "to": item["to"]}
        try:
            if ".." in pathlib.PurePosixPath(item["to"]).parts or pathlib.PurePosixPath(item["to"]).is_absolute():
                raise ValueError("overlay target escapes the workspace")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copyfile(src, dst)
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = str(e)
        detail["overlay"].append(rec)
    if any(not o.get("ok") for o in detail["overlay"]):
        detail["error"] = "overlay failed"
        return finish(0.0)

    r = run_cmd(f2p["cmd"], workspace, env, float(f2p.get("timeout") or 900))
    r["name"] = "f2p"
    detail["runs"].append(r)

    ok = all((x.get("exit") == 0) for x in detail["runs"])

    # 3. mutant (test-authoring tasks): revert the fix, the solver's tests must now FAIL.
    mutant = manifest.get("mutant")
    if ok and isinstance(mutant, dict) and mutant.get("cmd") and mutant.get("patch"):
        patch = bundle / mutant["patch"]
        ap = run_cmd("git apply --whitespace=nowarn %s || patch -p1 --forward --batch < %s" % (shlex.quote(str(patch)), shlex.quote(str(patch))),
                     workspace, env, 300)
        ap["name"] = "mutant_apply"
        detail["runs"].append(ap)
        if ap["exit"] != 0:
            detail["error"] = "mutant patch did not apply"
            return finish(0.0)
        m = run_cmd(mutant["cmd"], workspace, env, float(mutant.get("timeout") or 900))
        m["name"] = "mutant"
        m["expect"] = "fail"
        detail["runs"].append(m)
        # A mutant that still passes means the tests do not detect the defect.
        ok = (m["exit"] not in (0, None))
    return finish(1.0 if ok else 0.0)


if __name__ == "__main__":
    sys.exit(main())
