#!/usr/bin/env python3
"""Validate one environment cold from its snapshot.  Records, never asserts.

    validate_environment(daytona, snapshot, bundle, workdir, oracle, ref_shas, ref_runs, ...)

What it does, each step on a FRESH sandbox created from the snapshot with all egress
blocked (so "no network at grade time" is enforced, not hoped for):

  base   sandbox -> upload bundle -> leak check -> run verify.py           expect reward 0
  ref xN sandbox -> upload bundle -> apply oracle patch -> run verify.py   expect reward 1

Per run it keeps the verifier's own detail.json (per-command exit, seconds, output tails)
so the record says WHY: for the base run the F2P failure is classified heuristically as
assertion / compile_error / missing_target / timeout / other, with the matching line as
evidence, because a test that fails at base by not compiling certifies nothing about the
task.  `valid` is computed here from those records; the agent gets the verdict, not the pen.

Adapter-required environments (manifest.adapter.required): the prewritten tests are
rewritten against the tree by adapter.py BEFORE verify.py runs, on the host, because a
Daytona sandbox cannot reach the GLM relay.  The validator does that here: it pulls the
tree's context files out of the sandbox, runs the adapter locally, and uploads the adapted
tests over the bundle copy.  Both the adapted and the static (unadapted) results are
recorded for the ref run, so a rewrite that quietly weakens a test shows up as
"static 1, adapted 1, diff N lines" rather than being invisible.
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import shlex
import sys
import tarfile
import tempfile
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent

TESTS_REMOTE = "/envgen_tests"
LOGS_REMOTE = "/envgen_logs"

# Ordered: the first family that matches wins.  Evidence line is recorded alongside.
FAILURE_CLASSES: list[tuple[str, list[str]]] = [
    # Strong test-failure signatures first: a test that RAN and failed is the case we want,
    # and its output routinely also contains "No such file" noise from setup steps.
    ("assertion", [r"^--- FAIL", r"^FAIL[: ]", r"\bFAILED\b", r"AssertionError", r"# of unexpected failures",
                   r"unexpected failures", r"^\s*FAIL:", r"assert(?:ion)? (?:failed|error)", r"Assertion",
                   r"expected .* (?:but )?(?:got|was|to)", r"Expected", r"\bfailures?: [1-9]", r"[1-9]\d* failed",
                   r"\bnot ok\b", r"panic: ", r"✗", r"✘", r"Tests? failed", r"FAILURES"]),
    ("timeout", [r"panic: test timed out", r"\btimed out\b", r"Timeout exceeded"]),
    ("compile_error", [r"\bundefined: ", r"\[build failed\]", r"build failed", r"setup failed",
                       r"cannot use .* as .* value", r"too many arguments", r"not enough arguments",
                       r"has no field or method", r"declared and not used", r"imported and not used",
                       r"error\[E\d+\]", r"error: cannot find symbol", r"COMPILATION ERROR",
                       r"Could not compile", r"compilation failed", r"ImportError", r"ModuleNotFoundError",
                       r"SyntaxError", r"AttributeError: module", r"cannot find name", r"error TS\d+",
                       r"Cannot find module", r"is not a member of", r"undeclared identifier",
                       r"no member named", r"was not declared in this scope", r"error: (?:expected|unknown type)",
                       r"could not find function", r"NoMethodError", r"NameError",
                       r"parse error", r"Parse error", r"compile error", r"Compilation error"]),
    ("missing_target", [r"no test files", r"no tests to run", r"no tests ran", r"No tests found",
                        r"testing: warning: no tests to run", r"Unknown test", r"No test matched", r"matches no test",
                        r"collected 0 items", r"ERROR: file or directory not found", r"ERROR: not found:",
                        r"cannot find package", r"No such test", r"unknown target"]),
]


def classify_failure(text: str) -> tuple[str, str]:
    for cls, pats in FAILURE_CLASSES:
        for pat in pats:
            m = re.search(pat, text, re.M)
            if m:
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.end())
                line = text[line_start: line_end if line_end != -1 else None].strip()
                return cls, line[:300]
    return "other", ""


def _sh(sb, cmd: str, cwd: str | None = None, env: dict | None = None, timeout: int = 600) -> dict:
    sys.path.insert(0, str(HERE))
    from dt import run_in_sandbox
    return run_in_sandbox(sb, cmd, cwd, env, timeout)


def _upload_dir(sb, local: pathlib.Path, remote: str) -> dict:
    sys.path.insert(0, str(HERE))
    from dt import upload_path
    return upload_path(sb, local, remote)


def _download_json(sb, remote: str) -> dict | None:
    try:
        data = sb.fs.download_file(remote)
        return json.loads((data or b"{}").decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None


def _new_sandbox(d, snapshot: str, labels: dict, no_network: bool, purpose: str):
    from daytona import CreateSandboxFromSnapshotParams
    t0 = time.time()
    lab = dict(labels)
    lab["envgen_purpose"] = purpose
    sb = d.create(CreateSandboxFromSnapshotParams(
        snapshot=snapshot, labels=lab, ephemeral=True, auto_stop_interval=0, ttl_minutes=180,
        network_block_all=no_network), timeout=600)
    return sb, round(time.time() - t0, 2)


def leak_check(sb, workdir: str, ref_shas: list[str]) -> dict:
    """The reference commit must not be reachable from the workspace."""
    checks: dict = {"has_git": None, "commits_in_history": None, "ref_reachable": {}, "ok": True}
    r = _sh(sb, f"cd {shlex.quote(workdir)} && git rev-parse --is-inside-work-tree 2>/dev/null", timeout=60)
    checks["has_git"] = (r["exit"] == 0)
    if checks["has_git"]:
        r = _sh(sb, f"cd {shlex.quote(workdir)} && git rev-list --all --count 2>/dev/null", timeout=120)
        try:
            checks["commits_in_history"] = int((r["stdout"] or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            checks["commits_in_history"] = None
        for sha in ref_shas:
            r = _sh(sb, f"cd {shlex.quote(workdir)} && git cat-file -e {shlex.quote(sha)}^{{commit}} 2>/dev/null && echo REACHABLE || echo ABSENT", timeout=60)
            reach = "REACHABLE" in (r["stdout"] or "")
            checks["ref_reachable"][sha] = reach
            if reach:
                checks["ok"] = False
        r = _sh(sb, f"cd {shlex.quote(workdir)} && (grep -rIl -e {' -e '.join(shlex.quote(s[:12]) for s in ref_shas) if ref_shas else 'ZZZZ'} .git/packed-refs .git/FETCH_HEAD .git/ORIG_HEAD 2>/dev/null | head -3; true)", timeout=60)
        checks["ref_named_in_git_metadata"] = (r["stdout"] or "").strip().splitlines()
        if checks["ref_named_in_git_metadata"]:
            checks["ok"] = False
    return checks


def run_verifier(sb, workdir: str, extra_env: dict | None = None, timeout: int = 1800) -> dict:
    logs = f"{LOGS_REMOTE}/{uuid.uuid4().hex[:8]}"
    env = {"TASKTROVE_TESTS_DIR": TESTS_REMOTE, "TASKTROVE_WORKSPACE": workdir, "TASKTROVE_LOGS_DIR": logs}
    env.update(extra_env or {})
    t0 = time.time()
    r = _sh(sb, f"mkdir -p {logs} && python3 {TESTS_REMOTE}/envgen/verify.py", cwd=workdir, env=env, timeout=timeout)
    detail = _download_json(sb, f"{logs}/detail.json") or {}
    reward_doc = _download_json(sb, f"{logs}/reward.json") or {}
    reward = reward_doc.get("reward")
    if reward is None:
        lines = [ln for ln in (r["stdout"] or "").splitlines() if ln.strip()]
        try:
            reward = float(lines[-1]) if lines else None
        except ValueError:
            reward = None
    runs = {x.get("name"): x for x in detail.get("runs", []) if isinstance(x, dict)}
    rec = {"reward": reward, "verify_s": round(time.time() - t0, 1), "exit": r["exit"], "timed_out": r["timed_out"],
           "stdout_tail": (r["stdout"] or "")[-1500:], "stderr_tail": (r["stderr"] or "")[-1500:] if r["stderr"] else "",
           "detail": detail}
    for name in ("p2p", "f2p"):
        x = runs.get(name) or {}
        rec[f"{name}_exit"] = x.get("exit")
        rec[f"{name}_seconds"] = x.get("seconds")
        rec[f"{name}_tail"] = ((x.get("stdout_tail") or "") + "\n" + (x.get("stderr_tail") or ""))[-2500:]
        rec[f"{name}_skipped"] = bool(x.get("skipped"))
    return rec


def apply_oracle(sb, workdir: str, oracle: pathlib.Path) -> dict:
    remote = f"/tmp/oracle-{uuid.uuid4().hex[:8]}.patch"
    sb.fs.upload_file(oracle.read_bytes(), remote)
    cmd = (f"cd {shlex.quote(workdir)} && (git apply --whitespace=nowarn --verbose {remote} 2>&1 && echo APPLIED_WITH=git-apply) "
           f"|| (patch -p1 --forward --batch < {remote} 2>&1 && echo APPLIED_WITH=patch)")
    r = _sh(sb, cmd, timeout=300)
    return {"applied": r["exit"] == 0 and "APPLIED_WITH=" in (r["stdout"] or ""), "exit": r["exit"],
            "tail": (r["stdout"] or "")[-1200:]}


def run_adapter(sb, workdir: str, bundle: pathlib.Path, manifest: dict, adapter_context: str) -> dict:
    """Host-side adapter: pull context files from the sandbox, rewrite tests with GLM-5.3, push them back."""
    sys.path.insert(0, str(HERE))
    from adapter import adapt_bundle
    ad = manifest.get("adapter") or {}
    globs = list(ad.get("context_globs") or [])
    with tempfile.TemporaryDirectory(prefix="envgen-adapter-") as tmp:
        ws = pathlib.Path(tmp) / "ws"
        ws.mkdir()
        # Fetch the context files as one tar so a 40-file context is one round trip.
        find_expr = " -o ".join(f"-path {shlex.quote('./' + g.lstrip('./'))}" for g in globs) or "-false"
        r = _sh(sb, f"cd {shlex.quote(workdir)} && find . -type f \\( {find_expr} \\) -size -400k | head -400 | tar czf /tmp/adapter-ctx.tgz -T - 2>/dev/null; ls -la /tmp/adapter-ctx.tgz", timeout=120)
        try:
            data = sb.fs.download_file("/tmp/adapter-ctx.tgz") or b""
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                tf.extractall(ws, filter="data")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"context fetch failed: {e}", "find_out": (r["stdout"] or "")[-300:]}
        outdir = pathlib.Path(tmp) / "adapted"
        res = adapt_bundle(bundle, ws, outdir, manifest, extra_context=adapter_context or "")
        pushed = []
        for rel in res.get("written", []):
            src = outdir / rel
            if src.is_file():
                sb.fs.upload_file(src.read_bytes(), f"{TESTS_REMOTE}/envgen/{rel}")
                pushed.append(rel)
        res["pushed"] = pushed
        return res


def validate_environment(d, *, snapshot: str, bundle: pathlib.Path, workdir: str, oracle: pathlib.Path | None,
                         ref_shas: list[str], ref_runs: int = 1, run_timeout: int = 1800, adapter_context: str = "",
                         labels: dict | None = None, no_network: bool = True, keep_failed: bool = False) -> dict:
    labels = dict(labels or {"envgen": "1"})
    t_start = time.time()
    res: dict = {"snapshot": snapshot, "workdir": workdir, "network_blocked": no_network, "bundle": str(bundle),
                 "oracle": str(oracle) if oracle else None, "ref_shas": ref_shas, "ref_runs": ref_runs,
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "timings": {}, "reasons": [], "valid": False, "sandboxes": []}
    reasons = res["reasons"]
    mpath = bundle / "envgen" / "manifest.json"
    if not mpath.is_file():
        reasons.append(f"bundle has no envgen/manifest.json at {mpath}")
        res["summary"] = "bundle malformed"
        return res
    try:
        manifest = json.loads(mpath.read_text())
    except Exception as e:  # noqa: BLE001
        reasons.append(f"manifest.json unparseable: {e}")
        res["summary"] = "bundle malformed"
        return res
    if not (manifest.get("f2p") or {}).get("cmd"):
        reasons.append("manifest.f2p.cmd is missing")
    for item in manifest.get("overlay") or []:
        if not (bundle / "envgen" / item.get("from", "")).exists():
            reasons.append(f"overlay source missing from bundle: {item.get('from')}")
    if not (bundle / "envgen" / "verify.py").is_file():
        reasons.append("bundle has no envgen/verify.py (copy the harness one)")
    if not oracle or not oracle.is_file():
        reasons.append("oracle patch missing")
    if reasons:
        res["summary"] = "bundle malformed"
        return res
    adapter_required = bool((manifest.get("adapter") or {}).get("required"))
    res["verifier_kind"] = "adapter" if adapter_required else "static"

    def cleanup(sb):
        try:
            sb.delete()
        except Exception:  # noqa: BLE001
            pass

    # ---- base -----------------------------------------------------------------
    try:
        sb, create_s = _new_sandbox(d, snapshot, labels, no_network, "validate-base")
    except Exception as e:  # noqa: BLE001
        reasons.append(f"sandbox create from snapshot failed: {type(e).__name__}: {str(e)[:300]}")
        res["summary"] = "sandbox create failed"
        return res
    res["sandboxes"].append(sb.id)
    res["timings"]["sandbox_create_s"] = create_s
    base: dict = {}
    try:
        up = _upload_dir(sb, bundle, TESTS_REMOTE)
        res["timings"]["bundle_upload_s"] = up["seconds"]
        res["timings"]["bundle_bytes"] = up["bytes"]
        r = _sh(sb, f"test -d {shlex.quote(workdir)} && echo WORKDIR_OK; df -h / | tail -1; nproc; free -m | sed -n 2p", timeout=60)
        base["workdir_probe"] = (r["stdout"] or "")[-400:]
        if "WORKDIR_OK" not in (r["stdout"] or ""):
            reasons.append(f"workdir {workdir} does not exist in the snapshot")
        res["leak_check"] = leak_check(sb, workdir, ref_shas)
        if not res["leak_check"]["ok"]:
            reasons.append("reference commit reachable from the workspace (leak)")
        if adapter_required:
            base["adapter"] = run_adapter(sb, workdir, bundle, manifest, adapter_context)
        base.update(run_verifier(sb, workdir, timeout=run_timeout))
        if base.get("reward") is None:
            reasons.append("base run: verifier reported no reward")
        elif base["reward"] != 0.0:
            reasons.append(f"base run: reward {base['reward']} (expected 0)")
        if base.get("p2p_exit") not in (0, None) and not base.get("p2p_skipped"):
            reasons.append(f"base run: P2P failed (exit {base.get('p2p_exit')})")
        if base.get("p2p_skipped"):
            base["p2p_note"] = "no p2p in manifest"
        cls, ev = classify_failure(base.get("f2p_tail") or "")
        base["f2p_class"], base["f2p_class_evidence"] = cls, ev
        if base.get("f2p_exit") == 0:
            base["f2p_class"] = "passed_at_base"
    except Exception as e:  # noqa: BLE001
        reasons.append(f"base run crashed: {type(e).__name__}: {str(e)[:300]}")
    finally:
        if keep_failed and reasons:
            base["sandbox_kept"] = sb.id
        else:
            cleanup(sb)
    res["base"] = base

    # ---- ref x N --------------------------------------------------------------
    res["ref"] = []
    for i in range(max(1, ref_runs)):
        try:
            sb, create_s = _new_sandbox(d, snapshot, labels, no_network, f"validate-ref{i + 1}")
        except Exception as e:  # noqa: BLE001
            reasons.append(f"ref run {i + 1}: sandbox create failed: {str(e)[:200]}")
            break
        res["sandboxes"].append(sb.id)
        rr: dict = {"run": i + 1, "sandbox_create_s": create_s}
        try:
            _upload_dir(sb, bundle, TESTS_REMOTE)
            ap = apply_oracle(sb, workdir, oracle)
            rr["patch_applied"], rr["patch_tail"] = ap["applied"], ap["tail"]
            if not ap["applied"]:
                reasons.append(f"ref run {i + 1}: oracle patch did not apply")
            if adapter_required:
                # Static result first (unadapted tests), then the adapted one.  Two runs
                # on the same tree: verify.py overlays test files, so the static run
                # leaves the prewritten tests in place and the adapter then overwrites
                # them with its rewrite before the second overlay.
                static = run_verifier(sb, workdir, timeout=run_timeout)
                rr["static"] = {k: static.get(k) for k in ("reward", "p2p_exit", "f2p_exit", "verify_s", "f2p_tail")}
                rr["adapter"] = run_adapter(sb, workdir, bundle, manifest, adapter_context)
            rr.update(run_verifier(sb, workdir, timeout=run_timeout))
            if rr.get("reward") != 1.0:
                reasons.append(f"ref run {i + 1}: reward {rr.get('reward')} (expected 1)")
            if rr.get("p2p_exit") not in (0, None) and not rr.get("p2p_skipped"):
                reasons.append(f"ref run {i + 1}: P2P failed (exit {rr.get('p2p_exit')})")
        except Exception as e:  # noqa: BLE001
            reasons.append(f"ref run {i + 1} crashed: {type(e).__name__}: {str(e)[:300]}")
        finally:
            if keep_failed and reasons:
                rr["sandbox_kept"] = sb.id
            else:
                cleanup(sb)
        res["ref"].append(rr)

    res["timings"]["base_verify_s"] = base.get("verify_s")
    res["timings"]["ref_verify_s"] = [r.get("verify_s") for r in res["ref"]]
    res["valid"] = not reasons
    b = res.get("base") or {}
    res["summary"] = (f"{'VALID' if res['valid'] else 'INVALID'}: base reward={b.get('reward')} "
                      f"(p2p exit {b.get('p2p_exit')}, f2p exit {b.get('f2p_exit')}, f2p class {b.get('f2p_class')}); "
                      f"ref rewards={[r.get('reward') for r in res['ref']]}; leak ok={res.get('leak_check', {}).get('ok')}"
                      + (f"; {len(reasons)} reason(s): " + " | ".join(reasons[:4]) if reasons else ""))
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["validate_s"] = round(time.time() - t_start, 1)
    return res
