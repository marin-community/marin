#!/usr/bin/env python3
"""Test adapter: rewrite prewritten tests against the API as the candidate tree wrote it.

The design this stubs (and must not drift from): a small finetuned model transforms the
prewritten fail-to-pass tests into tests of the SAME logic against the solver's API surface,
and the rewritten tests are then run mechanically.  Until that model exists, GLM-5.3 through
the relay plays the adapter.  It is not a rubric judge: it never decides pass/fail, it only
emits test source, and the mechanical run decides.

Runs on the HOST (the validator or the grading harness), not inside the sandbox -- a Daytona
sandbox cannot reach the in-cluster relay.  Inputs are plain directories:

    adapt_bundle(bundle, workspace_view, outdir, manifest, extra_context="") -> dict

`workspace_view` holds the candidate tree's files matching manifest.adapter.context_globs.
For each overlay item flagged `adapt: true` (default: every overlay item when
adapter.required), the prewritten test plus the context files go to the model with the
`adapter.surface` notes (the proposal's grader_api_surface), and the model returns the
adapted file, written to <outdir>/<overlay.from>.  The record carries the unified-diff size
against the prewritten test so a rewrite that changes more than names is visible.

Endpoint: ENVGEN_ADAPTER_BASE_URL (falls back to GLM_BASE_URL) + ENVGEN_ADAPTER_TOKEN
(falls back to GLM_API_TOKEN); model ENVGEN_ADAPTER_MODEL (default glm-5.3).
"""
from __future__ import annotations

import difflib
import json
import os
import pathlib
import re
import time
import urllib.request

MAX_CONTEXT_CHARS = 90_000
MAX_TEST_CHARS = 60_000

SYSTEM = """You adapt an existing automated test file so that it exercises the SAME behaviour against the API exactly as it is implemented in the provided source tree.
Rules, all binding:
- Preserve every assertion and the logic of every test case. Do not delete, skip, weaken, loosen, or add tests. Do not change expected values.
- Change only what is necessary to bind to the tree's API as written: names, import paths, call signatures, argument order, constructor shapes, method vs function, return unpacking, types.
- If the tree does not expose the behaviour at all, return the test UNCHANGED. Never invent an API that is not in the tree.
- Output ONLY the complete adapted file content. No prose, no code fences, no explanations."""


def _endpoint() -> tuple[str, str, str]:
    base = os.environ.get("ENVGEN_ADAPTER_BASE_URL") or os.environ.get("GLM_BASE_URL") or ""
    tok = os.environ.get("ENVGEN_ADAPTER_TOKEN") or os.environ.get("GLM_API_TOKEN") or ""
    model = os.environ.get("ENVGEN_ADAPTER_MODEL", "glm-5.3")
    return base.rstrip("/"), tok, model


def _chat(base: str, tok: str, model: str, system: str, user: str, timeout: int = 900) -> tuple[str, dict]:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0, "max_tokens": 32768, "reasoning_effort": os.environ.get("ENVGEN_ADAPTER_EFFORT", "high"),
            "stream": False}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        doc = json.loads(r.read().decode("utf-8", "replace"))
    text = (doc.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return text, doc.get("usage") or {}


def _strip_fence(t: str) -> str:
    s = t.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s + ("\n" if not s.endswith("\n") else "")


def _collect_context(ws: pathlib.Path, globs: list[str]) -> str:
    parts: list[str] = []
    total = 0
    files = sorted({p for g in globs for p in ws.glob(g) if p.is_file()})
    for p in files:
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if total + len(txt) > MAX_CONTEXT_CHARS:
            txt = txt[: max(0, MAX_CONTEXT_CHARS - total)] + "\n... [truncated]\n"
        parts.append(f"===== {p.relative_to(ws)} =====\n{txt}")
        total += len(txt)
        if total >= MAX_CONTEXT_CHARS:
            break
    return "\n".join(parts)


def adapt_bundle(bundle: pathlib.Path, ws: pathlib.Path, outdir: pathlib.Path, manifest: dict,
                 extra_context: str = "") -> dict:
    base, tok, model = _endpoint()
    ad = manifest.get("adapter") or {}
    rec: dict = {"ok": True, "model": model, "endpoint_set": bool(base), "files": [], "written": [], "calls": 0,
                 "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "seconds": 0.0}
    if not base or not tok:
        rec["ok"] = False
        rec["error"] = "adapter endpoint or token not configured (ENVGEN_ADAPTER_BASE_URL / GLM_BASE_URL)"
        return rec
    ctx = _collect_context(ws, list(ad.get("context_globs") or []))
    if extra_context:
        try:
            ctx = pathlib.Path(extra_context).read_text(errors="replace")[:MAX_CONTEXT_CHARS] + "\n" + ctx
        except OSError:
            pass
    surface = ad.get("surface") or []
    notes = ad.get("notes") or ""
    t0 = time.time()
    for item in manifest.get("overlay") or []:
        if not item.get("adapt", True):
            continue
        src = bundle / "envgen" / item["from"]
        if not src.is_file():
            rec["files"].append({"from": item["from"], "error": "missing in bundle"})
            continue
        original = src.read_text(errors="replace")
        user = (f"Target path in the tree: {item['to']}\n\n"
                f"What the task statement told the solver about the API surface (the solver may have chosen different names/shapes; adapt to what the tree actually has):\n"
                + "\n".join(f"- {s}" for s in surface) + ("\n\nAdditional notes:\n" + notes if notes else "")
                + "\n\n===== SOURCE TREE (candidate's implementation) =====\n" + ctx
                + f"\n\n===== PREWRITTEN TEST ({item['from']}) =====\n" + original[:MAX_TEST_CHARS]
                + "\n\n===== END =====\nReturn the complete adapted test file now.")
        f: dict = {"from": item["from"], "to": item["to"], "original_chars": len(original)}
        try:
            text, usage = _chat(base, tok, model, SYSTEM, user)
            rec["calls"] += 1
            rec["usage"]["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            rec["usage"]["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            adapted = _strip_fence(text)
            if len(adapted.strip()) < 20:
                raise ValueError("adapter returned an empty file")
            diff = list(difflib.unified_diff(original.splitlines(), adapted.splitlines(), lineterm="", n=0))
            changed = sum(1 for ln in diff if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))
            f.update({"adapted_chars": len(adapted), "changed_lines": changed, "diff_head": "\n".join(diff[:40])[:3000],
                      "unchanged": adapted.strip() == original.strip()})
            # Guard against the one failure a rewrite step invites: fewer test cases than before.
            def _count(t: str) -> int:
                return len(re.findall(r"^\s*(?:func Test|def test_|it\(|test\(|@Test|#\[test\]|TEST\(|TEST_F\(|describe\()", t, re.M))
            f["tests_before"], f["tests_after"] = _count(original), _count(adapted)
            if f["tests_after"] < f["tests_before"]:
                f["warning"] = "adapted file has fewer test cases than the prewritten one; keeping the prewritten file"
                adapted = original
            out = outdir / item["from"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(adapted)
            rec["written"].append(item["from"])
        except Exception as e:  # noqa: BLE001
            f["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec["ok"] = False
        rec["files"].append(f)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = pathlib.Path(a.bundle)
    m = json.loads((b / "envgen" / "manifest.json").read_text())
    print(json.dumps(adapt_bundle(b, pathlib.Path(a.workspace), pathlib.Path(a.out), m), indent=1))
