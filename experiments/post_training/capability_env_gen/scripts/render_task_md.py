"""Render an exported TaskSpec as docs/exports/<slug>/task.md.

Input is a directory pulled with ``pull_snapshot.py`` containing
``validated/<item>/`` (the export) plus ``items/<item>/status.json`` and
``items/<item>/contract/accepted.json``.  Every value in the output is read from
those files or the catalog; nothing is paraphrased or inferred.

    uv run --frozen scripts/render_task_md.py <pulled-dir> [<pulled-dir> ...] --out docs/exports --run <label>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _find_capability(node, cid):
    if isinstance(node, dict):
        if cid in (node.get("capability_id"), node.get("id")) and (
            "name" in node or "title" in node or "outcome" in node
        ):
            return node
        for value in node.values():
            found = _find_capability(value, cid)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_capability(value, cid)
            if found:
                return found
    return None


def _subject(catalog, cid):
    """Return the curriculum/subject entry whose subtree contains the capability."""
    for curriculum in catalog.get("curricula", []) if isinstance(catalog, dict) else []:
        if _find_capability(curriculum, cid):
            return curriculum
    return None


def _code(value) -> str:
    return f"`{value}`"


def _json_inline(value) -> str:
    return "`" + json.dumps(value, separators=(",", ":"), sort_keys=True) + "`"


def _human_bytes(n: int) -> str:
    return f"{n:,}" if n < 10_000 else f"{n / 1024:.1f} K"


def render(pulled: Path, run_label: str) -> tuple[str, str]:
    export = next(p for p in (pulled / "validated").iterdir() if p.is_dir())
    item = export.name
    status = json.loads((pulled / "items" / item / "status.json").read_text())
    accepted = json.loads((pulled / "items" / item / "contract" / "accepted.json").read_text())
    proposal = accepted.get("proposal", {})
    spec = json.loads((export / "specification.json").read_text())
    binding = json.loads((export / "binding.json").read_text())
    renderings = json.loads((export / "renderings.json").read_text())
    task_toml = (export / "task.toml").read_text().strip()
    instruction = (export / "instruction.md").read_text()

    catalog_path = ROOT / "new_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    cid = proposal.get("capability_id", "")
    cap = _find_capability(catalog, cid) or {}
    subject = _subject(catalog, cid) or {}

    task_id = spec["id"]
    slug = task_id.split("/", 1)[-1].replace("/", "-")
    tc = status.get("taskcompendium") or {}
    meta = spec.get("metadata", {})
    source = meta.get("source", {})
    steps = spec.get("steps", [])
    req = spec.get("requirements", {})
    state = req.get("state", {})

    out: list[str] = []
    w = out.append
    w(f"# Task `{task_id}`\n")
    w(f"**{proposal.get('title', item)}.**  The exported, validated deliverable of run `{run_label}`.\n")
    w("This document is a rendering of the exported TaskSpec.  The authoritative bytes are "
      f"`validated/{item}/` ({sum(1 for p in export.rglob('*') if p.is_file())} files).\n")
    w("---\n\n## Identity\n")
    w("| field | value |\n| --- | --- |")
    w(f"| `id` | {_code(task_id)} |")
    w(f"| `schema_version` | {_code(spec.get('schema_version'))} |")
    w(f"| `difficulty` | {spec.get('difficulty')} |")
    w(f"| `success_policy` | {_code(spec.get('success_policy'))} |")
    w(f"| `metadata.task_shape` | {_code(meta.get('task_shape'))} |")
    w(f"| steps | {len(steps)} |")
    if tc.get("specification_sha256"):
        w(f"| `specification_sha256` | {_code(tc['specification_sha256'])} |")
    w("")
    if spec.get("coverage_tags"):
        w("`coverage_tags`: " + ", ".join(_code(t) for t in spec["coverage_tags"]) + "\n")
    w("### Provenance\n")
    w("| field | value |\n| --- | --- |")
    for key in ("dataset", "revision", "row", "importer_revision"):
        if key in source:
            w(f"| `source.{key}` | {_code(source[key])} |")
    w(f"| catalog | `new_catalog.json` @ {_code(catalog.get('catalog_version'))} |")
    cap_name = cap.get("name") or cap.get("title") or ""
    w(f"| capability | {_code(cid)}" + (f" — *{cap_name}*" if cap_name else "") + " |")
    cur = subject.get("curriculum", subject)
    subj_name = (f"{cur.get('subject_id')} — {cur.get('subject_name')}" if cur.get("subject_name")
                 else subject.get("name") or subject.get("title"))
    if subj_name:
        w(f"| subject | {subj_name} |")
    w(f"| proposal slot | {proposal.get('slot')} |")
    w(f"| task family | {proposal.get('task_family', '')} |")
    w("")
    if cap.get("outcome"):
        w(f"The capability's stated outcome: *\"{cap['outcome']}\"*")
        if cap.get("excludes"):
            ex = cap["excludes"]
            w("Declared **excludes**: " + ("; ".join(ex) if isinstance(ex, list) else str(ex)) + ".")
        w("")
    w("---\n\n## Environment and interface\n")
    w("| field | value |\n| --- | --- |")
    w(f"| proposed environment | {_code(proposal.get('environment'))} |")
    w(f"| `requirements.state.image` | {_json_inline(state.get('image'))} |")
    w(f"| `requirements.state.workdir` | {_code(state.get('workdir'))} |")
    w(f"| `requirements.state.setup_commands` | {_json_inline(state.get('setup_commands', []))} |")
    if state.get("additional_directories"):
        w(f"| `requirements.state.additional_directories` | {_json_inline(state['additional_directories'])} |")
    w(f"| `requirements.capabilities` | {_json_inline(req.get('capabilities', []))} |")
    w(f"| `requirements.action_interfaces` | {_json_inline(req.get('action_interfaces', []))} |")
    w(f"| `binding.json` | {_json_inline(binding)} |")
    for r in renderings if isinstance(renderings, list) else [renderings]:
        w(f"| rendering | {_json_inline(r)} |")
    for i, step in enumerate(steps):
        w(f"| step {i} `context_requirement` | {_code(step.get('context_requirement'))} |")
        w(f"| step {i} `answer_requirements` | {_json_inline(step.get('answer_requirements'))} |")
    w("")
    w("`task.toml`:\n\n```toml\n" + task_toml + "\n```\n")
    inputs = sorted(p for p in (export / "environment").rglob("*") if p.is_file()) if (export / "environment").exists() else []
    if inputs:
        w("Solver-visible input files shipped with the export:\n")
        w("| file | bytes |\n| --- | --- |")
        for p in inputs:
            w(f"| `{p.relative_to(export)}` | {p.stat().st_size:,} |")
        w("")
    w("---\n\n## The prompt\n")
    w(f"What follows is `instruction.md` in full, the complete solver-facing surface "
      f"({len(instruction.encode()) / 1024:.1f} KB, {instruction.count(chr(10)) + 1} lines).\n")
    w("---\n")
    w(instruction.rstrip() + "\n")
    w("---\n\n## How it is graded\n")
    w("The verifier is private evaluator material and never reaches the solver.\n")
    for i, step in enumerate(steps):
        v = step.get("verifier", {})
        rt = v.get("runtime", {})
        params = v.get("parameters", {})
        if len(steps) > 1:
            w(f"### Step {i}\n")
        w("| field | value |\n| --- | --- |")
        w(f"| `verifier.kind` | {_code(v.get('kind'))} |")
        if v.get("mode"):
            w(f"| `verifier.mode` | {_code(v.get('mode'))} |")
        for key in ("path", "output_path", "timeout", "args"):
            if key in params:
                w(f"| `parameters.{key}` | {_json_inline(params[key])} |")
        if rt:
            w(f"| runtime | {rt.get('kind')} {_code(rt.get('image'))}, workspace {_json_inline(rt.get('workspace'))}, timeout {rt.get('timeout')} s |")
        if v.get("implementation_revision"):
            w(f"| implementation revision | {_code(v['implementation_revision'])} |")
        w("")
    resources = spec.get("resources", [])
    if resources:
        w(f"Embedded resources ({len(resources)}):\n")
        w("| path | roles | lines |\n| --- | --- | --- |")
        for r in resources:
            content = r.get("content")
            text = content.get("text") if isinstance(content, dict) else None
            lines = (text.count("\n") + 1) if isinstance(text, str) else "—"
            w(f"| `{r.get('path')}` | {', '.join(r.get('roles', []))} | {lines} |")
        w("")
    w("---\n\n## Pipeline verdicts\n")
    w("| check | result |\n| --- | --- |")
    w(f"| item state | {_code(status.get('state'))} |")
    w(f"| runtime validated | {_code(status.get('runtime_validated'))} |")
    rd = status.get("repeated_diagnostics") or {}
    if rd:
        w(f"| repeated diagnostics | {_code(rd.get('state'))} |")
    qr = status.get("quality_review") or {}
    if qr:
        w(f"| quality review | {_code(qr.get('state'))} (artifact sha256 {_code(qr.get('artifact_sha256', '')[:16] + '…')}) |")
    budget = status.get("repair_budget")
    if budget:
        w(f"| repair budget | {_json_inline(budget)} |")
    w("")
    w("---\n\n## Files in the export\n")
    w("| file | bytes |\n| --- | --- |")
    for p in sorted(p for p in export.rglob("*") if p.is_file()):
        w(f"| `{p.relative_to(export)}` | {_human_bytes(p.stat().st_size)} |")
    w("")
    return slug, "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pulled", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "exports")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    for pulled in args.pulled:
        slug, text = render(pulled, args.run)
        dest = args.out / slug
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "task.md").write_text(text)
        print(f"{dest / 'task.md'}  ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
