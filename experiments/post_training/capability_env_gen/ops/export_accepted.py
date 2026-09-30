#!/usr/bin/env python3
"""Render task.md for newly quality_accepted HEALTHCARE items (hc1..hc4) into an output directory.

Idempotent: an item is skipped when any ``task.md`` under ``--out`` (or a ``--skip-dir``, by default
the live ``docs/exports``, read only) already names ``validated/<item>/``, or when the out-dir
ledger ``.export-ledger.json`` has it.  No build.md is written.

The renderer is ``scripts/render_task_md.py`` (unchanged); this tool only materialises the
directory shape it reads (``validated/<item>/``, ``items/<item>/status.json``,
``items/<item>/contract/accepted.json``) from the run's content-addressed S3 snapshot, verifying
every object against its sha256.

  cd /Users/k3sc0re/openathena/marin-construct   # with CW_KEY_ID/CW_KEY_SECRET exported
  PYTHONPATH=/Users/k3sc0re/openathena/capability_env_gen-dev/scripts \\
    uv run --frozen /Users/k3sc0re/openathena/capability_env_gen-dev/ops/export_accepted.py --out /tmp/exports [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conveyor as cv  # noqa: E402

DEV_ROOT = cv.OPS_DIR.parent
RENDERER = DEV_ROOT / "scripts" / "render_task_md.py"
LIVE_EXPORTS = cv.LIVE_TREE / "docs" / "exports"
HC_BASES = ("hc1", "hc2", "hc3", "hc4")
LEDGER_NAME = ".export-ledger.json"
RE_AUTHORITATIVE = re.compile(r"`validated/([^/`]+)/`")


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_task_md", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exported_items(dirs: list[Path]) -> dict[str, Path]:
    """Item names already rendered somewhere (task.md names its ``validated/<item>/`` directory)."""
    found: dict[str, Path] = {}
    for root in dirs:
        if not root.is_dir():
            continue
        for task in root.glob("*/task.md"):
            try:
                head = task.read_text(errors="replace")[:4000]
            except OSError:
                continue
            for match in RE_AUTHORITATIVE.finditer(head):
                found.setdefault(match.group(1), task)
        ledger = cv._read_json_file(root / LEDGER_NAME, {}) or {}
        for item, rec in (ledger.get("items") or {}).items():
            found.setdefault(item, root / rec.get("slug", "?") / "task.md")
    return found


def member_files(raw_manifest: bytes, item: str) -> dict[str, str]:
    """validated/<item>/** (or items/<item>/harbor/** as a fallback) + status + contract, rel -> sha."""
    text_item = re.escape(item.encode())
    out: dict[str, str] = {}
    for m in re.finditer(rb'"(validated/' + text_item + rb'/[^"\n]+)":\s*"([0-9a-f]{64})"', raw_manifest):
        out[m.group(1).decode()] = m.group(2).decode()
    if not out:
        # synthesize() exports validated/<item> as a copy of items/<item>/harbor.
        for m in re.finditer(rb'"items/' + text_item + rb'/harbor/([^"\n]+)":\s*"([0-9a-f]{64})"', raw_manifest):
            out[f"validated/{item}/{m.group(1).decode()}"] = m.group(2).decode()
    for rel in (f"items/{item}/status.json", f"items/{item}/contract/accepted.json"):
        m = re.search(rb'"' + re.escape(rel.encode()) + rb'":\s*"([0-9a-f]{64})"', raw_manifest)
        if m:
            out[rel] = m.group(1).decode()
    return out


def materialise(files: dict[str, str], fetch: Callable[[str], bytes], dest: Path) -> None:
    for rel, sha in files.items():
        path = dest / rel
        if ".." in Path(rel).parts or Path(rel).is_absolute():
            raise ValueError(f"unsafe member path {rel!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(sha))


def export(store: Any, out: Path, *, bases: tuple[str, ...] = HC_BASES, skip_dirs: list[Path] | None = None,
           dry_run: bool = False, run_label: str = "/muchanem/{run_name} ({run})",
           log: Callable[[str], None] = print, renderer: Any = None) -> dict[str, Any]:
    """Export every accepted, not-yet-exported item of ``bases``.  ``store`` is a conveyor.S3Store."""
    renderer = renderer or load_renderer()
    out.mkdir(parents=True, exist_ok=True)
    skip_dirs = [out, *(skip_dirs or [])]
    already = exported_items(skip_dirs)
    ledger_path = out / LEDGER_NAME
    ledger = cv._read_json_file(ledger_path, {}) or {}
    ledger.setdefault("items", {})
    summary: dict[str, Any] = {"exported": [], "skipped": [], "errors": [], "accepted_seen": 0}
    for base in bases:
        index = store.manifest_index(base)
        if index is None:
            summary["errors"].append(f"{base}: no manifest")
            log(f"ERROR {base}: no manifest")
            continue
        submission = {}
        sub_sha = (index.get("run_files") or {}).get("submission.json")
        if sub_sha:
            submission = store.get_object(base, sub_sha).get("data") or {}
        accepted = []
        for name, rec in sorted((index.get("items") or {}).items()):
            if not rec.get("status"):
                continue
            status = store.get_object(base, rec["status"]).get("data") or {}
            if status.get("state") == "quality_accepted":
                accepted.append(name)
        summary["accepted_seen"] += len(accepted)
        todo = [name for name in accepted if name not in already]
        for name in accepted:
            if name in already:
                summary["skipped"].append({"run": base, "item": name, "where": str(already[name])})
        if not todo:
            log(f"{base}: {len(accepted)} accepted, all already exported")
            continue
        raw = store.raw_manifest(base)
        label = run_label.format(run_name=submission.get("run_name") or f"cap-construct-003-{base}", run=cv.RUN, base=base)
        for name in todo:
            files = member_files(raw, name)
            missing = [r for r in (f"items/{name}/status.json", f"items/{name}/contract/accepted.json") if r not in files]
            if missing or not any(r.startswith("validated/") for r in files):
                msg = f"{base}/{name}: snapshot lacks {missing or 'validated/ export files'}"
                summary["errors"].append(msg)
                log(f"ERROR {msg}")
                continue
            if dry_run:
                log(f"WOULD EXPORT {base}/{name} ({len(files)} files)")
                summary["exported"].append({"run": base, "item": name, "dry_run": True})
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="export-accepted-") as tmp:
                    pulled = Path(tmp)
                    materialise(files, lambda sha, base=base: store.get_bytes(base, sha), pulled)
                    slug, text = renderer.render(pulled, label)
            except Exception as error:  # noqa: BLE001 - one bad item must not stop the rest
                msg = f"{base}/{name}: render failed: {type(error).__name__}: {error}"
                summary["errors"].append(msg)
                log(f"ERROR {msg}")
                continue
            dest = out / slug
            task_md = dest / "task.md"
            if task_md.exists():
                msg = f"{base}/{name}: {task_md} already exists for a different item; not overwriting"
                summary["errors"].append(msg)
                log(f"ERROR {msg}")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            task_md.write_text(text)
            ledger["items"][name] = {"run": base, "slug": slug, "status_sha": (index["items"][name] or {}).get("status"),
                                     "exported_at": cv.iso(datetime.now(UTC)), "label": label}
            cv._atomic_write(ledger_path, json.dumps(ledger, indent=1))
            already[name] = task_md
            summary["exported"].append({"run": base, "item": name, "slug": slug, "path": str(task_md), "bytes": len(text)})
            log(f"EXPORTED {base}/{name} -> {task_md} ({len(text):,} bytes)")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="output directory (one <slug>/task.md per item)")
    parser.add_argument("--bases", default=",".join(HC_BASES), help="comma-separated run bases (default hc1..hc4)")
    parser.add_argument("--skip-dir", action="append", type=Path, default=None,
                        help=f"also treat items rendered here as exported (default: {LIVE_EXPORTS}, read only)")
    parser.add_argument("--no-default-skip", action="store_true", help="do not consult the live docs/exports")
    parser.add_argument("--run-label", default="/muchanem/{run_name} ({run})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    for forbidden in (LIVE_EXPORTS.resolve(), (DEV_ROOT / "docs" / "exports").resolve()):
        if out == forbidden or forbidden in out.parents:
            parser.error(f"refusing to write under {forbidden}")
    skip = list(args.skip_dir or [])
    if not args.no_default_skip:
        skip.append(LIVE_EXPORTS)
    store = cv.S3Store()
    try:
        store.connect()
    except cv.ReadError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    summary = export(store, out, bases=tuple(b for b in args.bases.split(",") if b), skip_dirs=skip,
                     dry_run=args.dry_run, run_label=args.run_label)
    print(f"SUMMARY accepted_seen={summary['accepted_seen']} exported={len(summary['exported'])} "
          f"skipped={len(summary['skipped'])} errors={len(summary['errors'])}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
