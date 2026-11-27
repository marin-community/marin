# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the n=25 proposal run into the viewer's Postgres tables through Marina's SQL API.

Reads the run tree (one folder per capability), the capability catalog and the per-file usage summaries
``slot_usage.py`` writes, then upserts one ``capabilities`` row per catalog capability and one
``proposals`` row per planned slot. Request and completions files are never loaded; rows keep their
paths and the usage summary. Each row records ``source_key``, a digest of the ETags and sizes of the
files it was built from, so a rerun reads and uploads only rows whose sources changed. Rerunning is
always safe: rows are upserted by capability and slot.

    uv run python -m infra.marina.apps.taskforge_proposals.ingest --app-id UUID [--url ORIGIN]
"""

import hashlib
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import click
import fsspec
import requests
from marina.client import DEFAULT_MARINA_URL, MarinaRequestError, applet_query, marina_token_provider
from rigging.filesystem.factory import filesystem
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from .server.proposal import body_sections, first_sentence
from .server.schema import CAPABILITY_COLUMNS, PROPOSAL_COLUMNS
from .slot_usage import DEFAULT_OUTPUT as DEFAULT_USAGE
from .slot_usage import DEFAULT_RUN_ROOT, USAGE_FIELDS, file_key, list_tree

INGEST_VERSION = "1"
DEFAULT_CATALOG = (
    "s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/curriculum.json"
)
MAX_BATCH_BYTES = 1_000_000
KEY_PAGE = 10_000
CUT_OFF_TEXT = "cut off"


def names(columns: str) -> list[str]:
    return [column.strip().split()[0] for column in columns.split(",")]


def upsert_sql(table: str, columns: str, key: str) -> str:
    cols = names(columns)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key.split(", "))
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} "
        f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({columns}) "
        f"ON CONFLICT ({key}) DO UPDATE SET {updates}, loaded_at = now()"
    )


def batches(rows: list[dict[str, Any]]) -> list[str]:
    """Row arrays whose encoding, escaped once more as a JSON string, stays under ``MAX_BATCH_BYTES``."""
    out, batch, size = [], [], 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        length = len(json.dumps(encoded).encode()) + 1
        if length + 2 > MAX_BATCH_BYTES:
            raise ValueError(f"row {row.get('id') or (row['capability_id'], row['slot'])} exceeds the batch budget")
        if batch and size + length > MAX_BATCH_BYTES:
            out.append("[" + ",".join(batch) + "]")
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        out.append("[" + ",".join(batch) + "]")
    return out


def without_nul(value: Any) -> tuple[Any, int]:
    """``value`` with each NUL character replaced by U+FFFD, which Postgres text and JSONB cannot hold.

    Returns the cleaned value and how many characters were replaced.
    """
    if isinstance(value, str):
        count = value.count("\x00")
        return (value.replace("\x00", "\ufffd"), count) if count else (value, 0)
    if isinstance(value, dict):
        out, total = {}, 0
        for key, item in value.items():
            out[key], count = without_nul(item)
            total += count
        return out, total
    if isinstance(value, list):
        items = [without_nul(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    return value, 0


def digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps([INGEST_VERSION, *parts], sort_keys=True, default=str).encode()).hexdigest()[:32]


def catalog_capabilities(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every capability section with its subject and the names of the groups above it."""
    out = {}
    for entry in catalog["curricula"]:
        curriculum = entry["curriculum"]
        sections = {s["id"]: s for s in curriculum["sections"]}
        for section in curriculum["sections"]:
            if section["kind"] != "capability":
                continue
            chain, parent = [], section.get("parent_id")
            while parent and parent in sections:
                chain.append(sections[parent].get("name") or parent)
                parent = sections[parent].get("parent_id")
            chain.reverse()
            out[section["id"]] = {
                "name": section.get("name"),
                "outcome": section.get("outcome"),
                "subject_id": curriculum["subject_id"],
                "domain": curriculum["subject_name"],
                "area": chain[0] if chain else None,
                "section_path": " › ".join(chain),  # noqa: RUF001 (a display separator)
                "entry": section,
            }
    return out


def read_usage(fs: fsspec.AbstractFileSystem, usage_root: str) -> dict[str, dict[str, Any]]:
    """Usage summaries by file key; a later part replaces an earlier one for the same key."""
    out = {}
    for path in sorted(fs.glob(prefix_join(usage_root, "*.jsonl"))):
        for line in fs.cat(path).decode().splitlines():
            if line.strip():
                record = json.loads(line)
                out[record["key"]] = record
    return out


class Uploader:
    """Byte-bounded upserts through per-thread SQL clients."""

    def __init__(self, url: str, app_id: str, threads: int):
        self.url, self.app_id, self.threads = url, app_id, threads
        self.local = threading.local()
        self.provider = marina_token_provider()
        self.requests = 0

    def query(self, sql: str, params: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self.local, "query"):
            self.local.query = applet_query(
                self.url, self.app_id, session=requests.Session(), token_provider=self.provider
            )
        for attempt in range(5):
            try:
                self.requests += 1
                return self.local.query(sql, params)
            except (requests.RequestException, MarinaRequestError) as error:
                # Transport errors and 5xx are retried (the upsert is idempotent); a 4xx is the request's fault.
                if attempt == 4 or (isinstance(error, MarinaRequestError) and error.status_code < 500):
                    raise
                print(f"retrying after {type(error).__name__}: {str(error)[:200]}", flush=True)
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def upsert(self, table: str, columns: str, key: str, rows: list[dict[str, Any]]) -> int:
        sql = upsert_sql(table, columns, key)
        bodies = batches(rows)
        done = 0

        def send(body: str) -> int:
            nonlocal done
            count = self.query(sql, {"rows": body})["row_count"]
            done += 1
            if done % 50 == 0:
                print(f"  {table}: {done}/{len(bodies)} batches", flush=True)
            return count

        with ThreadPoolExecutor(self.threads) as pool:
            return sum(pool.map(send, bodies))

    def keys(self, sql: str) -> dict[Any, str]:
        out, offset = {}, 0
        while True:
            page = self.query(f"{sql} LIMIT {KEY_PAGE} OFFSET {offset}", {})["rows"]
            for row in page:
                out[row["k"]] = row["source_key"]
            if len(page) < KEY_PAGE:
                return out
            offset += KEY_PAGE


@click.command(help=__doc__)
@click.option("--app-id", required=True, help="The viewer's Marina app UUID.")
@click.option("--url", default=DEFAULT_MARINA_URL, show_default=True, help="Marina origin (a local preview's origin).")
@click.option("--run-root", default=DEFAULT_RUN_ROOT, show_default=True)
@click.option("--usage-root", default=DEFAULT_USAGE, show_default=True)
@click.option("--catalog", "catalog_url", default=DEFAULT_CATALOG, show_default=True)
@click.option("--read-threads", default=48, show_default=True)
@click.option("--upload-threads", default=4, show_default=True)
@click.option("--limit", type=int, default=None, help="Load only the first N capabilities (a preview).")
def main(
    app_id: str,
    url: str,
    run_root: str,
    usage_root: str,
    catalog_url: str,
    read_threads: int,
    upload_threads: int,
    limit: int | None,
) -> None:
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    configure_coreweave_s3()
    fs = filesystem("s3")
    uploader = Uploader(url, app_id, upload_threads)

    catalog_doc = json.loads(fs.cat(catalog_url))
    catalog = catalog_capabilities(catalog_doc)
    listing = list_tree(fs, run_root)
    files: dict[str, dict[str, dict[str, Any]]] = {}
    for path, info in listing.items():
        rel = path.split("/")
        if len(rel) == 2 and rel[0] in catalog:
            files.setdefault(rel[0], {})[rel[1]] = info
    if not files:
        raise SystemExit(f"no capability folders under {run_root}; check the CoreWeave keys and the root")
    usage = read_usage(fs, usage_root)
    if not usage:
        raise SystemExit(f"no usage summaries under {usage_root}; run slot_usage.py first")
    print(
        f"catalog {len(catalog)} capabilities, {len(files)} folders, {len(listing)} files, "
        f"{len(usage)} usage summaries ({time.monotonic() - started:.0f}s)",
        flush=True,
    )
    ids = sorted(catalog)[:limit] if limit else sorted(catalog)

    existing_caps = uploader.keys("SELECT id AS k, source_key FROM capabilities ORDER BY id")
    existing_slots: dict[str, dict[int, str]] = {}
    for composite, key in uploader.keys(
        "SELECT capability_id || '/' || slot AS k, source_key FROM proposals ORDER BY capability_id, slot"
    ).items():
        capability_id, slot = composite.rsplit("/", 1)
        existing_slots.setdefault(capability_id, {})[int(slot)] = key
    stored_count = sum(len(v) for v in existing_slots.values())
    print(f"stored: {len(existing_caps)} capabilities, {stored_count} proposals", flush=True)
    problems: Counter[str] = Counter()

    def usage_of(info: dict[str, Any] | None) -> dict[str, Any] | None:
        key = file_key(info)
        if key is None:
            return None
        record = usage.get(key)
        if record is None:
            problems["completions file without a current usage summary"] += 1
        return record

    def stored_path(cap: str, name: str) -> str | None:
        """The URL of a run file the viewer links to but does not load, when the file exists."""
        return prefix_join(run_root, f"{cap}/{name}") if name in files.get(cap, {}) else None

    def cat_json(info: dict[str, Any] | None) -> Any:
        return None if info is None else json.loads(fs.cat(info["name"]))

    def capability_rows(cap: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        meta, folder = catalog[cap], files.get(cap, {})
        plan_info, result_info, failure_info = (folder.get(n) for n in ("plan.json", "result.json", "failure.json"))
        plan_usage_info = folder.get("plan.completions.json")
        slot_numbers = sorted({int(n[5:7]) for n in folder if n.startswith("slot-")})
        slot_sources: dict[int, dict[str, Any]] = {}
        for n in slot_numbers:
            prefix = f"slot-{n:02d}."
            slot_sources[n] = {
                kind: file_key(folder.get(prefix + kind))
                for kind in ("proposal.json", "verdict.json", "failure.txt", "completions.json")
            }
            slot_sources[n]["usage"] = slot_sources[n]["completions.json"] in usage
            # A rerun of a slot rewrites its request first; a proposal older than the request is an earlier
            # attempt's leftover, not this slot's outcome (the rerun ended in failure.txt instead).
            proposal_info, request_info = folder.get(prefix + "proposal.json"), folder.get(prefix + "request.json")
            if proposal_info and request_info and request_info["LastModified"] > proposal_info["LastModified"]:
                slot_sources[n]["superseded"] = True
        cap_key = digest(
            [file_key(i) for i in (plan_info, result_info, failure_info, plan_usage_info)],
            (usage_of(plan_usage_info) or {}).get("key"),
            {str(n): s for n, s in slot_sources.items()},
            catalog_doc["catalog_version"],
        )
        plan_key = file_key(plan_info)

        def slot_key(n: int) -> str:
            sources = slot_sources.get(n)
            used = usage.get(sources["completions.json"]) if sources else None
            return digest(sources, plan_key, (used or {}).get("key"), meta["subject_id"])

        stored_slots = existing_slots.get(cap, {})
        if (
            existing_caps.get(cap) == cap_key
            and stored_slots
            and set(slot_numbers) <= set(stored_slots)
            and all(stored_slots[n] == slot_key(n) for n in stored_slots)
        ):
            return None, []
        plan, result, failure = cat_json(plan_info), cat_json(result_info), cat_json(failure_info)
        planned = {s["slot"]: s for s in (plan or {}).get("slots", [])}
        slot_results = (result or {}).get("slots", {})
        slots = sorted(set(planned) | set(slot_numbers))
        proposal_rows = []
        counts: Counter[str] = Counter()
        slot_usage_total = Counter()
        for n in slots:
            prefix = f"slot-{n:02d}."
            plan_slot = planned.get(n, {})
            completions_info = folder.get(prefix + "completions.json")
            slot_usage = usage_of(completions_info)
            for name in USAGE_FIELDS:
                slot_usage_total[name] += (slot_usage or {}).get(name) or 0
            key = slot_key(n)
            proposal_info, verdict_info, failure_txt = (
                folder.get(prefix + k) for k in ("proposal.json", "verdict.json", "failure.txt")
            )
            failure_text = fs.cat(failure_txt["name"]).decode().strip() if failure_txt else None
            slot_result = slot_results.get(str(n))
            cut_off = CUT_OFF_TEXT in (failure_text or "") or CUT_OFF_TEXT in str(slot_result or "")
            superseded = bool((slot_sources.get(n) or {}).get("superseded"))
            current = proposal_info is not None and not superseded
            status = "proposed" if current else "cut_off" if cut_off else "failed" if failure_text else "pending"
            counts[status] += 1
            if stored_slots.get(n) == key:
                continue
            proposal, verdict = cat_json(proposal_info), cat_json(verdict_info)
            header = ((proposal or {}).get("header") or {}) if current else {}
            body = ((proposal or {}).get("body") or "") if current else ""
            sections = {s["heading"]: s["text"] for s in body_sections(body)}
            verdict_decision = (verdict or {}).get("decision") if current else None
            proposal_rows.append(
                {
                    "capability_id": cap,
                    "slot": n,
                    "subject_id": meta["subject_id"],
                    "domain": meta["domain"],
                    "area": meta["area"],
                    "status": status,
                    "title": plan_slot.get("title"),
                    "workflow": plan_slot.get("workflow"),
                    "challenge": plan_slot.get("distinctive_challenge"),
                    "summary": first_sentence(sections.get("Task", "")) or None,
                    "plan_status": plan_slot.get("status"),
                    "planned_verification": plan_slot.get("verification"),
                    "planned_environment": plan_slot.get("environment"),
                    "verification": header.get("verification") or plan_slot.get("verification"),
                    "environment": header.get("environment") or plan_slot.get("environment"),
                    "solver_machine": header.get("solver_machine") or plan_slot.get("solver_machine"),
                    "grounding": header.get("grounding"),
                    "research_need": header.get("research_need") or plan_slot.get("research_need"),
                    "verdict": verdict_decision,
                    "reason": ("; ".join((verdict or {}).get("reasons") or []) or None) if current else None,
                    "superseded": superseded,
                    "cut_off": cut_off,
                    "failure": failure_text,
                    "completions": (slot_usage or {}).get("completions"),
                    **{name: (slot_usage or {}).get(name) for name in USAGE_FIELDS},
                    "finish_reasons": (slot_usage or {}).get("finish_reasons"),
                    "wall_time": (slot_usage or {}).get("wall_time"),
                    "body_chars": len(body) if current else None,
                    "request_path": stored_path(cap, prefix + "request.json"),
                    "completions_path": stored_path(cap, prefix + "completions.json"),
                    "completions_bytes": completions_info["size"] if completions_info else None,
                    "proposal": proposal,
                    "verdict_record": verdict,
                    "source_key": key,
                }
            )
        plan_usage = usage_of(plan_usage_info) or {}
        decisions = Counter(v if isinstance(v, str) else "other" for v in slot_results.values())
        if result is not None:
            observed = {name: slot_usage_total[name] + (plan_usage.get(name) or 0) for name in USAGE_FIELDS}
            if any(observed[name] != (result.get("usage") or {}).get(name) for name in USAGE_FIELDS):
                problems["capability usage differs from result.json"] += 1
        usage_total = (result or {}).get("usage") or {
            name: slot_usage_total[name] + (plan_usage.get(name) or 0) for name in USAGE_FIELDS
        }
        cap_row = {
            "id": cap,
            "name": meta["name"],
            "outcome": meta["outcome"],
            "subject_id": meta["subject_id"],
            "domain": meta["domain"],
            "area": meta["area"],
            "section_path": meta["section_path"],
            "status": "done" if result else "failed" if failure else "planned" if plan else "pending",
            "slots_planned": len(planned) or None,
            "proposed": counts["proposed"],
            "accepted": decisions["accept"] if result else None,
            "rejected": decisions["reject"] if result else None,
            "cut_off": counts["cut_off"],
            "requests": (result or {}).get("requests"),
            **{name: usage_total.get(name) for name in USAGE_FIELDS},
            "wall_time": (result or {}).get("wall_time"),
            "plan_prompt_tokens": plan_usage.get("prompt_tokens"),
            "plan_completion_tokens": plan_usage.get("completion_tokens"),
            "plan_reasoning_tokens": plan_usage.get("reasoning_tokens"),
            "coverage_rationale": (plan or {}).get("coverage_rationale"),
            "git_sha": ((result or {}).get("provenance") or {}).get("git_sha"),
            "plan": plan,
            "result": result,
            "failure": failure,
            "catalog_entry": meta["entry"],
            "plan_request_path": stored_path(cap, "plan.request.json"),
            "plan_completions_path": stored_path(cap, "plan.completions.json"),
            "source_key": cap_key,
        }
        return cap_row, proposal_rows

    read_started = time.monotonic()
    with ThreadPoolExecutor(read_threads) as pool:
        built = list(pool.map(capability_rows, ids))
    cap_rows, slot_rows = [], []
    for row in [row for row, _ in built if row is not None] + [row for _, rows in built for row in rows]:
        clean, replaced = without_nul(row)
        if replaced:
            problems["rows with NUL characters replaced by U+FFFD"] += 1
            print(
                f"warning: replaced {replaced} NUL characters in {row.get('id') or (row['capability_id'], row['slot'])}"
            )
        (cap_rows if "id" in clean else slot_rows).append(clean)
    print(
        f"read {len(cap_rows)} changed capabilities and {len(slot_rows)} changed proposals "
        f"in {time.monotonic() - read_started:.0f}s",
        flush=True,
    )
    upload_started = time.monotonic()
    # Slots first, so a stored capability key never vouches for slots that failed to land.
    uploaded_slots = uploader.upsert("proposals", PROPOSAL_COLUMNS, "capability_id, slot", slot_rows) if slot_rows else 0
    uploaded_caps = uploader.upsert("capabilities", CAPABILITY_COLUMNS, "id", cap_rows) if cap_rows else 0
    print(
        f"upserted {uploaded_caps} capabilities, {uploaded_slots} proposals in {time.monotonic() - upload_started:.0f}s"
    )

    stored = uploader.query(
        "SELECT (SELECT COUNT(*) FROM capabilities) AS capabilities, (SELECT COUNT(*) FROM proposals) AS proposals, "
        "(SELECT COUNT(*) FROM proposals WHERE status = 'proposed') AS proposed, "
        "(SELECT COUNT(*) FROM capabilities WHERE status = 'done') AS done, "
        "(SELECT COUNT(*) FROM capabilities WHERE status = 'failed') AS failed, "
        "(SELECT COALESCE(SUM(slots_planned), 0) FROM capabilities) AS planned",
        {},
    )["rows"][0]
    ingest = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "seconds": round(time.monotonic() - started),
        "run_root": run_root,
        "catalog": catalog_url,
        "catalog_version": catalog_doc["catalog_version"],
        "catalog_capabilities": len(catalog),
        "folders": len(files),
        "files": len(listing),
        "result_files": sum(1 for f in files.values() if "result.json" in f),
        "failure_files": sum(1 for f in files.values() if "failure.json" in f),
        "usage_summaries": len(usage),
        "changed_capabilities": uploaded_caps,
        "changed_proposals": uploaded_slots,
        "problems": dict(problems),
        "stored": stored,
        "sql_requests": uploader.requests,
        "ingest_version": INGEST_VERSION,
    }
    uploader.query(
        "INSERT INTO run_stats (key, value) VALUES ('ingest', CAST(:value AS JSONB)) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        {"value": json.dumps(ingest)},
    )
    uploader.query(
        "INSERT INTO run_stats (key, value) SELECT 'provenance', jsonb_build_object("
        "'git_sha', (SELECT jsonb_object_agg(git_sha, n) FROM (SELECT git_sha, COUNT(*) AS n FROM capabilities "
        "WHERE git_sha IS NOT NULL GROUP BY 1) g), "
        "'policy', (SELECT jsonb_agg(DISTINCT result -> 'provenance' -> 'policy') FROM capabilities "
        "WHERE result IS NOT NULL), "
        "'catalog', (SELECT jsonb_agg(DISTINCT result -> 'provenance' -> 'catalog') FROM capabilities "
        "WHERE result IS NOT NULL), "
        "'max_tokens', (SELECT jsonb_agg(DISTINCT result -> 'provenance' -> 'max_tokens') FROM capabilities "
        "WHERE result IS NOT NULL), "
        "'slots', (SELECT jsonb_agg(DISTINCT result -> 'provenance' -> 'slots') FROM capabilities "
        "WHERE result IS NOT NULL)) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        {},
    )
    print(json.dumps(ingest, indent=2))
    if problems:
        print(f"warning: {dict(problems)}", flush=True)


if __name__ == "__main__":
    main()
