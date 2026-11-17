# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Forge tasks app's data from a forge task export and its Atlas selection.

Inputs: the export root written by ``experiments/forge/export_tasks.py`` (``records/<id>/task.json.gz`` and
``trials.json.gz``), the selection written by ``experiments.post_training.task_curation.forge_release``, and the
capability catalog the tasks were proposed from.

Outputs:

- ``applet/data/`` (shipped in the package, loaded on first paint): ``meta.json``, ``tasks.json`` (one compact row
  per selected task, enough for every overview chart and the task browser) and ``subjects/<D..>.json`` (catalog
  entries of the covered capabilities, loaded when a subject or capability page opens).
- ``<export>/app/<snapshot>/detail/<subject>.pack`` and ``trials/<capability>.json.gz`` on object storage, which
  the backend serves and caches: one task's full public record, and trimmed solver transcripts.

Run with CoreWeave credentials exported::

    uv run infra/marina/apps/forge_tasks/build_data.py --export s3://marin-us-east-02a/marin/forge-viz/v1 \
        --release /tmp/forge-release --snapshot 20261117T1800Z
"""

import argparse
import base64
import concurrent.futures
import gzip
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

CATALOG = (
    "s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/curriculum.json"
)
APP_DATA = Path(__file__).parent / "applet" / "data"
BAND = (0.25, 0.75)
READ_THREADS = 64
PACK_HEADER_BYTES = 8
TEXT_FILE_LIMIT = 96_000
SHORT = 600
MEDIUM = 2_000
LONG = 6_000
REASONING_HEAD = 2_500
REASONING_TAIL = 1_200
TRANSCRIPT_HEAD = 24
TRANSCRIPT_TAIL = 8
FIELDS = (
    "id",
    "cap",
    "slot",
    "title",
    "grader",
    "env",
    "band",
    "passes",
    "rounds",
    "reverted",
    "controls",
    "fa",
    "fr",
    "ctl",
    "defects",
    "parts",
    "effects",
    "band_rounds",
    "score",
    "chars",
    "tags",
)


TRIM_MARK = re.compile(r"more characters\]|characters omitted\]")


def clip(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n… [{len(text) - limit:,} more characters]"


def head_tail(text: str | None, head: int, tail: int) -> str | None:
    if text is None or len(text) <= head + tail:
        return text
    return f"{text[:head]}\n\n… [{len(text) - head - tail:,} characters omitted] …\n\n{text[-tail:]}"


def subject_of(capability: str) -> str:
    return capability.split(".", 1)[0].upper()


def passed(grade: Mapping | None) -> bool:
    if grade is None or grade["status"] == "submission_failure":
        return False
    if grade.get("passed") is not None:
        return grade["passed"]
    return (grade.get("reward") or 0.0) >= grade.get("score_max", 1.0) - 1e-9


HEADING = re.compile(r"^#{1,6}\s+(.+)$")
MARKUP = re.compile(r"`|\*\*|__")
TITLE_LIMIT = 110


ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "st", "no", "fig", "vs", "eg", "ie", "etc", "inc", "ltd", "co", "approx"}
)
SENTENCE_END = re.compile(r"[.!?]\s+(?=[A-Z\"'(])")
MIN_TITLE_WORDS = 4


def first_sentence(text: str) -> str:
    """The first sentence of four or more words, not ending on an abbreviation such as "Dr."."""
    start = 0
    for end in SENTENCE_END.finditer(text):
        before = text[start : end.start()].split()
        if not before or before[-1].lower().replace(".", "") in ABBREVIATIONS:
            continue
        if len(before) >= MIN_TITLE_WORDS:
            return text[start : end.start() + 1]
        start = end.end()
    return text[start:]


def title_of(prompt: str, capability: str) -> str:
    """A specific heading of three or more words, else the prompt's first sentence, cut at a word."""
    body = []
    for line in prompt.splitlines():
        if match := HEADING.match(line.strip()):
            heading = MARKUP.sub("", match.group(1)).strip().rstrip(":")
            if len(heading.split()) >= 3 and not body:
                return heading[:TITLE_LIMIT]
            continue
        body.append(line)
    plain = MARKUP.sub("", " ".join(" ".join(body).split()))
    sentence = first_sentence(plain)
    if len(sentence) <= TITLE_LIMIT:
        return sentence or capability
    return sentence[:TITLE_LIMIT].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def prompt_messages(task: Mapping) -> list[dict]:
    return [
        {"role": event.get("role", "user"), "content": event.get("content") or ""}
        for event in task["context"]["events"]
        if event.get("type") == "message"
    ]


def files_of(task: Mapping) -> list[dict]:
    out = []
    for group, entries in task["resources"].items():
        for entry in entries:
            source = entry["source"]
            text = None
            size = None
            if source.get("kind") == "inline_file":
                raw = base64.b64decode(source["content_base64"])
                size = len(raw)
                if size <= TEXT_FILE_LIMIT:
                    try:
                        text = raw.decode()
                    except UnicodeDecodeError:
                        text = None
            out.append({"group": group, "path": entry["path"], "kind": source.get("kind"), "size": size, "text": text})
    return out


def grader_of(task: Mapping) -> dict:
    grader = dict(task["grader"])
    environment = grader.pop("environment", None)
    if environment:
        grader["image"] = environment.get("docker_image")
    return grader


def finding_view(finding: Mapping) -> dict:
    defect = finding.get("defect")
    grade = finding.get("grade")
    return {
        "claim": clip(finding["claim"], MEDIUM),
        "reply": clip(finding["candidate"]["reply"], MEDIUM),
        "files": [f["path"] for f in finding["candidate"]["files"]],
        "confirmed": finding["met"],
        "defect": defect["defect"] if defect else None,
        "deserved": defect["deserved_reward"] if defect else None,
        "reward": grade.get("reward") if grade else None,
        "status": grade["status"] if grade else None,
        "revision": finding.get("revision", 0),
    }


def defect_view(defect: Mapping) -> dict:
    return {
        "part": defect["part"],
        "effect": defect["effect"],
        "confidence": defect["confidence"],
        "description": clip(defect["description"], MEDIUM),
        "fix": clip(defect["fix"], SHORT * 2),
        "evidence": [{"trial": e["trial"], "excerpt": clip(e["excerpt"], SHORT)} for e in defect["evidence"]][:4],
    }


def grade_view(grade: Mapping | None) -> dict | None:
    if grade is None:
        return None
    diagnostics = grade.get("diagnostics") or {}
    return {
        "status": grade["status"],
        "reward": grade.get("reward"),
        "passed": passed(grade),
        "stdout": clip(diagnostics.get("stdout"), MEDIUM),
        "stderr": clip(diagnostics.get("stderr"), SHORT),
        "error": clip(grade.get("error"), SHORT),
        "seconds": diagnostics.get("grader_elapsed_seconds"),
    }


def round_view(record: Mapping, final_digest: str) -> dict:
    review = record.get("solve_review")
    return {
        "confirmation": record["confirmation"],
        "final": record["task_digest"] == final_digest,
        "solver": [
            {
                "trial": t["trial"],
                "passed": passed(t["grade"]),
                "reward": (t["grade"] or {}).get("reward"),
                "cause": t["cause"],
            }
            for t in record["solver"]
        ],
        "controls": [
            {"index": c["index"], "met": c["met"], "reward": (c["grade"] or {}).get("reward")}
            for c in record["controls"]
        ],
        "adversaries": [
            {"name": a["name"], "completed": a["completed"], "findings": [finding_view(f) for f in a["findings"]]}
            for a in record["adversary"]
        ],
        "review": (
            None
            if review is None
            else {
                "completed": review["completed"],
                "defects": [defect_view(d) for d in review["defects"]],
                "notes": [defect_view(d) for d in review["notes"]],
                "trials": [{**t, "reason": clip(t["reason"], SHORT)} for t in review["trials"]],
            }
        ),
        "lowering_error": clip(record.get("lowering_error"), SHORT),
        "missing": record.get("missing", []),
    }


def decision_view(decision: Mapping) -> dict:
    out = {"kind": decision["kind"]}
    for key in ("band", "placement", "revert", "cause", "count", "category"):
        if decision.get(key) is not None:
            out[key] = decision[key]
    for key in ("reasons", "findings"):
        if decision.get(key):
            out[key] = [clip(item, MEDIUM) for item in decision[key]]
    if decision.get("defects"):
        out["defects"] = [defect_view(d) for d in decision["defects"]]
    return out


def task_detail(record: Mapping, summary: Mapping) -> dict:
    workpiece = record["workpiece"]
    submission = workpiece["submission"]
    task = submission["task"]
    header = workpiece["proposal"]["header"]
    confirmation = workpiece["evidence"][-1]
    controls = []
    observed = {c["index"]: c for c in confirmation["controls"]}
    for index, control in enumerate(submission["controls"]):
        seen = observed.get(index)
        controls.append(
            {
                "kind": control["kind"],
                "reply": clip(control["candidate"]["reply"], LONG),
                "files": [f["path"] for f in control["candidate"]["files"]],
                "expected": control["expected"],
                "met": seen["met"] if seen else None,
                "reward": (seen["grade"] or {}).get("reward") if seen else None,
            }
        )
    requirements = task["environment_requirements"]
    return {
        "id": record["id"],
        "capability": record["capability"],
        "slot": record["slot"],
        "title": summary["title"],
        "prompt": [{**m, "content": clip(m["content"], 40_000)} for m in prompt_messages(task)],
        "answer": {
            "type": task["answer_type"],
            "format": task["answer_format"],
            "path": task["grader"].get("answer_path"),
        },
        "environment": {
            "kind": summary["env"],
            "workdir": requirements.get("working_directory"),
            "semantics": requirements.get("command_semantics"),
            "tools": [tool["name"] for tool in task["interaction_tools"] + task["final_tools"]],
        },
        "verification": submission["verification"] or header["verification"],
        "grader": grader_of(task),
        "files": files_of(task),
        "tags": task.get("tags", []),
        "controls": controls,
        "rounds": [round_view(r, workpiece["evidence"][-1]["task_digest"]) for r in workpiece["evidence"]],
        "decisions": [decision_view(d) for d in workpiece["decisions"]],
        "notes": clip(submission["notes"], LONG),
        "proposal": {
            "body": clip(workpiece["proposal"]["body"], 20_000),
            "grounding": header["grounding"],
            "research_need": header["research_need"],
            "build": header["build"],
        },
        "arm": record["arm"],
    }


def message_excerpt(message: Mapping) -> dict:
    out = {"role": message.get("role")}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    limit = LONG if message.get("role") == "assistant" else MEDIUM
    out["content"] = head_tail(content, limit - 1_000, 1_000) if content else content
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        out["reasoning_content"] = head_tail(reasoning, REASONING_HEAD, REASONING_TAIL)
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": call.get("id"),
                "function": {
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": clip((call.get("function") or {}).get("arguments"), MEDIUM),
                },
            }
            for call in message["tool_calls"]
        ]
    if message.get("tool_call_id"):
        out["tool_call_id"] = message["tool_call_id"]
    return out


def trial_excerpt(trial: Mapping) -> dict:
    messages = [m for conversation in trial.get("conversations", []) for m in conversation["messages"]]
    total = len(messages)
    if total > TRANSCRIPT_HEAD + TRANSCRIPT_TAIL:
        kept = messages[:TRANSCRIPT_HEAD] + messages[-TRANSCRIPT_TAIL:]
    else:
        kept = messages
    # The task prompt is shown on the task page; drop its copy from the transcript's first user turn.
    excerpt = [message_excerpt(m) for m in kept]
    if excerpt and excerpt[0]["role"] == "user":
        excerpt[0]["content"] = "(the task prompt, shown above)"
    return {
        "trial": trial["trial"],
        "grade": grade_view(trial["grade"]),
        "cause": trial["cause"],
        "stop_reason": trial.get("stop_reason"),
        "turns": sum(m.get("role") == "assistant" for m in messages),
        "tool_calls": sum(len(m.get("tool_calls") or []) for m in messages),
        "messages": total,
        "omitted": max(0, total - len(kept)),
        # True when anything was cut, so the page knows to load the full transcript behind the excerpt.
        "trimmed": total > len(kept) or bool(TRIM_MARK.search(json.dumps(excerpt))),
        "omitted_at": TRANSCRIPT_HEAD if total > len(kept) else None,
        "transcript": excerpt if trial.get("conversations") else None,
    }


def summary_row(record: Mapping, catalog_index: Mapping[str, int], selection: Mapping) -> dict:
    workpiece = record["workpiece"]
    submission = workpiece["submission"]
    task = submission["task"]
    confirmation = workpiece["evidence"][-1]
    prompt = "\n\n".join(m["content"] for m in prompt_messages(task))
    fa = fr = ctl = defects = band_rounds = 0
    parts: set[str] = set()
    effects: set[str] = set()
    for evidence in workpiece["evidence"]:
        for adversary in evidence["adversary"]:
            confirmed = sum(f["met"] for f in adversary["findings"])
            if adversary["name"] == "false_accept":
                fa += confirmed
            elif adversary["name"] == "false_reject":
                fr += confirmed
        ctl += sum(not c["met"] for c in evidence["controls"])
        review = evidence.get("solve_review") or {}
        for defect in review.get("defects", []):
            defects += 1
            parts.add(defect["part"])
            effects.add(f"{defect['part']}:{defect['effect']}")
    for decision in workpiece["decisions"]:
        if (
            decision["kind"] == "modify"
            and not decision.get("defects")
            and decision.get("placement") not in (None, "in_band")
        ):
            if all("solved" in finding for finding in decision.get("findings", [])):
                band_rounds += 1
    return {
        "id": record["id"],
        "cap": catalog_index[record["capability"]],
        "slot": record["slot"],
        "title": title_of(prompt, record["capability"]),
        "grader": selection["verification"],
        "env": selection["environment"],
        "band": selection["band"],
        "passes": sum(passed(t["grade"]) for t in confirmation["solver"]),
        "rounds": selection["modify_rounds"],
        "reverted": selection["reverted"],
        "controls": len(submission["controls"]),
        "fa": fa,
        "fr": fr,
        "ctl": ctl,
        "defects": defects,
        "parts": sorted(parts),
        "effects": sorted(effects),
        "band_rounds": band_rounds,
        "score": selection["score"],
        "chars": len(prompt),
        "tags": task.get("tags", [])[:8],
    }


def catalog_tables(catalog: Mapping, covered: set[str]) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    """Subjects with their capability totals, the covered capabilities (index order), and per-subject detail."""
    sections = {s["id"]: (c["curriculum"], s) for c in catalog["curricula"] for s in c["curriculum"]["sections"]}

    def top_group(section: Mapping) -> Mapping:
        while section.get("parent_id") and sections[section["parent_id"]][1].get("parent_id"):
            section = sections[section["parent_id"]][1]
        return sections[section["parent_id"]][1] if section.get("parent_id") else section

    subjects = []
    for entry in catalog["curricula"]:
        curriculum = entry["curriculum"]
        capabilities = [s for s in curriculum["sections"] if s["kind"] == "capability"]
        subjects.append(
            {
                "id": curriculum["subject_id"],
                "name": curriculum["subject_name"],
                "capabilities": len(capabilities),
                "covered": sum(s["id"] in covered for s in capabilities),
            }
        )
    caps = []
    detail: dict[str, list[dict]] = defaultdict(list)
    for capability in sorted(covered):
        curriculum, section = sections[capability]
        group = top_group(section)
        caps.append(
            {
                "id": capability,
                "name": section["name"],
                "subject": curriculum["subject_id"],
                "group": group["id"],
                "group_name": group["name"],
            }
        )
        detail[curriculum["subject_id"]].append(
            {
                "id": capability,
                "name": section["name"],
                "outcome": section.get("outcome"),
                "includes": section.get("includes", []),
                "excludes": section.get("excludes", []),
                "sample_tasks": section.get("sample_tasks", []),
                "group": group["id"],
                "group_name": group["name"],
                "group_scope": group.get("scope"),
            }
        )
    return subjects, caps, detail


def gz_json(value: object) -> bytes:
    return gzip.compress(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(), mtime=0)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


def write_gz_json(path: Path, value: object) -> None:
    """Marina serves package files uncompressed, so the large index files ship gzipped; the page inflates them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gz_json(value))


def packed_shard(tasks: Mapping[str, object]) -> bytes:
    """One subject's task records as independently gzipped members behind an offset index.

    Layout: an 8-byte big-endian index length, the JSON index ``{id: [offset, length]}``, then the members. The
    backend serves a task by slicing its member out, with no decompression or recompression.
    """
    members, index, offset = [], {}, 0
    for identity, task in tasks.items():
        member = gz_json(task)
        index[identity] = [offset, len(member)]
        members.append(member)
        offset += len(member)
    header = json.dumps(index, separators=(",", ":")).encode()
    return len(header).to_bytes(PACK_HEADER_BYTES, "big") + header + b"".join(members)


def build(export: str, release: Path, snapshot: str, catalog_url: str) -> None:
    fs, base = fsspec.core.url_to_fs(export)
    selection = [json.loads(line) for line in (release / "selection.jsonl").read_text().splitlines()]
    manifest = json.loads((release / "manifest.json").read_text())
    chosen = {row["id"]: row for row in selection if row["choice"] == "selected"}
    if len(chosen) != manifest["selected"]:
        raise ValueError(f"selection lists {len(chosen)} selected tasks, manifest {manifest['selected']}")
    catalog_fs, catalog_path = fsspec.core.url_to_fs(catalog_url)
    catalog = json.loads(catalog_fs.cat_file(catalog_path))

    def read(identity: str) -> tuple[dict, dict]:
        prefix = f"{base}/records/{identity}"
        return (
            json.loads(gzip.decompress(fs.cat_file(f"{prefix}/task.json.gz"))),
            json.loads(gzip.decompress(fs.cat_file(f"{prefix}/trials.json.gz"))),
        )

    with concurrent.futures.ThreadPoolExecutor(READ_THREADS) as pool:
        pairs = list(pool.map(read, sorted(chosen)))
    covered = {record["capability"] for record, _ in pairs}
    subjects, caps, subject_detail = catalog_tables(catalog, covered)
    catalog_index = {cap["id"]: index for index, cap in enumerate(caps)}

    rows = []
    details: dict[str, dict] = defaultdict(dict)
    trials: dict[str, dict] = defaultdict(dict)
    for record, trial_record in pairs:
        row = summary_row(record, catalog_index, chosen[record["id"]])
        rows.append(row)
        details[subject_of(record["capability"])][record["id"]] = task_detail(record, row)
        trials[record["capability"]][record["id"]] = [trial_excerpt(t) for t in trial_record["trials"]]

    meta = {
        "snapshot": snapshot,
        "catalog_version": catalog["catalog_version"],
        "band": BAND,
        "trials": 8,
        "pool": {"accepted": manifest["candidates"], **manifest["choices"]},
        "rule": manifest["rule"],
        "subjects": subjects,
        "caps": caps,
        "data": f"{export}/app/{snapshot}",
    }
    APP_DATA.mkdir(parents=True, exist_ok=True)
    for stale in [*APP_DATA.rglob("*.json"), *APP_DATA.rglob("*.json.gz")]:
        stale.unlink()
    write_gz_json(APP_DATA / "meta.json.gz", meta)
    write_gz_json(APP_DATA / "tasks.json.gz", {"fields": FIELDS, "rows": [[row[f] for f in FIELDS] for row in rows]})
    for subject, entries in subject_detail.items():
        write_json(APP_DATA / "subjects" / f"{subject}.json", entries)

    target = f"{base}/app/{snapshot}"
    uploads = [(f"{target}/detail/{subject}.pack", packed_shard(tasks)) for subject, tasks in details.items()]
    uploads += [(f"{target}/trials/{capability}.json.gz", gz_json(tasks)) for capability, tasks in trials.items()]
    with concurrent.futures.ThreadPoolExecutor(READ_THREADS) as pool:
        list(pool.map(lambda item: fs.pipe_file(*item), uploads))
    print(f"{len(rows)} tasks, {len(caps)} capabilities, {len(details)} detail shards, {len(trials)} trial shards")
    print(f"package data: {sum(p.stat().st_size for p in APP_DATA.rglob('*.json')):,} bytes in {APP_DATA}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--catalog", default=CATALOG)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    build(args.export, args.release, args.snapshot, args.catalog)


if __name__ == "__main__":
    main()
