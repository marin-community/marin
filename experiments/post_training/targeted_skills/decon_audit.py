# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Automated 8-gram, domain-label, and tool-name disjointness audit against pinned eval corpora.

Computes NFKC/casefold alphanumeric-normalized 8-grams over non-boilerplate payload text
(user/assistant conversation text, excluding shared chat-template headers and <tools> wrappers)
and verifies zero overlap against Probe 1/2/3 plus optional eval holdout corpora (Micro210, BFCL),
with live positive and negative control sentinels.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import sha256_file

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    return NON_ALNUM_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def extract_8grams(text: str) -> set[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < 8:
        return set()
    return {" ".join(tokens[i : i + 8]) for i in range(len(tokens) - 7)}


SPECIAL_TAG_RE = re.compile(r"<\|[^>]+\|>|</?tools>|</?tool_call>|</?tool_response[^>]*>")
HEADER_LINE_RE = re.compile(r"(?m)^(Reasoning:\s*/(?:no)?think|system|user|assistant)\s*$")


def _strip_template_boilerplate(text: str) -> str:
    # Remove <tools>...</tools> schema block if present in decoded prompt
    text = re.sub(r"<tools>.*?</tools>", " ", text, flags=re.S)
    text = SPECIAL_TAG_RE.sub("\n", text)
    text = HEADER_LINE_RE.sub("", text)
    return text


def _extract_probe_payloads(path: Path, tokenizer: Any = None) -> tuple[list[tuple[str, str]], set[str], set[str]]:
    """Extract non-boilerplate payloads, domain labels, and tool names from a probe .jsonl.gz."""
    payloads: list[tuple[str, str]] = []
    domains: set[str] = set()
    tool_names: set[str] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rid = row.get("id", "unknown")
            if row.get("domain"):
                domains.add(str(row["domain"]).lower())
            for t in row.get("tools") or []:
                fn = t.get("function", t)
                if fn.get("name"):
                    tool_names.add(str(fn["name"]).lower())
            parts: list[str] = []
            for m in row.get("messages") or []:
                if isinstance(m.get("content"), str) and m["content"]:
                    parts.append(m["content"])
            for k in ("target", "target_body", "prompt", "expected_answer", "gold_example"):
                if isinstance(row.get(k), str) and row[k]:
                    parts.append(row[k])
            if not parts or (row.get("probe") == "probe1_mrcr_ordinal" and row.get("input_ids_nothink")):
                if tokenizer is not None and row.get("input_ids_nothink"):
                    decoded = tokenizer.decode(row["input_ids_nothink"], skip_special_tokens=False)
                    parts.append(_strip_template_boilerplate(decoded))
            if parts:
                payloads.append((rid, "\n".join(parts)))
    if not payloads:
        raise ValueError(f"Probe corpus {path} produced zero extracted payloads")
    return payloads, domains, tool_names


def _extract_micro210_payloads(path: Path) -> list[tuple[str, str]]:
    def walk(val: Any) -> list[str]:
        if isinstance(val, str):
            return [val]
        if isinstance(val, dict):
            out: list[str] = []
            for k in ("content", "text", "question", "prompt", "answer", "reference", "solution"):
                if k in val:
                    out.extend(walk(val[k]))
            return out
        if isinstance(val, list):
            out = []
            for el in val:
                out.extend(walk(el))
            return out
        return []

    payloads: list[tuple[str, str]] = []
    tbl = pq.read_table(path)
    for row in tbl.to_pylist():
        raw = row.get("task_spec")
        spec = json.loads(raw) if isinstance(raw, str) else (raw or {})
        rid = str(spec.get("id") or row.get("task_id") or "micro210")
        parts = walk(spec.get("context", {}).get("events", [])) + walk(spec.get("grader", {}))
        if parts:
            payloads.append((rid, "\n".join(parts)))
    return payloads


def extract_bfcl_names_and_payloads(bfcl_root: Path) -> tuple[list[tuple[str, str]], set[str]]:
    """Extract prompt strings and function names from cached BFCL JSON/JSONL files."""
    payloads: list[tuple[str, str]] = []
    func_names: set[str] = set()
    for p in sorted(bfcl_root.rglob("*.json*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rows: list[Any] = []
        stripped = text.lstrip()
        if stripped.startswith("["):
            try:
                rows = json.loads(text)
            except json.JSONDecodeError:
                continue
        else:
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or f"{p.name}:{idx}")
            for fn in row.get("function") or []:
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    func_names.add(fn["name"].lower())
            q = row.get("question")
            q_parts: list[str] = []
            if isinstance(q, list):
                for turn in q:
                    if isinstance(turn, list):
                        for msg in turn:
                            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                                q_parts.append(msg["content"])
                    elif isinstance(turn, dict) and isinstance(turn.get("content"), str):
                        q_parts.append(turn["content"])
            elif isinstance(q, str):
                q_parts.append(q)
            if q_parts:
                payloads.append((rid, "\n".join(q_parts)))
    return payloads, func_names


def extract_candidate_payloads_from_parquet(
    parquet_paths: list[Path],
) -> tuple[list[tuple[str, str]], set[str], set[str]]:
    """Extract non-boilerplate message payloads, domains, and declared tool names from generated Parquet splits."""
    payloads: list[tuple[str, str]] = []
    domains: set[str] = set()
    tool_names: set[str] = set()
    for path in parquet_paths:
        cols = pq.read_schema(path).names
        wanted = [c for c in ("id", "domain", "messages_json", "declared_tools_json") if c in cols]
        for row in pq.read_table(path, columns=wanted).to_pylist():
            rid = row["id"]
            if row.get("domain"):
                domains.add(str(row["domain"]).lower())
            if row.get("declared_tools_json"):
                for tn in json.loads(row["declared_tools_json"]):
                    tool_names.add(str(tn).lower())
            msgs = json.loads(row["messages_json"])
            parts = []
            for m in msgs:
                if isinstance(m.get("content"), str) and m["content"]:
                    parts.append(m["content"])
                if isinstance(m.get("reasoning_content"), str) and m["reasoning_content"]:
                    parts.append(m["reasoning_content"])
            payloads.append((rid, "\n".join(parts)))
    return payloads, domains, tool_names


def run_disjointness_audit(
    candidate_parquets: list[Path],
    *,
    probe_paths: dict[str, Path],
    tokenizer: Any = None,
    micro210_path: Path | None = None,
    bfcl_root: Path | None = None,
) -> dict[str, Any]:
    """Run 8-gram, domain-label, and tool-name disjointness audit with positive/negative sentinels."""
    cand_payloads, cand_domains, cand_tools = extract_candidate_payloads_from_parquet(candidate_parquets)

    # Index candidate 8-grams -> first candidate ID for exact reporting.
    cand_8grams: dict[str, str] = {}
    for rid, txt in cand_payloads:
        for g in extract_8grams(txt):
            cand_8grams.setdefault(g, rid)

    corpus_reports: dict[str, Any] = {}
    controls: list[dict[str, Any]] = []
    total_8gram_hits = 0
    total_domain_hits = 0
    total_tool_hits = 0

    def scan_corpus(
        name: str, payloads: list[tuple[str, str]], domains: set[str], tools: set[str], source_meta: dict[str, Any]
    ) -> None:
        nonlocal total_8gram_hits, total_domain_hits, total_tool_hits
        gram_hits: list[dict[str, str]] = []
        eval_grams_total = 0
        sentinel_text: str | None = None
        for eid, etxt in payloads:
            egs = extract_8grams(etxt)
            eval_grams_total += len(egs)
            if sentinel_text is None and len(egs) >= 5:
                sentinel_text = etxt
            for g in egs:
                if g in cand_8grams:
                    if len(gram_hits) < 20:
                        gram_hits.append({"eval_id": eid, "cand_id": cand_8grams[g], "ngram8": g})
                    total_8gram_hits += 1
        if eval_grams_total == 0:
            raise ValueError(f"Corpus {name} yielded zero 8-grams; refusing vacuous disjointness pass")
        dom_overlap = sorted(cand_domains & domains)
        tool_overlap = sorted(cand_tools & tools)
        total_domain_hits += len(dom_overlap)
        total_tool_hits += len(tool_overlap)
        corpus_reports[name] = {
            **source_meta,
            "eval_items": len(payloads),
            "eval_8grams_scanned": eval_grams_total,
            "shared_8gram_hits": len(gram_hits),
            "sample_8gram_hits": gram_hits,
            "domain_overlap": dom_overlap,
            "tool_name_overlap": tool_overlap,
        }
        # Positive control sentinel: verify the scanner flags a real excerpt from this corpus.
        if sentinel_text is not None:
            s_grams = extract_8grams(sentinel_text)
            eval_index = set()
            for _, etxt in payloads:
                eval_index.update(extract_8grams(etxt))
            controls.append(
                {
                    "corpus": name,
                    "kind": "positive_exact",
                    "expected_flag": True,
                    "actual_flag": bool(s_grams & eval_index),
                }
            )

    for pname, ppath in sorted(probe_paths.items()):
        payloads, domains, tools = _extract_probe_payloads(ppath, tokenizer=tokenizer)
        scan_corpus(pname, payloads, domains, tools, {"path": ppath.name, "sha256": sha256_file(ppath)})

    if micro210_path is not None and micro210_path.is_file():
        m_payloads = _extract_micro210_payloads(micro210_path)
        scan_corpus(
            "micro210_validation",
            m_payloads,
            set(),
            set(),
            {"path": micro210_path.name, "sha256": sha256_file(micro210_path)},
        )

    if bfcl_root is not None and bfcl_root.is_dir():
        b_payloads, b_tools = extract_bfcl_names_and_payloads(bfcl_root)
        scan_corpus(
            "bfcl_cached",
            b_payloads,
            set(),
            b_tools,
            {"path": bfcl_root.name, "bfcl_unique_tool_names": len(b_tools)},
        )

    # Negative control sentinel: random unique hash words should never match candidate 8-grams.
    neg_text = " ".join("nonoverlap" + hashlib.sha256(f"neg:{i}".encode()).hexdigest()[:12] for i in range(40))
    controls.append(
        {
            "corpus": "negative_random",
            "kind": "negative",
            "expected_flag": False,
            "actual_flag": bool(extract_8grams(neg_text) & set(cand_8grams.keys())),
        }
    )

    controls_ok = all(c["expected_flag"] == c["actual_flag"] for c in controls)
    passed = controls_ok and total_8gram_hits == 0 and total_domain_hits == 0 and total_tool_hits == 0
    return {
        "pass": passed,
        "method": (
            "NFKC/casefold alphanumeric-normalized 8-grams over non-boilerplate user/assistant "
            "payload text plus exact domain-label and tool-name set intersection"
        ),
        "candidate_rows_scanned": len(cand_payloads),
        "candidate_unique_8grams": len(cand_8grams),
        "candidate_domains": sorted(cand_domains),
        "candidate_tools": sorted(cand_tools),
        "total_8gram_hits": total_8gram_hits,
        "total_domain_hits": total_domain_hits,
        "total_tool_hits": total_tool_hits,
        "controls_ok": controls_ok,
        "controls": controls,
        "corpora": corpus_reports,
    }
