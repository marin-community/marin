# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the option-G official-test label table for the comp-prog sources.

Reads the in-region audit join (cache row to UUID), the pinned Min-Li judge
receipts and rule flags, verifies the identity chain and the option table, and
writes one gzip JSONL label table plus a receipt. The table feeds
``comp_prog_label_rescue`` (see ``exp3_filters.CompProgLabels``).
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)


def read_remote_parquet(uri: str):
    """Read one small parquet object through the bucket-aware filesystem."""
    fs = filesystem_for(uri)
    fs = fs[0] if isinstance(fs, tuple) else fs
    return pq.read_table(io.BytesIO(fs.cat(uri)))


AUDIT_REVISION = "9735445ff6adb6308603f014fa833cce0b05d67d"
AUDIT_REPO = "Min-Li/marin-sft-data-audit"
JUDGE_RESULTS = {
    "cpp": (
        "comp_prog_cpp/official_tests/judge_results.parquet",
        "36a4909500b422667595c1db1b158304b1897b8bdea4ea0280aaca90037119df",
    ),
    "python": (
        "comp_prog_python/official_tests/judge_results.parquet",
        "9735f70e716987e578014445d28b6404cead003bb61e5b12a27a33516a9fc1bf",
    ),
}
RULE_FLAGS = {
    "cpp": "comp_prog_filter_check/rule_flags_cpp.parquet",
    "python": "comp_prog_filter_check/rule_flags_python.parquet",
}
THINK_LIMIT = 50_000
EXPECTED_JUDGED = 149_067
EXPECTED_TOTAL = 665_326
# Drop totals from the scripted-v3 ledgers, per the plan section 2.2.
EXPECTED_STAGE_DROPS = {"give_up_endings": 43_852, "think_length": 178_005, "": 443_469}
# The audit's own option table (tradeoffs.json at the 20261118 evidence prefix).
EXPECTED = {
    "A": {
        "judged_kept": 87_171,
        "fail_share_kept": 0.0401,
        "passing_kept": 83_673,
        "failing_kept": 3_498,
        "problems_lost": 466,
        "problems_lost_ge3000": 193,
    },
    "E2": {
        "judged_kept": 122_653,
        "fail_share_kept": 0.0,
        "passing_kept": 122_653,
        "failing_kept": 0,
        "problems_lost": 11,
        "problems_lost_ge3000": 9,
    },
    "G": {
        "judged_kept": 122_664,
        "fail_share_kept": 0.0,
        "passing_kept": 122_664,
        "failing_kept": 0,
        "problems_lost": 0,
        "problems_lost_ge3000": 0,
    },
}


@dataclass(frozen=True)
class SourceSpec:
    """Pinned inputs for one comp-prog language partition."""

    lang: str
    cache_source: str


def sha256_uri(uri: str) -> str:
    return hashlib.sha256(StoragePath(uri).read_bytes()).hexdigest()


@cache
def judged_labels(lang: str) -> dict[str, dict]:
    """Official-test verdicts keyed by UUID, selected exactly like the audit."""
    name, expected = JUDGE_RESULTS[lang]
    path = hf_hub_download(AUDIT_REPO, name, repo_type="dataset", revision=AUDIT_REVISION)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Judge receipt digest changed for {lang}: {digest}")
    table = pq.read_table(path).to_pandas()
    judged = table[table.wording_single_answer & (table.review_label.isna() | (table.review_label == "unique"))]
    judged = judged[judged.status != "not_judged"]
    return {
        str(record.uuid): {"problem": str(record.problem), "rating": record.rating, "status": str(record.status)}
        for record in judged.itertuples(index=False)
    }


@cache
def rule_flags(lang: str) -> dict[str, dict]:
    """Per-row think tokens and give-up matches, keyed by UUID."""
    path = hf_hub_download(AUDIT_REPO, RULE_FLAGS[lang], repo_type="dataset", revision=AUDIT_REVISION)
    flags = {}
    for record in pq.read_table(path).to_pylist():
        if not record["converted"]:
            continue
        flags[str(record["uuid"])] = {
            "give_up": bool(record["give_up_in_tail"]) or bool(record["give_up_in_answer"]),
            "think_tokens": int(record["think_tokens"] or 0),
        }
    return flags


def join_rows(
    specs: list[SourceSpec], join_prefix: str
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str], dict[str, str]]:
    """Map (lang, cache row) to UUID and ledger stage from the in-region audit join."""
    identity_sha = {name: sha256_uri(f"{join_prefix}/{name}") for name in ("source_ids.json", "chat_id_to_uuid.json")}
    source_ids = json.loads(StoragePath(f"{join_prefix}/source_ids.json").read_text())
    chat_id_to_uuid = json.loads(StoragePath(f"{join_prefix}/chat_id_to_uuid.json").read_text())
    row_to_uuid: dict[tuple[str, int], str] = {}
    row_stage: dict[tuple[str, int], str] = {}
    stage_drops: dict[str, int] = {}
    decisions = read_remote_parquet(f"{join_prefix}/our_row_decisions.parquet").to_pylist()
    audit_rows = {(record["lang"], record["part"], record["row"]): record for record in decisions}
    seen = 0
    for spec in specs:
        parts = sorted(source_ids[spec.lang])
        offset = 0
        for part in parts:
            part_key = part.removesuffix(".parquet")
            uuids = [chat_id_to_uuid[spec.lang][chat_id] for chat_id in source_ids[spec.lang][part]]
            for row, uuid in enumerate(uuids):
                row_to_uuid[(spec.lang, offset + row)] = uuid
                audit = audit_rows.get((spec.lang, part_key, row))
                if audit is None or audit["uuid"] != uuid:
                    raise ValueError(f"Audit identity mismatch for {spec.lang} {part} row {row}")
                row_stage[(spec.lang, offset + row)] = audit["our_stage"]
                stage_drops[audit["our_stage"]] = stage_drops.get(audit["our_stage"], 0) + 1
            offset += len(uuids)
            seen += len(uuids)
    if seen != EXPECTED_TOTAL:
        raise ValueError(f"Joined cache rows: {seen}, expected {EXPECTED_TOTAL}")
    if stage_drops != EXPECTED_STAGE_DROPS:
        raise ValueError(f"Ledger stage totals {stage_drops} differ from the audit {EXPECTED_STAGE_DROPS}")
    return row_to_uuid, row_stage, identity_sha


def evaluate_options(rows: list[dict]) -> dict[str, dict]:
    ratings = {}
    for row in rows:
        if row["status"] == "accepted" and row["rating"] is not None:
            ratings[row["problem"]] = row["rating"]
    counts = {}
    for option in ("A", "E2", "G"):
        judged = [row for row in rows if row["status"] is not None and row[f"keep_{option}"]]
        passing = {row["problem"] for row in judged if row["status"] == "accepted"}
        lost = [problem for problem in ratings if problem not in passing]
        counts[option] = {
            "judged_kept": len(judged),
            "fail_share_kept": (
                round(1 - sum(row["status"] == "accepted" for row in judged) / len(judged), 4) if judged else 0.0
            ),
            "passing_kept": sum(row["status"] == "accepted" for row in judged),
            "failing_kept": sum(row["status"] == "failed" for row in judged),
            "problems_lost": len(lost),
            "problems_lost_ge3000": sum(ratings[problem] >= 3000 for problem in lost),
        }
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, action="append", help="SourceSpec JSON (lang, cache_source)")
    parser.add_argument("--join-prefix", required=True, help="In-region audit join prefix")
    parser.add_argument("--scripted-plan", required=True)
    parser.add_argument("--scripted-plan-sha256", required=True)
    parser.add_argument("--destination", required=True, help="Output prefix for the label table and receipt")
    parser.add_argument("--output", required=True, help="Local directory for artifacts and logs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    specs = [SourceSpec(**json.loads(value)) for value in args.spec]
    plan = json.loads(read_pinned(args.scripted_plan, args.scripted_plan_sha256))
    part_paths: dict[str, dict[str, str]] = {}
    for spec in specs:
        jobs = [job for job in plan["jobs"] if job["source"] == spec.cache_source]
        mapping = {}
        for job in jobs:
            part = Path(job["input_path"].rstrip("/")).name
            if part in mapping:
                raise ValueError(f"Two scripted jobs cover part {part} of {spec.cache_source}")
            mapping[part] = job["input_path"]
        part_paths[spec.lang] = mapping
    row_to_uuid, row_stage, identity_sha = join_rows(specs, args.join_prefix)

    rows: list[dict] = []
    for spec in specs:
        labels, flags = judged_labels(spec.lang), rule_flags(spec.lang)
        # Rebuild the per-part layout so the table addresses rows the way the
        # build does: source, part path, part-local row.
        source_ids = json.loads(StoragePath(f"{args.join_prefix}/source_ids.json").read_text())
        offset = 0
        for part in sorted(source_ids[spec.lang]):
            part_name = part.removesuffix(".parquet")
            input_path = part_paths[spec.lang].get(part_name)
            if input_path is None:
                raise ValueError(f"Plan has no shard for part {part_name} of {spec.cache_source}")
            count = len(source_ids[spec.lang][part])
            for row in range(offset, offset + count):
                uuid = row_to_uuid[(spec.lang, row)]
                judged = labels.get(uuid)
                flag = flags.get(uuid)
                status = judged["status"] if judged else None
                give_up = bool(flag["give_up"]) if flag else False
                over_limit = bool(flag["think_tokens"] > THINK_LIMIT) if flag else False
                rows.append(
                    {
                        "lang": spec.lang,
                        "cache_part": input_path,
                        "cache_row": row - offset,
                        "uuid": uuid,
                        "problem": judged["problem"] if judged else None,
                        "rating": judged["rating"] if judged else None,
                        "status": status,
                        "give_up": give_up,
                        "think_tokens": flag["think_tokens"] if flag else 0,
                        "cache_source": spec.cache_source,
                        "keep_A": not (give_up or over_limit),
                        "keep_E2": (status == "accepted" and not give_up) if status else False,
                    }
                )
                if rows[-1]["keep_A"] != (row_stage[(spec.lang, row)] == ""):
                    raise ValueError(f"Filter disagreement for {spec.lang} row {row}")
            offset += count
        judged_in_caches = sum(1 for key in row_to_uuid if key[0] == spec.lang and row_to_uuid[key] in labels)
        logger.info("%s: %d cache rows, %d judged", spec.lang, offset, judged_in_caches)

    for row in rows:
        if row["status"] is None:
            row["keep_E2"] = row["keep_A"]

    # Option G rescue: each problem left with no accepted row keeps its shortest accepted row.
    accepted = [(index, row) for index, row in enumerate(rows) if row["status"] == "accepted"]
    survivors = {row["problem"] for _index, row in accepted if not row["give_up"]}
    shortest: dict[str, tuple[int, int]] = {}
    for index, row in accepted:
        if row["problem"] in survivors:
            continue
        # First occurrence wins ties, matching the frozen row order (cpp parts, then python).
        if row["problem"] not in shortest or row["think_tokens"] < shortest[row["problem"]][0]:
            shortest[row["problem"]] = (row["think_tokens"], index)
    rescued = {index for _tokens, index in shortest.values()}
    for row in rows:
        row["keep_G"] = row["keep_E2"]
    for index in rescued:
        rows[index]["keep_G"] = True
        if rows[index]["status"] != "accepted" or not rows[index]["give_up"]:
            raise ValueError("A rescued row is not an accepted row with a give-up phrase")

    judged_rows = [row for row in rows if row["status"]]
    if len(judged_rows) != EXPECTED_JUDGED:
        raise ValueError(f"Judged rows in caches: {len(judged_rows)}, expected {EXPECTED_JUDGED}")
    counts = evaluate_options(rows)
    for option, expected in EXPECTED.items():
        for key, value in expected.items():
            if counts[option][key] != value:
                raise ValueError(f"Option {option} {key}: {counts[option][key]} vs {value}")

    table_records = [
        {
            "lang": row["lang"],
            "cache_part": row["cache_part"],
            "cache_row": row["cache_row"],
            "uuid": row["uuid"],
            "problem": row["problem"],
            "rating": row["rating"],
            "status": row["status"],
            "give_up": row["give_up"],
            "think_tokens": row["think_tokens"],
            "cache_source": row["cache_source"],
        }
        for row in judged_rows
    ]
    table_path = output / "label-table.jsonl.gz"
    table_path.write_bytes(
        gzip.compress("\n".join(json.dumps(record, sort_keys=True) for record in table_records).encode())
    )
    receipt = {
        "judged_rows": len(judged_rows),
        "total_rows": len(rows),
        "rescued_rows": len(rescued),
        "rescued_problems": sorted({rows[index]["problem"] for index in rescued}),
        "option_counts": counts,
        "join_artifacts_sha256": identity_sha,
        "judge_results_sha256": {lang: expected for lang, (_name, expected) in JUDGE_RESULTS.items()},
        "label_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "audit_revision": AUDIT_REVISION,
        "think_limit": THINK_LIMIT,
        "spec_order": [spec.lang for spec in specs],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    destination = StoragePath(args.destination)
    (destination / "label-table.jsonl.gz").write_bytes(table_path.read_bytes())
    (destination / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps({"label_table": str(destination / "label-table.jsonl.gz"), "rescued_rows": len(rescued)}))


if __name__ == "__main__":
    main()
