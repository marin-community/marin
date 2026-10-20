# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a frozen local selection into the private Snowball browser."""

import argparse
import base64
import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from marina.client import DEFAULT_MARINA_URL, LOCAL_HOSTS, marina_request, marina_token_provider
from rigging.auth import TokenProvider
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_curriculum.bundle import verify_manifest

MAX_BATCH_BYTES = 900_000
Query = Callable[[str, dict[str, Any]], dict[str, Any]]
TASK_COLUMNS = (
    "contract_id TEXT, task_id TEXT, source_id TEXT, family TEXT, difficulty TEXT, "
    "preview TEXT, search_text TEXT, detail TEXT"
)


def native_structure_contract(lowered: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Extract solver-visible native closure inputs, excluding expected continuations."""
    configs = [resource for resource in lowered["task"]["resources"]["verifier"] if resource["path"] == "config.json"]
    if len(configs) != 1 or configs[0]["source"]["kind"] != "inline_file":
        raise ValueError("Native structural task requires one inline config.json")
    config = json.loads(base64.b64decode(configs[0]["source"]["content_base64"], validate=True))
    extras = config["extras"]
    if lowered["session"]["task_session"] != "structure_closure":
        raise ValueError("Native structural task declares a different session")
    return {
        "session_kind": "structure_closure",
        "supplied_prefix": extras["prefix"],
        "prompt_token_ids": extras["prompt_token_ids"],
        "format": extras["format"],
        "family": extras["family"],
        "control": extras["control"],
        "tokenizer_model": row["tokenizer_model"],
        "tokenizer_revision": row["tokenizer_revision"],
        "runtime_compatibility": "Historical source contract; current model rendering is unverified",
    }


def public_record(row: dict[str, Any], families: dict[str, str], aliases: list[dict[str, str]]) -> dict[str, Any]:
    """Project model-visible fields without verifier resources or reference answers."""
    contract_kind = row.get("contract_kind", "task_spec")
    native = None
    if contract_kind == "native-structure_closure":
        lowered = json.loads(row["lowered_task_spec"])
        task = lowered["task"]
        native = native_structure_contract(lowered, row)
    else:
        raw = row.get("task", row.get("task_json"))
        task = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(task, dict):
        raise ValueError("Selected row has no TaskSpec")
    source_id = row["source_id"]
    detail = {
        "task_id": row["task_id"],
        "contract_id": row["contract_id"],
        "contract_kind": contract_kind,
        "public_content_id": row.get("public_content_id"),
        "source_id": source_id,
        "family": families.get(source_id, "unknown"),
        "difficulty_prior": row.get("difficulty", "unknown"),
        "population": row.get("population", "unknown"),
        "prior_admission_status": row.get("prior_admission_status", "unknown"),
        "inclusion_probability": row.get("inclusion_probability"),
        "aliases": aliases,
        "context": task.get("context", {}),
        "answer_type": task.get("answer_type"),
        "answer_format": task.get("answer_format", {}),
        "final_tools": task.get("final_tools", []),
        "interaction_tools": task.get("interaction_tools", []),
        "output_paths": task.get("output_paths", []),
        "tags": task.get("tags", []),
        "source": task.get("source", {}),
        "grader_kind": task.get("grader", {}).get("kind", "unknown"),
        "environment_capabilities": task.get("environment_requirements", {}).get("capabilities", []),
    }
    context = detail["context"]
    prompt = "\n\n".join(event.get("content") or "" for event in context.get("events", []))
    preview = prompt
    if native is not None:
        detail["native_contract"] = native
        prefix = native["supplied_prefix"]
        prompt += "\n" + prefix + "\n" + " ".join(str(native[key]) for key in ("format", "family", "control"))
        preview = prefix
    return {
        "contract_id": row["contract_id"],
        "task_id": row["task_id"],
        "source_id": source_id,
        "family": detail["family"],
        "difficulty": detail["difficulty_prior"],
        "preview": preview[:260],
        "search_text": "\n".join((source_id, row["task_id"], row["contract_id"], prompt, " ".join(detail["tags"]))),
        "detail": json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
    }


def read_pool(
    root: Path, *, expected_manifest_sha256: str, artifact_uri: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and validate the complete frozen selection before any serving mutation."""
    verify_manifest(StoragePath(str(root)), expected_manifest_sha256)
    report = json.loads((root / "selection.json").read_text())
    recipe_path = root / "recipe.json"
    recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else {}
    families = recipe.get("resolved_source_families", recipe.get("source_families", recipe.get("extensions", {})))
    aliases_path = root / "selected-aliases.jsonl"
    aliases = {}
    if aliases_path.exists():
        for line in aliases_path.read_text().splitlines():
            row = json.loads(line)
            aliases[row["contract_id"]] = row["aliases"]
    tasks = []
    digest = hashlib.sha256()
    with (root / "tasks.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                row = json.loads(line)
                tasks.append(
                    public_record(
                        row,
                        families,
                        aliases.get(row["contract_id"], [{"source_id": row["source_id"], "task_id": row["task_id"]}]),
                    )
                )
    if len(tasks) != report["selected_count"]:
        raise ValueError("Task count differs from selection.json")
    if len({task["contract_id"] for task in tasks}) != len(tasks):
        raise ValueError("Selected pool contains duplicate contract IDs")
    counts = dict(Counter(task["source_id"] for task in tasks))
    if "source_quotas" in report and counts != {key: count for key, count in report["source_quotas"].items() if count}:
        raise ValueError("Source counts differ from the frozen selection quotas")
    readiness_path = root / "readiness.json"
    readiness = (
        json.loads(readiness_path.read_text())
        if readiness_path.exists()
        else {"contract_smoke": "not performed", "latest_stack_validation": "not performed"}
    )
    verify_manifest(StoragePath(str(root)), expected_manifest_sha256)
    metadata = {
        "selection": report,
        "recipe": recipe,
        "readiness": readiness,
        "artifact_uri": artifact_uri,
        "tasks_sha256": digest.hexdigest(),
        "source_manifest_sha256": expected_manifest_sha256,
        "difficulty_basis": "Provisional source/task-family prior; uncalibrated unless explicitly recorded in recipe.",
        "privacy": "Model-visible task projection. Reference answers and private grading resources are excluded.",
    }
    return metadata, tasks


def batches(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    batch = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        length = len(json.dumps(encoded).encode()) + 1
        if length + 2 > MAX_BATCH_BYTES:
            raise ValueError("One public task exceeds the SQL request budget")
        if size + length > MAX_BATCH_BYTES:
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        yield "[" + ",".join(batch) + "]"


def load_pool(
    root: Path, query: Query, *, expected_manifest_sha256: str, artifact_uri: str | None = None
) -> dict[str, Any]:
    """Stage a complete snapshot, verify counts, then atomically select it."""
    metadata, tasks = read_pool(root, expected_manifest_sha256=expected_manifest_sha256, artifact_uri=artifact_uri)
    current = query("SELECT load_id FROM pool_current WHERE name = 'latest'", {})["rows"]
    expected = current[0]["load_id"]
    load_id = uuid.uuid4().hex
    query(
        "INSERT INTO pool_loads (load_id, metadata) VALUES (:load_id, :metadata)",
        {
            "load_id": load_id,
            "metadata": json.dumps(metadata, ensure_ascii=False),
        },
    )
    columns = ", ".join(column.strip().split()[0] for column in TASK_COLUMNS.split(","))
    for body in batches(tasks):
        query(
            f"INSERT INTO pool_tasks (load_id, {columns}) SELECT :load_id, {columns} "
            f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({TASK_COLUMNS})",
            {"load_id": load_id, "rows": body},
        )
    count = query("SELECT COUNT(*) AS count FROM pool_tasks WHERE load_id = :load_id", {"load_id": load_id})["rows"][0][
        "count"
    ]
    if count != len(tasks):
        raise ValueError("Stored task count differs from complete selection")
    selected = query(
        "UPDATE pool_current SET load_id = :load_id WHERE name = 'latest' " "AND load_id IS NOT DISTINCT FROM :expected",
        {"load_id": load_id, "expected": expected},
    )
    if selected["row_count"] != 1:
        raise RuntimeError("Serving snapshot changed during load; staged data has not been selected")
    return {"load_id": load_id, "selected_count": count, "tasks_sha256": metadata["tasks_sha256"]}


def applet_query(
    service_url: str,
    applet_id: str,
    *,
    session: requests.Session | None = None,
    token_provider: TokenProvider | None = None,
) -> Query:
    details = marina_request(
        service_url,
        "GET",
        f"/api/marina/applets/{applet_id}",
        session=session,
        token_provider=token_provider,
    )
    assert isinstance(details, dict)
    parsed = urlparse(details["authenticated_url"])
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else service_url.rstrip("/")

    def query(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        result = marina_request(
            origin,
            "POST",
            f"/a/{applet_id}/query",
            json_body={"sql": sql, "parameters": parameters},
            session=session,
            token_provider=token_provider,
        )
        assert isinstance(result, dict)
        return result

    return query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--applet-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--artifact-uri", help="Private full-TaskSpec artifact URI shown separately from browser details"
    )
    parser.add_argument("--url", default=DEFAULT_MARINA_URL)
    args = parser.parse_args()
    provider = marina_token_provider() if urlparse(args.url).hostname not in LOCAL_HOSTS else None
    with requests.Session() as session:
        query = applet_query(args.url, args.applet_id, session=session, token_provider=provider)
        print(
            json.dumps(
                load_pool(
                    args.pool, query, expected_manifest_sha256=args.manifest_sha256, artifact_uri=args.artifact_uri
                )
            )
        )


if __name__ == "__main__":
    main()
