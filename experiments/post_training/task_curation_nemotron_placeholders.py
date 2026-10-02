# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Acquire immutable indexed source records for the Ultra release's math placeholders."""

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

PINS = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
    "Skywork/Skywork-OR1-RL-Data": "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
}


def placeholder_sources(
    rows: list[dict[str, Any]], cache_root: Path, *, max_transfer_bytes: int = 8 * 1024 * 1024
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add pinned upstream evidence while retaining every original blend field unchanged."""
    cache_root.mkdir(parents=True, exist_ok=True)
    enriched = []
    provenance = []
    transferred = 0
    for row in rows:
        placeholder = row.get("_hf_question_placeholder")
        if placeholder is None:
            enriched.append(row)
            continue
        dataset, split, index = placeholder["dataset"], placeholder["split"], int(placeholder["row"])
        pin = PINS[dataset]
        identity = hashlib.sha256(f"{dataset}@{pin}/{split}/{index}".encode()).hexdigest()
        path = cache_root / f"{identity}.json"
        if path.exists():
            source = json.loads(path.read_text())
        else:
            with requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": dataset, "config": "default", "split": split, "offset": index, "length": 1},
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()
                if response.headers.get("x-revision") != pin:
                    raise ValueError(f"Viewer revision does not match immutable source pin for {dataset}")
                chunks = []
                for chunk in response.iter_content(65536):
                    transferred += len(chunk)
                    if transferred > max_transfer_bytes:
                        raise ValueError("Placeholder acquisition exceeded transfer budget")
                    chunks.append(chunk)
            payload = json.loads(b"".join(chunks))
            if len(payload["rows"]) != 1 or payload["rows"][0]["row_idx"] != index:
                raise ValueError("Viewer did not return the exact requested source row")
            item = payload["rows"][0]
            if item.get("truncated_cells"):
                raise ValueError("Viewer truncated a required source field")
            record = item["row"]
            digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            source = {
                "dataset": dataset,
                "revision": pin,
                "split": split,
                "row_index": index,
                "record_sha256": digest,
                "record": record,
            }
            path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
        digest = hashlib.sha256(json.dumps(source["record"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if source["revision"] != pin or source["record_sha256"] != digest:
            raise ValueError("Retained placeholder source does not match its immutable identity/hash")
        enriched.append({**row, "placeholder_source": source})
        provenance.append({key: value for key, value in source.items() if key != "record"})
    return enriched, {
        "source_records": provenance,
        "transferred_bytes": transferred,
        "transfer_budget_bytes": max_transfer_bytes,
    }
