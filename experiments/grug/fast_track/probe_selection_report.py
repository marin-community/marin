# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print the source mix, document length, and score of quality selections and their raw prefixes."""

import json
import sys
from collections import Counter

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath, prefix_join


def _summary(shards: list[str]) -> dict:
    tokens: Counter[str] = Counter()
    documents = 0
    scores: list[float] = []
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            columns = [name for name in ("source", "token_count", "quality_score") if name in parquet.schema_arrow.names]
            table = parquet.read(columns=columns)
        for source, count in zip(
            table.column("source").to_pylist(), table.column("token_count").to_pylist(), strict=True
        ):
            tokens[source] += count
            documents += 1
        if "quality_score" in columns:
            scores.extend(table.column("quality_score").to_pylist())
    total = sum(tokens.values())
    return {
        "documents": documents,
        "tokens": total,
        "mean_document_tokens": total / documents,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "top_sources": {name: round(count / total, 4) for name, count in tokens.most_common(12)},
    }


def main() -> None:
    report = {}
    for path in sys.argv[1:]:
        with StoragePath(prefix_join(path, "selection.json")).open("r") as stream:
            selection = json.load(stream)
        report[path] = {
            "method": selection["selection_method"],
            "requested_training_tokens": selection["requested_training_tokens"],
            "raw_prefix_selected_tokens": selection["raw_prefix_selected_tokens"],
            "score_bounds": selection["score_bounds"],
            "selected": _summary(selection["selected_shards"]),
            "raw_prefix": _summary(selection["rung_prefix_shards"]),
        }
    print("REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
