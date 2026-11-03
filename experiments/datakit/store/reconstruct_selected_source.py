# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover selected source text from original provider files using a disk-backed ID join."""

import json
import sqlite3
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.store.davinci_provider_rows import DavinciCtxReadAudit, read_davinci_context
from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    provider_normalizer,
)
from experiments.datakit.store.structured_provider_rows import ADAPTERS

RECONSTRUCTED_TEXT_SCHEMA = pa.schema([("marin_source", pa.string()), ("id", pa.string()), ("text", pa.string())])
PROVIDER_COLUMNS = {
    **{ProviderTransform(adapter.transform): list(adapter.columns) for adapter in ADAPTERS.values()},
    ProviderTransform.PENFEVER: ["conversations", "verifier_output", "result"],
    ProviderTransform.SYNTHETIC1: ["prompt", "llm_response", "score"],
    ProviderTransform.GPT_OSS: ["user_content", "assistant_thinking", "assistant_content"],
    ProviderTransform.AGENTTROVE: [
        "conversations",
        "result",
        "model",
        "model_provider",
        "original_teacher",
        "original_source",
    ],
    ProviderTransform.CODERFORGE: ["messages", "reward"],
    ProviderTransform.DAVINCI_ENV: ["messages", "success"],
    ProviderTransform.DAVINCI_CTX: [
        "repo_name",
        "repo_description",
        "title",
        "body",
        "related_issue",
        "pr_summary",
        "relevant_files",
        "commits",
    ],
    ProviderTransform.SOFTWARE_HERITAGE: ["sha1_git", "sha1", "content", "present"],
    ProviderTransform.SOFTWARE_HERITAGE_EMPTY: ["sha1_git", "sha1", "content", "present"],
    ProviderTransform.FINETRANSLATIONS: ["id", "og_full_text", "translated_text"],
    ProviderTransform.INSTITUTIONAL_BOOKS: ["text_by_page_gen", "text_by_page_src"],
    ProviderTransform.STACK_V3: ["repo_path", "repo_id", "commit_id", "github_metadata", "num_files", "files"],
    ProviderTransform.SWE_CONTREE: ["text"],
    ProviderTransform.SWE_ZERO: ["messages"],
}


def provider_rows(path: str, columns: list[str], batch_size: int) -> Iterator[dict[str, Any]]:
    """Stream original provider records, projecting only fields required for replay."""
    fs, key = filesystem_for(path)
    if path.endswith(".parquet"):
        with fs.open(key, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            selected = [name for name in columns if name in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(columns=selected, batch_size=min(batch_size, 8)):
                yield from batch.to_pylist()
        return
    with fs.open(key, "rt", compression="infer") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                yield {name: record[name] for name in columns if name in record}


def reconstruct_selected_source(
    selected_paths: list[str],
    provider_paths: list[str],
    output_dir: str,
    *,
    marin_source: str,
    recipe: ReconstructionRecipe,
    batch_size: int = 8192,
) -> dict:
    """Reconstruct every selected occurrence of one source from raw provider rows.

    Selected paths must be the complete selected ledger for the desired cluster.
    Provider paths must refer to the pinned original files, not staged transform
    outputs. The caller owns those provenance checks. Text output preserves the
    ledger's multiplicity and does not pick an original occurrence among identical
    provider rows. Candidate multiplicity is reported separately. Consumers must
    require ``_complete.json``; an interrupted or incomplete replay leaves no marker.

    Multirow transforms need a source-specific complete-input assembly stage and
    are rejected here. They can use ``reconstruct_selected_document`` after assembly.
    """
    if recipe.transform is ProviderTransform.BIODIVERSITY:
        raise ValueError("Biodiversity requires complete multi-file item assembly before reconstruction")
    if recipe.transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
        raise ValueError("Audited SWH empty filtering requires the bound parallel file/source audit stages")
    replay = provider_normalizer(recipe)
    output_fs, output_key = filesystem_for(output_dir)
    if output_fs.exists(output_key):
        raise FileExistsError(f"Use a fresh reconstruction directory: {output_dir}")
    columns = PROVIDER_COLUMNS.get(recipe.transform, [recipe.text_field, recipe.id_field])
    examined = 0
    emitted = 0
    pending: list[dict] = []
    reader_audits = []
    with tempfile.TemporaryDirectory(prefix="hero-reconstruction-") as temporary:
        with sqlite3.connect(Path(temporary) / "selected.sqlite") as index:
            index.execute("CREATE TABLE selected (id TEXT PRIMARY KEY, occurrences INTEGER, provider_matches INTEGER)")
            for path in selected_paths:
                fs, key = filesystem_for(path)
                with fs.open(key, "rb") as stream:
                    parquet = pq.ParquetFile(stream)
                    for batch in parquet.iter_batches(columns=["marin_source", "id"], batch_size=batch_size):
                        index.executemany(
                            "INSERT INTO selected VALUES (?, 1, 0) "
                            "ON CONFLICT(id) DO UPDATE SET occurrences=occurrences+1",
                            ((row["id"],) for row in batch.to_pylist() if row["marin_source"] == marin_source),
                        )
            index.commit()
            selected_count, distinct_ids = index.execute(
                "SELECT COALESCE(SUM(occurrences),0), COUNT(*) FROM selected"
            ).fetchone()
            if not selected_count:
                raise ValueError(f"Selected ledger has no occurrences for {marin_source}")
            output_fs.makedirs(output_key, exist_ok=True)
            text_path = prefix_join(output_key, "text.parquet")
            with (
                output_fs.open(text_path, "wb") as stream,
                pq.ParquetWriter(stream, RECONSTRUCTED_TEXT_SCHEMA) as writer,
            ):
                for path in provider_paths:
                    audit = DavinciCtxReadAudit()
                    originals = (
                        read_davinci_context(path, audit)
                        if recipe.transform is ProviderTransform.DAVINCI_CTX
                        else provider_rows(path, columns, batch_size)
                    )
                    for raw in originals:
                        examined += 1
                        for document in replay([raw]):
                            target = index.execute(
                                "SELECT occurrences, provider_matches FROM selected WHERE id=?", (document["id"],)
                            ).fetchone()
                            if target is None:
                                continue
                            index.execute(
                                "UPDATE selected SET provider_matches=provider_matches+1 WHERE id=?", (document["id"],)
                            )
                            count, matches = target
                            if matches:
                                continue
                            for _ in range(count):
                                pending.append(
                                    {"marin_source": marin_source, "id": document["id"], "text": document["text"]}
                                )
                                emitted += 1
                                if len(pending) >= batch_size:
                                    writer.write_table(pa.Table.from_pylist(pending, schema=RECONSTRUCTED_TEXT_SCHEMA))
                                    pending.clear()
                    if recipe.transform is ProviderTransform.DAVINCI_CTX:
                        reader_audits.append({"provider_path": path, **audit.report()})
                    index.commit()
                if pending:
                    writer.write_table(pa.Table.from_pylist(pending, schema=RECONSTRUCTED_TEXT_SCHEMA))
            missing_ids, missing_occurrences = index.execute(
                "SELECT COUNT(*), COALESCE(SUM(occurrences),0) FROM selected WHERE provider_matches=0"
            ).fetchone()
            ambiguous = index.execute("SELECT COUNT(*) FROM selected WHERE provider_matches>1").fetchone()[0]
            report = {
                "marin_source": marin_source,
                "recipe": asdict(recipe),
                "selected_paths": selected_paths,
                "provider_paths": provider_paths,
                "selected_occurrences": selected_count,
                "selected_distinct_ids": distinct_ids,
                "provider_rows_examined": examined,
                "reconstructed_occurrences": emitted,
                "missing_ids": missing_ids,
                "missing_occurrences": missing_occurrences,
                "ids_with_multiple_provider_matches": ambiguous,
                "historical_occurrence_provenance": "not_established",
                "text_path": prefix_join(output_dir, "text.parquet"),
            }
            if recipe.transform is ProviderTransform.DAVINCI_CTX:
                report["historical_provider_filter"] = reader_audits
            with output_fs.open(prefix_join(output_key, "report.json"), "wt") as stream:
                json.dump(report, stream, indent=2)
            if missing_occurrences or emitted != selected_count:
                raise ValueError(f"Original-provider replay left {missing_occurrences} selected occurrences unresolved")
            with output_fs.open(prefix_join(output_key, "_complete.json"), "wt") as stream:
                json.dump(report, stream, indent=2)
            return report
