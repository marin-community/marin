# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join token-store Open-SWE rows to preserved outcomes using small metadata columns."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for

BOS = "<|begin_of_text|>"
EOS_SUFFIX = " <|end_of_text|>"
METADATA_COLUMNS = ("id", "source_id", "resolved")


@dataclass
class ColumnBudget:
    remaining: int

    def consume(self, size: int) -> None:
        if size > self.remaining:
            raise ValueError("Metadata-column read budget exhausted")
        self.remaining -= size


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(url: str, directory: Path, name: str) -> dict:
    path = directory / f"{name}.json"
    if not path.exists():
        fs, key = filesystem_for(url + "/.artifact.json")
        with fs.open(key, "rb") as stream:
            data = stream.read(262145)
        if len(data) > 262144:
            raise ValueError("Unexpectedly large artifact metadata")
        path.write_bytes(data)
    value = json.loads(path.read_text())
    if value["output_path"] != url:
        raise ValueError("Artifact output identity mismatch")
    return value


def dependency(value: dict) -> str:
    paths = value["dep_paths"]
    if len(paths) != 1:
        raise ValueError("Expected a single pinned source dependency")
    return paths[0]


def normalized_identity(source: dict) -> str:
    text = source["text"]
    if not text.startswith(BOS) or not text.endswith(EOS_SUFFIX):
        raise ValueError("Token-store framing differs from the verified SFT contract")
    return generate_id(text[len(BOS) : -len(EOS_SUFFIX)])


def column_bytes(parquet: pq.ParquetFile, group: int, columns: tuple[str, ...]) -> int:
    metadata = parquet.metadata.row_group(group)
    return sum(
        metadata.column(index).total_compressed_size
        for index in range(metadata.num_columns)
        if metadata.column(index).path_in_schema in columns
    )


def source_identity(source: dict, normalized_url: str, directory: Path, budget: ColumnBudget) -> dict:
    name = source["shard"].rsplit("/", 1)[1] + ".parquet"
    url = normalized_url + "/outputs/main/" + name
    fs, key = filesystem_for(url)
    offset = source["row"]
    with fs.open(key, "rb", block_size=65536) as stream:
        parquet = pq.ParquetFile(stream)
        for group in range(parquet.metadata.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if offset >= count:
                offset -= count
                continue
            size = column_bytes(parquet, group, ("id", "source_id"))
            budget.consume(size)
            rows = parquet.read_row_group(group, columns=["id", "source_id"]).to_pylist()
            identity = rows[offset]
            if identity["id"] != normalized_identity(source):
                raise ValueError("Normalized content hash differs; row position is not proof of identity")
            receipt = {"url": url, "row": source["row"], "row_group": group, "identity": identity}
            (directory / f"{source['id']}-identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return receipt
    raise ValueError("Source row outside normalized shard")


def outcome_rows(chat_url: str, targets: set[str], directory: Path, budget: ColumnBudget) -> dict[str, dict]:
    fs, prefix = filesystem_for(chat_url + "/outputs/main")
    found = {}
    for key in sorted(fs.glob(prefix + "/part-*.parquet")):
        url = chat_url + "/outputs/main/" + key.rsplit("/", 1)[1]
        with fs.open(key, "rb", block_size=65536) as stream:
            parquet = pq.ParquetFile(stream)
            if not set(METADATA_COLUMNS) <= set(parquet.schema_arrow.names):
                raise ValueError("Outcome-bearing source schema is missing required columns")
            row_offset = 0
            for group in range(parquet.metadata.num_row_groups):
                metadata = parquet.metadata.row_group(group)
                id_column = next(
                    metadata.column(i) for i in range(metadata.num_columns) if metadata.column(i).path_in_schema == "id"
                )
                stats = id_column.statistics
                possible = targets
                if stats is not None and stats.has_min_max:
                    possible = {key for key in targets if stats.min <= key <= stats.max}
                if possible:
                    budget.consume(column_bytes(parquet, group, METADATA_COLUMNS))
                    rows = parquet.read_row_group(group, columns=list(METADATA_COLUMNS)).to_pylist()
                    for index, row in enumerate(rows):
                        if row["id"] not in targets:
                            continue
                        if row["id"] in found:
                            raise ValueError("Ambiguous outcome identity")
                        found[row["id"]] = {"url": url, "row": row_offset + index, **row}
                row_offset += metadata.num_rows
    (directory / "matched-outcomes.json").write_text(json.dumps(found, indent=2) + "\n")
    return found


def run(manifest: Path, catalog: Path, root: Path, max_column_bytes: int) -> None:
    root.mkdir(parents=True, exist_ok=False)
    source_catalog = json.loads(catalog.read_text())
    grouped = {}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        source_file = Path(row["source_file"])
        source = json.loads(source_file.read_text())
        if source["source"].startswith("open_swe_traces/"):
            grouped.setdefault(source["source"], []).append((source_file, source))
    budget = ColumnBudget(max_column_bytes)
    output = []
    for source_name, sources in grouped.items():
        directory = root / source_name.replace("/", "--")
        directory.mkdir()
        store = artifact(source_catalog[source_name]["cache_path"], directory, "store")
        tokenized = artifact(dependency(store), directory, "tokenized")
        normalized_url = dependency(tokenized)
        normalized = artifact(normalized_url, directory, "normalized")
        rendered = artifact(dependency(normalized), directory, "rendered")
        chat_url = dependency(rendered)
        artifact(chat_url, directory, "chat")
        identities = {source["id"]: source_identity(source, normalized_url, directory, budget) for _, source in sources}
        targets = {identity["identity"]["source_id"] for identity in identities.values()}
        outcomes = outcome_rows(chat_url, targets, directory, budget)
        for source_file, source in sources:
            identity = identities[source["id"]]
            outcome = outcomes.get(identity["identity"]["source_id"])
            if outcome is None or outcome["resolved"] not in (-1, 0, 1):
                raise ValueError("Missing or unrecognized source outcome")
            disposition = {1: "eligible", 0: "excluded_unsuccessful", -1: "excluded_unknown"}[outcome["resolved"]]
            proof = {
                "id": source["id"],
                "source_file": str(source_file),
                "source_file_sha256": digest(source_file),
                "identity": identity,
                "outcome": outcome,
                "disposition": disposition,
                "evidence_files": {
                    str(directory / name): digest(directory / name)
                    for name in (
                        "store.json",
                        "tokenized.json",
                        "normalized.json",
                        "rendered.json",
                        "chat.json",
                        "matched-outcomes.json",
                        f"{source['id']}-identity.json",
                    )
                },
            }
            proof_path = directory / f"{source['id']}-eligibility.json"
            proof_path.write_text(json.dumps(proof, indent=2) + "\n")
            output.append(
                {"id": source["id"], "source": source_name, "disposition": disposition, "proof": str(proof_path)}
            )
    report = {
        "rows": output,
        "counts": dict(Counter(row["disposition"] for row in output)),
        "projected_column_bytes_read": max_column_bytes - budget.remaining,
        "text_or_messages_columns_read": False,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-column-bytes", type=int, required=True)
    args = parser.parse_args()
    run(args.manifest, args.catalog, args.root, args.max_column_bytes)


if __name__ == "__main__":
    main()
