# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only, bounded object-storage exploration for Marina."""

import csv
import io
import json
import math
import mimetypes
import tarfile
import zipfile
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from itertools import islice
from typing import Any
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from marina.applets import AppletServices
from pydantic import TypeAdapter, ValidationError
from rigging.filesystem.buckets import MissingCredentials, filesystem_for
from rigging.filesystem.paged_listing import is_child
from rigging.fsutil.compression import uncompressed_name
from rigging.fsutil.listing import entry_mtime, list_entries, parent_url, read_decompressed_preview

MAX_GROUP_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_MEDIA_BYTES = 10 * 1024 * 1024
MAX_TEXT_ROWS = 10000
MEDIA_TYPES = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/avif": "image",
    "application/pdf": "pdf",
    "audio/mpeg": "audio",
    "audio/wav": "audio",
    "audio/ogg": "audio",
    "audio/flac": "audio",
    "video/mp4": "video",
    "video/webm": "video",
}


def checked_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in ("s3", "gs") or not parsed.netloc or parsed.query or parsed.fragment:
        raise HTTPException(400, "Enter an s3://bucket/key or gs://bucket/key URL.")
    return url


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    # JavaScript cannot represent larger integers exactly.
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    return value


def parquet_preview(url: str, offset: int, limit: int, columns: list[str] | None) -> dict:
    fs, path = filesystem_for(url)
    with fs.open(path, "rb", block_size=256 * 1024, cache_type="readahead") as source:
        source.seek(-8, 2)
        tail = source.read(8)
        if tail[4:] != b"PAR1":
            raise HTTPException(422, "Invalid Parquet footer.")
        if int.from_bytes(tail[:4], "little") > MAX_RESPONSE_BYTES:
            raise HTTPException(413, "Parquet footer exceeds 4 MiB.")
        source.seek(0)
        parquet = pq.ParquetFile(source)
        metadata = parquet.metadata
        schema = [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow]
        selected = columns if columns is not None else parquet.schema_arrow.names
        if set(selected) - set(parquet.schema_arrow.names):
            raise HTTPException(400, "Unknown column selection.")
        rows = []
        start = 0
        notice = None
        for index in range(metadata.num_row_groups):
            group = metadata.row_group(index)
            end = start + group.num_rows
            if end <= offset:
                start = end
                continue
            if len(rows) >= limit:
                break
            chunk_bytes = sum(
                group.column(i).total_uncompressed_size
                for i in range(group.num_columns)
                if group.column(i).path_in_schema.split(".")[0] in selected
            )
            if chunk_bytes > MAX_GROUP_BYTES:
                if columns is not None:
                    raise HTTPException(413, "Selected columns exceed 32 MiB in this row group. Select fewer columns.")
                notice = "This row group exceeds 32 MiB. Use Columns to read a smaller selection."
                break
            local_offset = max(0, offset - start)
            for batch in parquet.iter_batches(batch_size=limit, row_groups=[index], columns=selected, use_threads=False):
                if local_offset >= batch.num_rows:
                    local_offset -= batch.num_rows
                    continue
                take = min(limit - len(rows), batch.num_rows - local_offset)
                rows.extend(batch.slice(local_offset, take).to_pylist())
                local_offset = 0
                if len(rows) >= limit:
                    break
            start = end
        return {
            "kind": "table",
            "format": "Parquet",
            "schema": schema,
            "columns": selected,
            "rows": json_value(rows),
            "total": metadata.num_rows,
            "offset": offset,
            "next_offset": offset + len(rows) if offset + len(rows) < metadata.num_rows else None,
            "row_groups": metadata.num_row_groups,
            "notice": notice,
        }


def text_preview(url: str, offset: int, limit: int) -> dict:
    preview = read_decompressed_preview(url)
    data = preview.data
    name = uncompressed_name(url).lower()
    if b"\x00" in data[:8192]:
        return {"kind": "binary", "format": "Binary", "text": data[:4096].hex(" "), "truncated": True}
    text = data.decode("utf-8", errors="replace")
    rows = None
    value: Any = None
    if name.endswith((".yaml", ".yml")) and not preview.truncated:
        value = yaml.safe_load(text)
    elif name.endswith((".jsonl", ".ndjson")):
        lines = text.splitlines()
        if preview.truncated:
            lines = lines[:-1]
        rows = [json.loads(line) for line in islice((line for line in lines if line.strip()), MAX_TEXT_ROWS + 1)]
    elif name.endswith(".json") and not preview.truncated:
        value = json.loads(text)
        if isinstance(value, list):
            rows = value
            value = None
    elif name.endswith((".csv", ".tsv")):
        lines = text.splitlines(keepends=True)
        if preview.truncated:
            lines = lines[:-1]
        rows = list(
            islice(
                csv.DictReader(io.StringIO("".join(lines)), delimiter="\t" if name.endswith(".tsv") else ","),
                MAX_TEXT_ROWS + 1,
            )
        )
    result = {
        "text": text[: MAX_RESPONSE_BYTES // 2],
        "truncated": preview.truncated or len(text) > MAX_RESPONSE_BYTES // 2,
        "kind": "text",
        "format": "Text",
    }
    if value is not None:
        result.update(
            kind="json", format="YAML" if name.endswith((".yaml", ".yml")) else "JSON", value=json_value(value)
        )
    if rows is not None:
        if len(rows) > MAX_TEXT_ROWS:
            rows = rows[:MAX_TEXT_ROWS]
            result["truncated"] = True
        rows = [row if isinstance(row, dict) else {"value": row} for row in rows]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        result.update(
            kind="table",
            format=name.rsplit(".", 1)[-1].upper(),
            columns=columns,
            rows=json_value(rows[offset : offset + limit]),
            total=len(rows),
            offset=offset,
            next_offset=offset + limit if offset + limit < len(rows) else None,
            schema=[{"name": key, "type": "inferred"} for key in columns],
        )
    return result


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()

    @api.exception_handler(MissingCredentials)
    async def missing_credentials(request, error):
        return Response(json.dumps({"detail": str(error)}), status_code=503, media_type="application/json")

    @api.exception_handler(FileNotFoundError)
    async def missing_file(request, error):
        return Response(
            json.dumps({"detail": "Object not found. Check the URL or open its parent folder."}),
            status_code=404,
            media_type="application/json",
        )

    @api.get("/browse")
    def browse(url: str = "", token: str | None = None) -> dict:
        if not url:
            return {"entries": json_value([asdict(entry) for entry in list_entries("")]), "next_token": None}
        checked_url(url)
        fs, path = filesystem_for(url)
        entries, next_token = fs.listing.page(path, token, "/")
        scheme = urlsplit(url).scheme
        return {
            "entries": [
                {
                    "url": f"{scheme}://{item['name']}",
                    "name": item["name"].rstrip("/").rsplit("/", 1)[-1],
                    "size": item.get("size"),
                    "mtime": json_value(entry_mtime(item)),
                    "is_dir": item["type"] == "directory",
                }
                for item in entries
                if is_child(path, item["name"])
            ],
            "next_token": next_token,
            "parent": parent_url(url),
        }

    @api.exception_handler(PermissionError)
    async def permission_denied(request, error):
        return Response(
            json.dumps({"detail": "Marina does not have permission to read this bucket or object."}),
            status_code=403,
            media_type="application/json",
        )

    @api.get("/inspect")
    def inspect(
        url: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), columns: str | None = None
    ) -> dict:
        checked_url(url)
        fs, path = filesystem_for(url)
        info = fs.info(path)
        if info["type"] == "directory":
            return {"kind": "directory"}
        name = url.lower()
        mime = mimetypes.guess_type(name)[0]
        result: dict[str, Any]
        try:
            selected_columns = TypeAdapter(list[str]).validate_json(columns) if columns is not None else None
        except ValidationError as error:
            raise HTTPException(400, "Columns must be a JSON array of column names.") from error
        if name.endswith(".parquet"):
            result = parquet_preview(url, offset, limit, selected_columns)
        elif name.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            if info.get("size", 0) > MAX_MEDIA_BYTES:
                raise HTTPException(413, "Archive exceeds the 10 MiB inspection limit.")
            data = fs.cat_file(path, start=0, end=MAX_MEDIA_BYTES + 1)
            if len(data) > MAX_MEDIA_BYTES:
                raise HTTPException(413, "Archive exceeds the 10 MiB inspection limit.")
            members = []
            if name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    members = [
                        {"name": item.filename, "size": item.file_size, "compressed_size": item.compress_size}
                        for item in archive.infolist()[:5000]
                    ]
            else:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r|*") as archive:
                    for item in archive:
                        members.append(
                            {"name": item.name, "size": item.size, "type": "folder" if item.isdir() else "file"}
                        )
                        if len(members) >= 5000:
                            break
            keys = list(members[0]) if members else ["name", "size"]
            result = {
                "kind": "table",
                "format": "Archive",
                "columns": keys,
                "schema": [{"name": key, "type": "archive metadata"} for key in keys],
                "rows": members[offset : offset + limit],
                "total": len(members),
                "offset": offset,
                "next_offset": offset + limit if offset + limit < len(members) else None,
                "truncated": len(members) == 5000,
            }
        elif mime is not None and mime in MEDIA_TYPES:
            kind = MEDIA_TYPES[mime]
            if info.get("size", 0) > MAX_MEDIA_BYTES:
                raise HTTPException(413, "Media exceeds the 10 MiB preview limit.")
            result = {"kind": kind, "format": kind.upper()}
        else:
            try:
                result = text_preview(url, offset, limit)
            except (ValueError, csv.Error) as error:
                raise HTTPException(422, f"Cannot parse file: {error}") from error
        if not name.endswith(".parquet") and selected_columns is not None and result["kind"] == "table":
            if set(selected_columns) - set(result["columns"]):
                raise HTTPException(400, "Unknown column selection.")
            result["columns"] = selected_columns
            result["rows"] = [{key: row.get(key) for key in selected_columns} for row in result["rows"]]
        result.update(url=url, size=info.get("size"), modified=json_value(entry_mtime(info)))
        if len(json.dumps(result).encode()) > MAX_RESPONSE_BYTES:
            raise HTTPException(413, "Preview exceeds 4 MiB. Select fewer columns or fewer rows.")
        return result

    @api.get("/media")
    def media(url: str) -> Response:
        checked_url(url)
        mime = mimetypes.guess_type(url)[0]
        if mime not in MEDIA_TYPES:
            raise HTTPException(400, "This media format is not supported.")
        fs, path = filesystem_for(url)
        if fs.size(path) > MAX_MEDIA_BYTES:
            raise HTTPException(413, "Media exceeds the 10 MiB preview limit.")
        data = fs.cat_file(path, start=0, end=MAX_MEDIA_BYTES + 1)
        if len(data) > MAX_MEDIA_BYTES:
            raise HTTPException(413, "Media exceeds the 10 MiB preview limit.")
        return Response(data, media_type=mime, headers={"X-Content-Type-Options": "nosniff"})

    return api
