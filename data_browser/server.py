import gzip
import io
import json

import fsspec
import zstandard as zstd
from flask import Flask, jsonify, request, send_from_directory
from pyarrow.parquet import ParquetFile

app = Flask(__name__, static_folder="build")

# Initialize fsspec GCS filesystem
fs = fsspec.filesystem("gcs")


def list_files(path: str) -> dict:
    """List all files in the given path."""
    files = fs.ls(path, detail=True, refresh=True)

    return {
        "type": "directory",
        "files": files,
    }


def read_json_file(path: str) -> dict:
    """Reads a JSON file."""
    MAX_BYTES = 100 * 1024 * 1024
    with fs.open(path) as f:
        raw = f.read(MAX_BYTES)  # Don't OOM on huge files
        if len(raw) == MAX_BYTES:
            return {
                "type": "json",
                "error": f"File too large (exceeded {MAX_BYTES} bytes)",
            }
        data = json.loads(raw)

    return {
        "type": "json",
        "data": data,
    }


def read_text_file(
    path: str, get_json: bool, offset: int, count: int, gzipped: bool = False, zstded: bool = False
) -> dict:
    """
    Reads a range of lines (offset to offset + count) from a text file (possibly compressed using gzip or zstd).
    Interpret each line as a JSON if `get_json` is set.
    """
    with fs.open(path, "rb") as f:
        # Unzip
        if gzipped:
            f = gzip.GzipFile(fileobj=f)
        if zstded:
            f = zstd.ZstdDecompressor().stream_reader(f)

        # Assume text (Unicode)
        f = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")

        # Seek to line `offset` (note: O(offset) time!)
        for _ in range(offset):
            if not f.readline():
                break

        # Read `count` lines
        items = []
        for _ in range(count):
            line = f.readline()
            if not line:
                break

            items.append(json.loads(line) if get_json else line)

    return {
        "type": "jsonl" if get_json else "text",
        "items": items,
    }


def read_parquet_file(path: str, offset: int, count: int) -> dict:
    """Reads a range of records (offset to offset + count) from a parquet file."""
    pf = ParquetFile(path)
    # Note: can make this more efficient by skipping the first offset without reading into memory
    rows = next(pf.iter_batches(batch_size=offset + count))[offset:]
    return {
        "type": "parquet",
        "items": json.loads(rows.to_pandas().to_json(orient="records")),
    }


@app.route("/api/view", methods=["GET"])
def view():
    path = request.args.get("path")
    offset = int(request.args.get("offset", 0))
    count = int(request.args.get("count", 1))

    try:
        if not path:
            return jsonify({"error": "Path not specified"}), 404
        if not fs.exists(path):
            return jsonify({"error": "Path does not exist"}), 404

        # Directory
        if fs.isdir(path):
            return jsonify(list_files(path))

        # jsonl files
        if path.endswith(".jsonl"):
            return jsonify(read_text_file(path=path, get_json=True, offset=offset, count=count))
        if path.endswith(".json.gz") or path.endswith(".ndjson.gz") or path.endswith(".jsonl.gz"):
            # json.gz is because Dolma files are named like this (should be jsonl.gz)
            return jsonify(read_text_file(path=path, get_json=True, offset=offset, count=count, gzipped=True))
        if path.endswith(".jsonl.zstd") or path.endswith(".jsonl.zst"):
            return jsonify(read_text_file(path=path, get_json=True, offset=offset, count=count, zstded=True))

        # parquet files (what Hugging Face datasets use)
        if path.endswith(".parquet"):
            return jsonify(read_parquet_file(path=path, offset=offset, count=count))

        # json files (.SUCCESS is used in Marin to keep track of progress)
        if path.endswith(".json") or path.endswith(".SUCCESS"):
            return jsonify(read_json_file(path))

        # Assume text file (treated as a list of lines)
        return jsonify(read_text_file(path=path, gzipped=False, get_json=False, offset=offset, count=count))

    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/")
@app.route("/view")
def serve():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
