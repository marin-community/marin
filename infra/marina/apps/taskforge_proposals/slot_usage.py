# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize the token usage in a proposal run's completions files, as an Iris job.

The ``*.completions.json`` files hold every model completion of a plan or slot, stream events included,
and add up to far more than the viewer should read from a workstation. This job reads one shard of them
inside the storage region and writes one JSON line per file: the completion count and the summed usage.
Files a previous part already covers (same path, size and ETag) are skipped, so rerunning after the
run grows only reads the new files.

The completions files are pretty-printed with sorted keys, and stream events are stored as escaped
strings, so the completion-level ``usage`` and ``finish_reason`` keys are the only unescaped occurrences
at four spaces of indent. Each file's usage blocks are checked against its completion count.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import click
import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

DEFAULT_RUN_ROOT = "s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102/run"
DEFAULT_OUTPUT = "s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102/derived/slot-usage"
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")
COMPLETION_START = re.compile(rb"^  \{$", re.M)
USAGE = re.compile(rb'^    "usage": (\{.*?^    \})', re.M | re.S)
FINISH = re.compile(rb'^    "finish_reason": (null|"[^"]*")', re.M)
CONTINUATIONS = re.compile(rb'^    "continuations": (\d+)', re.M)
WALL_TIME = re.compile(rb'^    "wall_time": ([0-9.eE+-]+)', re.M)
SLOT_FILE = re.compile(r"slot-(\d+)\.completions\.json$")


def file_key(info: dict[str, Any]) -> str:
    return f"{info['name']}|{info['size']}|{info.get('ETag', '')}"


def summarize(data: bytes) -> dict[str, Any]:
    """Count the completions in one file and sum their usage."""
    completions = len(COMPLETION_START.findall(data))
    blocks = [json.loads(block) for block in USAGE.findall(data)]
    if len(blocks) != completions:
        raise ValueError(f"{completions} completions but {len(blocks)} usage blocks")
    totals = {name: sum((block or {}).get(name) or 0 for block in blocks) for name in USAGE_FIELDS}
    return {
        "completions": completions,
        **totals,
        "finish_reasons": [json.loads(value) for value in FINISH.findall(data)],
        "continuations": sum(int(value) for value in CONTINUATIONS.findall(data)),
        "wall_time": sum(float(value) for value in WALL_TIME.findall(data)),
    }


def covered(fs: fsspec.AbstractFileSystem, output: str) -> set[str]:
    keys = set()
    for path in fs.glob(f"{output}/*.jsonl"):
        for line in fs.cat(path).decode().splitlines():
            if line.strip():
                keys.add(json.loads(line)["key"])
    return keys


@click.command(help=__doc__)
@click.option("--run-root", default=DEFAULT_RUN_ROOT, show_default=True)
@click.option("--output", default=DEFAULT_OUTPUT, show_default=True)
@click.option("--shard", type=int, required=True)
@click.option("--shards", type=int, required=True)
@click.option("--threads", type=int, default=16, show_default=True)
@click.option("--stamp", required=True, help="Names this pass's part files, so later passes add files.")
def main(run_root: str, output: str, shard: int, shards: int, threads: int, stamp: str) -> None:
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    root = run_root.removeprefix("s3://").rstrip("/")
    output = output.removeprefix("s3://").rstrip("/")
    started = time.monotonic()
    listing = fs.find(root, detail=True)
    files = sorted(
        (info for path, info in listing.items() if path.endswith(".completions.json")), key=lambda i: i["name"]
    )
    if not files:
        raise SystemExit(f"no completions files under {run_root}; refusing to write an empty part")
    done = covered(fs, output)
    todo = [info for info in files if file_key(info) not in done][shard::shards]
    print(f"{len(files)} completions files, {len(done)} covered, shard {shard}/{shards} reads {len(todo)}", flush=True)

    def row(info: dict[str, Any]) -> dict[str, Any]:
        rel = info["name"][len(root) + 1 :]
        capability, name = rel.split("/", 1)
        match = SLOT_FILE.match(name)
        slot = int(match.group(1)) if match else 0
        record = {"key": file_key(info), "capability": capability, "slot": slot, "path": rel, "bytes": info["size"]}
        try:
            record.update(summarize(fs.cat(info["name"])))
        except (OSError, ValueError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"warning: {rel}: {record['error']}", flush=True)
        return record

    with ThreadPoolExecutor(threads) as pool:
        rows = list(pool.map(row, todo))
    errors = sum(1 for r in rows if "error" in r)
    if rows:
        body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows if "error" not in r)
        fs.pipe(f"{output}/part-{stamp}-{shard:04d}-of-{shards:04d}.jsonl", body.encode())
    print(
        f"wrote {len(rows) - errors} rows, {errors} errors, {sum(r['bytes'] for r in rows) / 1e9:.1f} GB "
        f"in {time.monotonic() - started:.0f}s",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
