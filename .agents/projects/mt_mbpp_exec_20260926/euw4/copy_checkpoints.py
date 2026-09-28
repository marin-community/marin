"""One-time server-side copy of three OlmoBaseEval Easy 1e21 HF exports from us-east5 to europe-west4 (40.7 GB).

Calvin approved this transfer on 2026-09-26 because v6e-4 capacity in us-east5-b was exhausted while europe-west4-a
had free slices; only the checkpoints move, and the MT-MBPP request files are uploaded from the local frozen copy.
Each object is copied with the GCS rewrite API (no bytes pass through this machine), skipped if the destination
already matches, and verified by size and CRC32C. Progress goes to copy_checkpoints.log beside this file.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import fsspec

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plan_east5.json"
LOG = HERE / "copy_checkpoints.log"


def log(message: str) -> None:
    with LOG.open("a") as handle:
        handle.write(f"{datetime.now().strftime('%H:%M:%S')} {message}\n")


def identity(info: dict) -> tuple:
    return (int(info["size"]), info["crc32c"])


def main() -> None:
    fs = fsspec.filesystem("gs")
    rows = json.loads(PLAN.read_text())["rows"]
    jobs = []
    for row in rows:
        src = row["checkpoint_uri"].removeprefix("gs://")
        dst = src.replace("marin-us-east5/", "marin-eu-west4/", 1)
        for filename in row["checkpoint_files"]:
            jobs.append((f"{src}/{filename}", f"{dst}/{filename}"))
    total = sum(int(fs.info(s)["size"]) for s, _ in jobs)
    log(f"start: {len(jobs)} objects from {len(rows)} checkpoints, {total / 1e9:.2f} GB")
    done, t0 = 0, time.time()
    for src, dst in jobs:
        info = fs.info(src)
        if fs.exists(dst) and identity(fs.info(dst)) == identity(info):
            log(f"skip (already identical): {dst}")
        else:
            start = time.time()
            fs.copy(src, dst)
            log(f"copied {dst} ({int(info['size']) / 1e9:.2f} GB) in {time.time() - start:.0f} s")
        if identity(fs.info(dst)) != identity(info):
            log(f"FAILED: identity mismatch for {dst}")
            sys.exit(1)
        done += int(info["size"])
        rate = done / max(time.time() - t0, 1e-9)
        log(f"progress {done / 1e9:.2f}/{total / 1e9:.2f} GB; ETA {(total - done) / rate / 60:.1f} min")
    log("DONE: every object matches its source in size and CRC32C")


if __name__ == "__main__":
    main()
