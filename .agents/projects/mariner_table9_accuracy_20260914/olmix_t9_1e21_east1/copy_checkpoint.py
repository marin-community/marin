"""One-time server-side copy of the Olmix OlmoBaseEval Easy 1e21 HF export from us-east5 to us-east1.

Calvin approved this single checkpoint transfer on 2026-09-26; evaluation inputs are not copied. Each object is copied
with the GCS rewrite API (no bytes pass through this machine), skipped if the destination already matches, and verified
by size and CRC32C afterwards. Progress goes to copy_checkpoint.log beside this file.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import fsspec

SRC = "marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_matched_olmix_scaling_v6e_20260910/olmixq_t9_kl0p005_cap04_1e21_seed662005-3f95f2/hf/step-22056"
DST = SRC.replace("marin-us-east5/", "marin-us-east1/", 1)
LOG = Path(__file__).with_name("copy_checkpoint.log")


def log(message: str) -> None:
    with LOG.open("a") as handle:
        handle.write(f"{datetime.now().strftime('%H:%M:%S')} {message}\n")


def identity(info: dict) -> tuple:
    return (int(info["size"]), info["crc32c"])


def main() -> None:
    fs = fsspec.filesystem("gs")
    objects = [o for o in fs.ls(SRC, detail=True) if o["type"] == "file"]
    total = sum(int(o["size"]) for o in objects)
    log(f"start: {len(objects)} objects, {total / 1e9:.2f} GB, {SRC} -> {DST}")
    done, t0 = 0, time.time()
    for obj in objects:
        name = obj["name"].rsplit("/", 1)[1]
        dst = f"{DST}/{name}"
        if fs.exists(dst) and identity(fs.info(dst)) == identity(obj):
            log(f"skip (already identical): {name}")
        else:
            start = time.time()
            fs.copy(obj["name"], dst)
            log(f"copied {name} ({int(obj['size']) / 1e9:.2f} GB) in {time.time() - start:.0f} s")
        if identity(fs.info(dst)) != identity(obj):
            log(f"FAILED: identity mismatch for {name}")
            sys.exit(1)
        done += int(obj["size"])
        rate = done / max(time.time() - t0, 1e-9)
        log(f"progress {done / 1e9:.2f}/{total / 1e9:.2f} GB; ETA {(total - done) / rate / 60:.1f} min")
    log("DONE: every object matches its source in size and CRC32C")


if __name__ == "__main__":
    main()
