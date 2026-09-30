"""Runs INSIDE a network-less silo sandbox: stream the rootfs out as numbered parts.

    python3 silo_rootfs_stream.py /tmp/capability-rootfs-capture/job.json

stdlib only; the guest may have nothing but python3, tar and gzip (the same floor
as the Daytona ``in_sandbox_capture.py``).  A silo sandbox has no network
interface, so nothing is uploaded from here and no credential or presigned URL
ever enters the sandbox.  ``tar --one-file-system | gzip`` is cut into fixed-size
parts under ``out_dir``; each part is written as ``part-NNNNN.tmp`` and renamed
to ``part-NNNNN`` only when complete.  The trusted harness downloads a part
through silo's file API, uploads it to object storage itself, and deletes it.
At most ``max_outstanding`` complete parts wait on disk at once, so the sandbox
never holds more than a few parts of the archive.  ``out_dir`` lives under
/tmp, which the reviewed capture excludes from the archive.

``result.json`` (also written tmp-then-rename) carries the same fields the
Daytona in-sandbox capture prints, minus ETags, which only exist harness-side.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

READ = 8 * 1024 * 1024


def _outstanding(out_dir):
    return sum(1 for name in os.listdir(out_dir) if name.startswith("part-") and not name.endswith(".tmp"))


def _atomic_write(path, data):
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    with open(sys.argv[1]) as stream:
        job = json.load(stream)
    out_dir = job["out_dir"]
    part_bytes = int(job["part_bytes"])
    max_outstanding = int(job["max_outstanding"])
    stall_seconds = float(job["stall_seconds"])
    level = str(int(job.get("gzip_level", 6)))
    os.makedirs(out_dir, exist_ok=True)

    tar = ["tar", "--one-file-system", "--numeric-owner", "-cf", "-", "-C", job.get("root", "/")]
    for pattern in job["excludes"]:
        tar.append("--exclude=" + pattern)
    tar.append(".")

    started = time.time()
    tar_stderr_path = os.path.join(out_dir, "tar.stderr")
    with open(tar_stderr_path, "wb") as tar_stderr:
        p1 = subprocess.Popen(tar, stdout=subprocess.PIPE, stderr=tar_stderr)
        p2 = subprocess.Popen(["gzip", "-" + level, "-c"], stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()

        digest = hashlib.sha256()
        compressed = 0
        parts = 0
        error = None
        buffer = bytearray()

        def emit(data):
            nonlocal parts, error
            deadline = time.time() + stall_seconds
            while _outstanding(out_dir) >= max_outstanding:
                if time.time() > deadline:
                    error = "consumer_stalled"
                    return False
                time.sleep(0.2)
            parts += 1
            _atomic_write(os.path.join(out_dir, f"part-{parts:05d}"), bytes(data))
            return True

        while error is None:
            chunk = p2.stdout.read(READ)
            if not chunk:
                break
            digest.update(chunk)
            compressed += len(chunk)
            buffer += chunk
            while len(buffer) >= part_bytes and error is None:
                if emit(buffer[:part_bytes]):
                    del buffer[:part_bytes]
        if error is None and (buffer or parts == 0):
            emit(buffer)
        if error is not None:
            for process in (p1, p2):
                try:
                    process.kill()
                except OSError:
                    pass
        gzip_exit = p2.wait()
        tar_exit = p1.wait()
    with open(tar_stderr_path, "rb") as stream:
        tar_err = stream.read().decode("utf-8", "replace")[-4000:]
    elapsed = max(time.time() - started, 1e-9)
    result = {
        "ok": error is None and gzip_exit == 0,
        "error": error,
        "tar_exit": tar_exit,
        "gzip_exit": gzip_exit,
        "parts": parts,
        "compressed_bytes": compressed,
        "sha256": digest.hexdigest(),
        "seconds": round(elapsed, 1),
        "mb_per_s": round(compressed / 1e6 / elapsed, 1),
        "bad_parts": {},
        "tar_stderr_tail": tar_err,
    }
    _atomic_write(os.path.join(out_dir, "result.json"), json.dumps(result).encode())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
