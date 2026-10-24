"""In-pod probe: can this pod list and read a CoreWeave S3 prefix? Prints the endpoint used and counts."""

import os
import sys
import time

from rigging.filesystem.buckets import filesystem_for

for url in sys.argv[1:]:
    start = time.time()
    fs, path = filesystem_for(url)
    print("PROBE endpoint", getattr(fs, "client_kwargs", None), "CW_ENDPOINT", os.environ.get("CW_ENDPOINT"), flush=True)
    try:
        names = fs.ls(path.rstrip("/"))
        print("PROBE", url, "entries", len(names), "first", names[:2], f"{time.time() - start:.1f}s", flush=True)
    except Exception as exc:
        print("PROBE", url, "ERROR", type(exc).__name__, str(exc)[:300], f"{time.time() - start:.1f}s", flush=True)
