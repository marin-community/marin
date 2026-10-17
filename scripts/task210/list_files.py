"""Task #210: file-level listing (size, mtime) of small or unclear prefixes. Read-only."""
import sys
import fsspec

fs = fsspec.filesystem("s3")
for p in sys.argv[1:]:
    p = p.rstrip("/").removeprefix("s3://")
    for path, info in sorted(fs.find(p, detail=True).items()):
        print("FILE", path, info["size"], str(info.get("LastModified", ""))[:19], flush=True)
print("LISTDONE", flush=True)
