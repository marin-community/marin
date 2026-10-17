"""Task #210: sha256 (and git blob sha1 for small files) of every object under one S3 prefix. Read-only.

Usage: hash_prefix.py <label> <s3 prefix>. Prints HASH <label> <relpath> <size> <sha256> <gitsha1|-> per file.
"""
import concurrent.futures as cf, hashlib, sys
import fsspec

label, src = sys.argv[1], sys.argv[2].rstrip("/").removeprefix("s3://")
fs = fsspec.filesystem("s3")
files = fs.find(src, detail=True)
CHUNK = 64 << 20


def h(path):
    info = files[path]
    s = hashlib.sha256()
    g = hashlib.sha1(f"blob {info['size']}\0".encode()) if info["size"] < (10 << 20) else None
    with fs.open(path, "rb", block_size=CHUNK) as f:
        while b := f.read(CHUNK):
            s.update(b)
            if g:
                g.update(b)
    return path, info["size"], s.hexdigest(), g.hexdigest() if g else "-"


with cf.ThreadPoolExecutor(12) as ex:
    for path, size, sha, gsha in ex.map(h, sorted(files)):
        print("HASH", label, path[len(src) + 1:], size, sha, gsha, flush=True)
print("HASHDONE", label, len(files), sum(i["size"] for i in files.values()), flush=True)
