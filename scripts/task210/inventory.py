"""Task #210: size inventory of #9920/#9984 experiment prefixes in s3://marin-us-east-02a.

Prints one line per directory node (up to MAX_DEPTH below each root): NODE <path> <bytes> <objects> <newest mtime>.
Read-only: lists objects, writes nothing.
"""
import collections, sys
import fsspec

B = "marin-us-east-02a/marin"
fs = fsspec.filesystem("s3")
MAX_DEPTH = 8

# Discover task*-* under experiments and every dir under users/karan and teacher-traces.
for d in ["experiments", "users/karan", "users/karan/skyrl", "users/karan/mopd2", "teacher-traces",
          "teacher-traces/cruxeval-transfer", "teacher-traces/cruxeval-test-exposed", "teacher-traces/mopd2"]:
    try:
        for e in fs.ls(f"{B}/{d}", detail=False):
            print("LS", d, e, flush=True)
    except FileNotFoundError:
        print("LS", d, "MISSING", flush=True)

roots = [f"experiments/{e.rsplit('/', 1)[1]}" for e in fs.ls(f"{B}/experiments", detail=False)
         if e.rsplit("/", 1)[1].startswith("task")]
roots += [f"users/karan/{e.rsplit('/', 1)[1]}" for e in fs.ls(f"{B}/users/karan", detail=False)]
roots += [f"teacher-traces/{x}" for x in ["cruxeval-transfer", "cruxeval-test-exposed", "mopd2"]]

for root in roots:
    agg = collections.defaultdict(lambda: [0, 0, ""])
    try:
        files = fs.find(f"{B}/{root}", detail=True)
    except Exception as ex:  # noqa: BLE001
        print("ERR", root, repr(ex), flush=True)
        continue
    for path, info in files.items():
        rel = path[len(B) + 1:]
        parts = rel.split("/")[:-1]
        mt = str(info.get("LastModified", ""))[:19]
        for i in range(1, min(len(parts), MAX_DEPTH + root.count("/") + 1) + 1):
            a = agg["/".join(parts[:i])]
            a[0] += info["size"]; a[1] += 1; a[2] = max(a[2], mt)
    for k in sorted(agg):
        if k.count("/") >= root.count("/"):
            b, n, mt = agg[k]
            print(f"NODE {k} {b} {n} {mt}", flush=True)
    print("DONE", root, len(files), flush=True)
print("ALLDONE", flush=True)
