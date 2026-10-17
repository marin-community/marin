"""Task #210: read-only search of small JSON/YAML/MD/TXT files under given prefixes for references to in-scope paths.

Reads files with 32 threads and prints progress every 2,000 files.
"""
import concurrent.futures as cf, re, sys
import fsspec

fs = fsspec.filesystem("s3")
PAT = re.compile(r"step92-fdc3c3c5|task1(33|45|47|57|65|66)-(cruxeval|e2|mixed)|users/karan/(skyrl|mopd2|cruxeval-opd)|teacher-traces/(cruxeval-transfer|cruxeval-test-exposed|mopd2/task16)")


def scan(path):
    return path, sorted({m.group(0) for m in PAT.finditer(fs.cat_file(path).decode("utf-8", "replace"))})


for root in sys.argv[1:]:
    files = [p for p, i in fs.find(root.removeprefix("s3://"), detail=True).items()
             if p.endswith((".json", ".yaml", ".yml", ".md", ".txt")) and i["size"] < (5 << 20)]
    print("LISTED", root, len(files), flush=True)
    with cf.ThreadPoolExecutor(32) as ex:
        for n, (path, hits) in enumerate(ex.map(scan, files), 1):
            if hits:
                print("REF", path, hits, flush=True)
            if n % 2000 == 0:
                print("PROGRESS", root, n, flush=True)
    print("SCANNED", root, len(files), flush=True)
