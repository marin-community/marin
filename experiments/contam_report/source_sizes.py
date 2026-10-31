"""List each hero source's normalized Parquet files and sizes."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.buckets import filesystem_for

from experiments.datakit.hero_data import MANIFEST_PREFIX, manifest_path

manifest = json.loads(manifest_path().read_text())
sources = {k.removeprefix("normalized/"): v for k, v in manifest.items() if k.startswith("normalized/")}


def files_of(item):
    name, rel = item
    url = f"{MANIFEST_PREFIX}/{rel}/outputs/main"
    fs, path = filesystem_for(url)
    found = fs.find(path, detail=True)
    files = sorted((p, d["size"]) for p, d in found.items() if p.endswith(".parquet"))
    proto = url.split("://")[0]
    return name, {"dir": url, "files": [[f"{proto}://{p}", s] for p, s in files]}


with ThreadPoolExecutor(8) as ex:
    out = dict(ex.map(files_of, sorted(sources.items())))
json.dump(out, open(sys.argv[1], "w"))
tot = sum(s for v in out.values() for _, s in v["files"])
print(len(out), "sources", sum(len(v["files"]) for v in out.values()), "files", tot / 1e12, "TB")
