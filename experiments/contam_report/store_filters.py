# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the per-source filter inputs of the hero store ``store_4d2e363d`` (Gate 5).

Reads the store's ``.artifact.json`` and the exact and verified fuzzy dedup
artifacts it was built from, and writes ``{source: {"dir", "decon", "exact",
"fuzzy"}}`` for `kept.py`: the normalized directory and the attribute
directories the store joined against it. ``fuzzy`` is null for sources the store
exempts from fuzzy dedup.

Usage: store_filters.py SOURCES_JSON OUT_JSON
"""

import json
import sys

from rigging.filesystem.buckets import filesystem_for

ROOT = "s3://marin-us-east-02a/marin"
STORE = f"{ROOT}/datakit/store_4d2e363d"
EXACT_DUPS = f"{ROOT}/datakit/repack_exact_dups/common-crawl-focus-2026-22_61d8bc42"
# The store depends on hero/verified_fuzzy_dups, a frozen step pinned to this path.
VERIFIED_FUZZY_DUPS = f"{ROOT}/datakit/verify_fuzzy_dups_v11_c075"
DECONTAM_PATHS = "experiments/datakit/hero_data_decontam_paths.json"
MAIN_OUTPUT = "outputs/main"


def artifact(path: str) -> dict:
    fs, fs_path = filesystem_for(f"{path}/.artifact.json")
    with fs.open(fs_path, "r") as f:
        return json.load(f)


def main() -> None:
    sources = json.load(open(sys.argv[1]))
    store = artifact(STORE)
    exempt = set(store["config"]["fuzzy_dedup_exempt"])
    store_deps = set(store["deps"])
    for dep in ("datakit/repack_exact_dups/common-crawl-focus-2026-22_61d8bc42", "hero/verified_fuzzy_dups_882d9bf0"):
        if dep not in store_deps:
            raise ValueError(f"store does not depend on {dep}")
    exact = artifact(EXACT_DUPS)["result"]["sources"]
    fuzzy = artifact(VERIFIED_FUZZY_DUPS)["result"]["sources"]
    decon = json.load(open(DECONTAM_PATHS))
    if set(sources) != set(store["result"]["source_names"]):
        raise ValueError("inventory sources differ from the store's sources")
    filters = {}
    for name, info in sorted(sources.items()):
        key = info["dir"].removeprefix(f"{ROOT}/")
        filters[name] = {
            "dir": info["dir"],
            "decon": f"{ROOT}/{decon[name]}/{MAIN_OUTPUT}",
            "exact": f"{ROOT}/{exact[key]['attr_dir']}",
            "fuzzy": None if name in exempt else f"{ROOT}/{fuzzy[key]['attr_dir']}",
        }
    json.dump(filters, open(sys.argv[2], "w"), indent=1)
    print(f"{len(filters)} sources, {sum(f['fuzzy'] is None for f in filters.values())} fuzzy-exempt")


if __name__ == "__main__":
    main()
