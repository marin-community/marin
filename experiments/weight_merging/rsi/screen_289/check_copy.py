"""Check a verified S3 copy of a merge against the merge's own manifest (sha256 and size per object).

Usage: check_copy.py MERGE_MANIFEST_JSON COPY_COMPLETION_MANIFEST_JSON
(the completion manifest is `<dest>/.verified-copy-manifest.json`, written by `fsutil verified-copy`).
"""

import json
import sys


def main():
    merge = json.load(open(sys.argv[1]))
    copy = json.load(open(sys.argv[2]))
    expected = {o["name"]: (o["bytes"], o["sha256"]) for o in merge["objects"]}
    copied = {}
    for entry in copy["files"]:
        copied[entry["path"].lstrip("/")] = (entry["size"], entry["sha256"])
    missing = sorted(set(expected) - set(copied))
    mismatched = sorted(name for name in expected if name in copied and copied[name] != expected[name])
    extra = sorted(set(copied) - set(expected))
    print(
        json.dumps(
            dict(
                expected=len(expected),
                matched=len(expected) - len(missing) - len(mismatched),
                missing=missing,
                mismatched=mismatched,
                extra_not_in_manifest=extra,
            ),
            indent=2,
        )
    )
    sys.exit(1 if missing or mismatched else 0)


if __name__ == "__main__":
    main()
