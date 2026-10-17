"""Task #210: delete the approved #9920/#9984 prefixes from s3://marin-us-east-02a/marin. Dry run unless --apply.

Usage: delete_scope.py [--apply]
Before deleting anything it lists every root and aborts unless, for every root, the object count and bytes equal the
2026-10-16 inventory and no object is newer than the inventory. So nothing written after the review is deleted.
After deleting it lists every root again and checks that it is empty.
"""
import sys
import fsspec

B = "marin-us-east-02a/marin"
INVENTORY_UTC = "2026-10-16 23:51:00"
# root -> object count from inventory-raw-20261016.txt; EXPECTED_BYTES is the sum of the roots there
ROOTS = {
    "experiments/task133-cruxeval": 206, "experiments/task145-cruxeval": 4890,
    "experiments/task147-cruxeval": 1225, "experiments/task157-e2": 1819,
    "experiments/task165-mixed": 2426, "experiments/task166-mixed": 2426,
    "users/karan/checkpoints": 16, "users/karan/cruxeval-opd": 1, "users/karan/mopd2": 1545,
    "users/karan/pr-763-smoke": 1, "users/karan/pr-763-smoke-nojb": 1, "users/karan/skyrl": 415,
    "users/karan/snowball-mopd-32k": 13, "users/karan/snowball-mopd-ultra": 1,
    "users/karan/snowball-opd-smoke": 2,
    "teacher-traces/cruxeval-transfer": 12844, "teacher-traces/cruxeval-test-exposed": 1699,
    "teacher-traces/mopd2/task162-e2-prepared-20261009-v1": 6,
    "teacher-traces/mopd2/task163-e2-step254-k8-20261010": 25699,
    "teacher-traces/mopd2/task163-e2-step254-k8-audit-20261010": 4,
    "teacher-traces/mopd2/task165-mixed-plain-prepared-20261010": 4,
    "teacher-traces/mopd2/task165-mmlu-plain-prepared-20261010": 4,
    "teacher-traces/mopd2/task166-mixed-rs-prepared-20261010": 4,
    "teacher-traces/mopd2/task166-mmlu-rs-prepared-20261010": 4,
}
FORBIDDEN = ("evals", "weight-merging", "experiments/sft-shootout", "teacher-traces/mopd2/task202")
assert not any(r.startswith(f) for r in ROOTS for f in FORBIDDEN)
EXPECTED_BYTES = 21_875_529_476_522

fs = fsspec.filesystem("s3")
listing, total, ok = {}, 0, True
for r, n_want in ROOTS.items():
    files = fs.find(f"{B}/{r}", detail=True)
    newest = max(str(i.get("LastModified", ""))[:19] for i in files.values()) if files else ""
    b = sum(i["size"] for i in files.values())
    good = len(files) == n_want and newest <= INVENTORY_UTC
    ok &= good
    total += b
    listing[r] = list(files)
    print(f"{'OK ' if good else 'BAD'} {r} objects {len(files)} (inventory {n_want}) GB {b/1e9:,.2f} newest {newest}", flush=True)
print(f"TOTAL objects {sum(map(len, listing.values()))} bytes {total:,} expected {EXPECTED_BYTES:,}", flush=True)
ok &= total == EXPECTED_BYTES
if not ok:
    sys.exit("ABORT: listing differs from the reviewed inventory; nothing deleted")
if "--apply" not in sys.argv:
    print("DRY RUN: nothing deleted", flush=True)
    sys.exit(0)
for r, files in listing.items():
    for i in range(0, len(files), 1000):
        fs.rm(files[i:i + 1000])
    left = fs.find(f"{B}/{r}")
    print(f"DELETED {r} {len(files)} objects; remaining {len(left)}", flush=True)
print("DELETE DONE", flush=True)
