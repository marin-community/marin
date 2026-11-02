#!/bin/bash
# Iris state of the shard jobs, one line per shard from shards/progress-NNNN.json, totals, and the control file's
# value and age. A progress file older than 180 s is flagged SILENT; a control file older than 300 s is flagged DEAD.
#   status.sh [pilot|run] [JOB-PREFIX]   (defaults: run, tfprop-)
set -euo pipefail
source "$(dirname "$0")/env.sh"
SUB=${1:-run} P=${2:-tfprop-}
cd "$WT"
jobs=$(timeout 120 uv run --frozen iris --cluster=marin job list --prefix "/muchanem/$P" --limit 500 2>/dev/null) || jobs=""
echo "$jobs" | grep -E "/muchanem/$P" | awk '{print $1, $2}' || echo "no jobs under /muchanem/$P (or iris unreachable)"
uv run --frozen python3 - "$ROOT/$SUB" <<'EOF'
import json
import sys
import time
from datetime import datetime

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

SILENT = 180
DEAD = 300
configure_coreweave_s3()
fs, root = fsspec.core.url_to_fs(sys.argv[1])
now = time.time()
paths = sorted(fs.glob(f"{root}/shards/progress-*.json"))
if not paths:
    sys.exit(f"no progress files under {sys.argv[1]}/shards; nothing has started or the keys are wrong")
totals = {"done": 0, "failed": 0, "pending": 0, "in_flight": 0, "limit": 0, "requests": 0, "out_tokens": 0}
for path, raw in sorted(fs.cat(paths).items()):
    p = json.loads(raw)
    age = now - p["at"]
    flag = " SILENT" if age > SILENT else ""
    print(
        f"shard {p['shard']:>3} age {age:5.0f}s{flag} | caps {p['capabilities_done']} done {p['capabilities_failed']} "
        f"failed {p['capabilities_pending']} pending | in_flight {p['in_flight']}/{p['limit']} waiting "
        f"{p['waiting_on_limit']} window {p['capabilities_in_flight']}/{p['window']} | requests {p['requests']} out "
        f"{p['completion_tokens']} | failures {p['slot_failures']} repairs {p['repairs']} | control_age "
        f"{p['control_age']} halted {p['halted']}"
    )
    for key, field in (("done", "capabilities_done"), ("failed", "capabilities_failed"),
                       ("pending", "capabilities_pending"), ("in_flight", "in_flight"), ("limit", "limit"),
                       ("requests", "requests"), ("out_tokens", "completion_tokens")):
        totals[key] += p[field]
print("total", totals)
control = f"{root}/control.json"
if not fs.exists(control):
    print("control: MISSING (shards hold their launch --concurrency)")
else:
    c = json.loads(fs.cat_file(control))
    age = now - datetime.fromisoformat(c["written_at"]).timestamp()
    print(f"control: {c} age {age:.0f}s{' DEAD (shards hold)' if age > DEAD else ''}")
EOF
