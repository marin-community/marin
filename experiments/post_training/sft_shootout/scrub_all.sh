#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Scrub every finished run under the given roots with scrub_run.py, one prefix per run, and print one
# "RESULT <prefix> <json>" line each. A run is a traces shard (or relay round) holding _SUCCESS, or any
# directory directly under generation/<kind>/. Unfinished shards are listed as "SKIP <prefix>".
#   bash experiments/post_training/sft_shootout/scrub_all.sh <traces or generation root>...
set -u
dir=experiments/post_training/sft_shootout
prefixes=$(uv run python - "$@" <<'EOF'
import sys
from rigging.filesystem.storage_path import StoragePath as P

for root in sys.argv[1:]:
    root = root.rstrip("/")
    for kind in sorted(str(path).rstrip("/") for path in P(root).ls() if path.isdir()):
        for run in sorted(str(path).rstrip("/") for path in P(kind).ls() if path.isdir()):
            rounds = [f"{run}/round-{r}" for r in range(1, 6)] if "-relay" in kind else [run]
            for prefix in rounds:
                if "/generation/" in prefix or P(f"{prefix}/_SUCCESS").exists():
                    print("RUN", prefix)
                elif P(prefix).exists():
                    print("SKIP", prefix)
EOF
)
echo "$prefixes" | grep '^SKIP'
for prefix in $(echo "$prefixes" | awk '$1 == "RUN" {print $2}'); do
  echo "RESULT $prefix $(uv run python $dir/scrub_run.py "$prefix" | tail -1)"
done
