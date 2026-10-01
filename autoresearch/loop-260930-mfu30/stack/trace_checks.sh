#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Pull a traced arm's profile and run the stack checks on it.
#
#   trace_checks.sh <run-id> [<out-dir>]
#
# Writes <out-dir>/{trace.xplane.pb,rows.pkl,opnames.pkl,anatomy.txt,checks.txt} (default out-dir:
# ./trace-<run-id>) and prints checks.txt: forward carry D2H stall, main-level copy schedule, exposed
# memcpy, and the scope anatomy header.
set -euo pipefail
RUN="${1:?run id}"
OUT="${2:-trace-${RUN}}"
ROOT="$(git rev-parse --show-toplevel)"
TK="${ROOT}/autoresearch/loop-260930-mfu30"
REMOTE="marin-cw:hero-checkpoints/tmp/ttl=30d/xprof/${RUN}/plugins/profile"
mkdir -p "${OUT}"
if [ ! -s "${OUT}/trace.xplane.pb" ]; then
  XPLANE=$(rclone lsf -R "${REMOTE}" | grep 'xplane.pb$' | head -1)
  [ -n "${XPLANE}" ] || { echo "no xplane.pb under ${REMOTE}" >&2; exit 1; }
  rclone copyto "${REMOTE}/${XPLANE}" "${OUT}/trace.xplane.pb"
fi
cd "${ROOT}"
[ -s "${OUT}/rows.pkl" ] || uv run python "${TK}/tfop_dump.py" "${OUT}/trace.xplane.pb" "${TK}" "${OUT}/rows.pkl"
[ -s "${OUT}/opnames.pkl" ] || uv run python "${TK}/hlo_opnames.py" "${OUT}/trace.xplane.pb" "${TK}" "${OUT}/opnames.pkl"
uv run python "${TK}/anatomy.py" "${OUT}/rows.pkl" "${OUT}/opnames.pkl" > "${OUT}/anatomy.txt"
{
  echo "== ${RUN}"
  head -1 "${OUT}/anatomy.txt"
  echo "-- forward carry D2H"
  uv run python "${TK}/stack/carry_stall.py" "${OUT}/rows.pkl"
  echo "-- main-level schedule (copies >= 5 GiB)"
  uv run python "${TK}/stack/copy_schedule.py" "${OUT}/trace.xplane.pb" 5
  echo "-- exposed memcpy"
  uv run python "${TK}/exposed_detail.py" "${OUT}/rows.pkl" "${OUT}/opnames.pkl" "${TK}" 2>/dev/null \
    | sed -n '/exposed memcpy by (stream/,/exposed memcpy by step decile/p' || true
  echo "-- XLA remat kernels"
  uv run python - "${OUT}/rows.pkl" <<'EOF'
import pickle, sys
d = pickle.load(open(sys.argv[1], "rb"))
rows = [r for r in d["rows"] if r[6] == "jit_train_step" and r[0].startswith("Stream #17") and ".remat" in r[5]]
n = len(d["launches"])
print(f"{len({r[5] for r in rows})} distinct clones, {sum(r[3] - r[2] for r in rows) / 1e12 / n:.3f} s/step")
EOF
} | tee "${OUT}/checks.txt"
