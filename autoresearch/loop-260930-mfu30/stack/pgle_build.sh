#!/usr/bin/env bash
# Build a PGLE latency profile from a traced arm and commit it into this checkout.
#
#   pgle_build.sh <trace-run-id>
#
# Writes experiments/grug/moe_hero_ep/pgle/<trace-run-id>.pbtxt (force-added: the Iris bundle must
# carry it) and prints the XLA flag for the scored rerun. The profile matches HLO instruction
# names, so the rerun must be the same commit and the same XLA flags as the trace, plus this flag.
set -euo pipefail
RUN="${1:?trace run id}"
ROOT="$(git rev-parse --show-toplevel)"
REMOTE="marin-cw:hero-checkpoints/tmp/ttl=30d/xprof/${RUN}/plugins/profile"
XPLANE=$(rclone lsf -R "${REMOTE}" | grep 'xplane.pb$' | head -1)
[ -n "${XPLANE}" ] || { echo "no xplane.pb under ${REMOTE}" >&2; exit 1; }
TMP=$(mktemp -d)
rclone copyto "${REMOTE}/${XPLANE}" "${TMP}/trace.xplane.pb"
OUT="experiments/grug/moe_hero_ep/pgle/${RUN}.pbtxt"
mkdir -p "${ROOT}/experiments/grug/moe_hero_ep/pgle"
(cd "${ROOT}" && uv run python -m experiments.grug.moe_hero_ep.pgle_profile "${TMP}/trace.xplane.pb" "${OUT}")
rm -rf "${TMP}"
(cd "${ROOT}" && git add -f "${OUT}" && git commit -q -m "mfu30-stack: PGLE profile from ${RUN}")
echo "profile committed at $(cd "${ROOT}" && git rev-parse --short HEAD)"
echo "rerun flag: --xla_gpu_pgle_profile_file_or_directory_path=${OUT}"
