#!/usr/bin/env bash
# Build PGLE latency profiles from a traced arm and commit them into this checkout.
#
#   pgle_build.sh <trace-run-id>
#
# Writes, force-added (the Iris bundle must carry them):
#   experiments/grug/moe_hero_ep/pgle/<trace-run-id>.pbtxt      plain profile (pgle_profile.py)
#   experiments/grug/moe_hero_ep/pgle/<trace-run-id>-d2h.pbtxt  plus A's end-of-step D2H patch
#     (autoresearch/loop-260930-mfu30/a/pgle_patch_d2h.py: each large optimizer-state D2H copy-start
#     costs the batch's serialized total, so the LHS starts the writebacks under optimizer compute)
# and prints the XLA flag for each. The profile matches HLO instruction names, so the rerun must be the
# same commit and the same XLA flags as the trace, plus the PGLE flag.
set -euo pipefail
RUN="${1:?trace run id}"
ROOT="$(git rev-parse --show-toplevel)"
REMOTE="marin-cw:hero-checkpoints/tmp/ttl=30d/xprof/${RUN}/plugins/profile"
XPLANE=$(rclone lsf -R "${REMOTE}" | grep 'xplane.pb$' | head -1)
[ -n "${XPLANE}" ] || { echo "no xplane.pb under ${REMOTE}" >&2; exit 1; }
TMP=$(mktemp -d)
rclone copyto "${REMOTE}/${XPLANE}" "${TMP}/trace.xplane.pb"
OUT="experiments/grug/moe_hero_ep/pgle/${RUN}.pbtxt"
OUT_D2H="experiments/grug/moe_hero_ep/pgle/${RUN}-d2h.pbtxt"
TK=autoresearch/loop-260930-mfu30
mkdir -p "${ROOT}/experiments/grug/moe_hero_ep/pgle"
cd "${ROOT}"
uv run python -m experiments.grug.moe_hero_ep.pgle_profile "${TMP}/trace.xplane.pb" "${OUT}"
uv run python "${TK}/tfop_dump.py" "${TMP}/trace.xplane.pb" "${TK}" "${TMP}/rows.pkl"
uv run python "${TK}/a/pgle_patch_d2h.py" "${TMP}/rows.pkl" "${OUT}" "${OUT_D2H}"
rm -rf "${TMP}"
git add -f "${OUT}" "${OUT_D2H}"
git commit -q -m "mfu30-stack: PGLE profiles from ${RUN} (plain and D2H-patched)"
echo "profiles committed at $(git rev-parse --short HEAD)"
echo "plain:       --xla_gpu_pgle_profile_file_or_directory_path=${OUT}"
echo "D2H-patched: --xla_gpu_pgle_profile_file_or_directory_path=${OUT_D2H}"
