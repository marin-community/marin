#!/usr/bin/env bash
# Prepare a CI test leg to build the native Rust extensions from source.
#
# The unified unit workflow runs this for legs flagged `setup: rust` (see
# infra/ci/select_tests.py). Those legs test code that a PR's Rust change
# touches; a prebuilt wheel would not carry that change, so the leg must build
# the extension from source instead (issue #7254).
#
# `rust_mode.py dev` repoints marin-dupekit-native and marin-finelog-server at
# their local rust/ trees. The leg's subsequent `uv run` (no --frozen)
# re-resolves and builds only the extension the tested package depends on; the
# other native package's crate is untouched. The Rust toolchain and cargo cache
# are provided by the workflow, not this script, so it stays a thin, reusable
# "switch to source build" step.
#
# Side effect: `uv run` rewrites uv.lock in the working tree to match the dev
# sources. That is expected and discarded with the ephemeral runner.
set -euo pipefail

python3 scripts/rust_mode.py dev
