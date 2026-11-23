#!/bin/sh
set -eu

# Pier requires a public tests entrypoint when loading a local task. In
# separate-verifier mode the executable grader remains inside the verifier image.
exec python /tests/grader.py
