#!/bin/sh
# Assemble the frontend: the static app plus the data index build_data.py wrote to data/.
set -eu
cd "$(dirname "$0")"
test -f data/meta.json.gz || { echo "data/meta.json.gz is missing: run build_data.py first" >&2; exit 1; }
rm -rf dist
cp -R ../static dist
cp -R data dist/data
