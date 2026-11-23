#!/bin/bash
set -euo pipefail

WORKDIR=@@WORKDIR@@

cd "$WORKDIR"
EXCLUDES=(.gitmodules '*.tar.gz' '*.tgz' '*.zip' '*.rej' '*.snap' '**/snapshots/**' 'testdata/result/**')

@@APPLY_PATCH@@
cp /solution/gold.patch /tmp/gold.patch
apply_patch /tmp/gold.patch
