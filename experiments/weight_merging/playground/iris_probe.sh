#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
work_dir=$(mktemp -d /tmp/playground-probe-XXXXXX)
uv --no-config venv "$work_dir/io" --python 3.12
uv --no-config pip install --python "$work_dir/io/bin/python" 'marin-rigging @ git+https://github.com/marin-community/marin.git@c7c14a252dc2d6f7cd86136f99799960e1667e53#subdirectory=lib/rigging'
for endpoint in default https://cwobject.com; do
  "$work_dir/io/bin/python" - "$endpoint" "$PLAYGROUND_INPUT" <<'PY'
import os
import sys
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.transfer import copy_plan
endpoint, url = sys.argv[1:]
if endpoint != 'default':
    os.environ['CW_S3_ENDPOINT'] = endpoint
fs, path = filesystem_for(url)
print('endpoint', endpoint, 'client', fs.client_kwargs, flush=True)
for operation in ('ls', 'find', 'copy_plan'):
    try:
        if operation == 'ls':
            result = fs.ls(path)
        elif operation == 'find':
            result = fs.find(path)
        else:
            result = copy_plan([url], '/tmp/unused-probe', recursive=True, no_clobber=False).copies
        print(operation, 'count', len(result), flush=True)
    except Exception as exc:
        print(operation, 'error', type(exc).__name__, str(exc)[:200], flush=True)
PY
done
