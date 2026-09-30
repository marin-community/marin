#!/usr/bin/env bash
# dt -- thin launcher for dt.py under the interpreter that has the Daytona SDK.
# The worker venv is built from the repo's declared deps and daytona is not one of them,
# so run_envgen.sh makes a private venv at $DT_PYTHON's location; locally, point DT_PYTHON
# at any interpreter with `daytona==0.200.2` installed.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${DT_PYTHON:-/tmp/dtenv/bin/python3}"
[ -x "$PY" ] || { printf 'dt: interpreter %s not found (set DT_PYTHON)\n' "$PY" >&2; exit 2; }
exec "$PY" "$HERE/dt.py" "$@"
