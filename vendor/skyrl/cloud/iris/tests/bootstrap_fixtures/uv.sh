#!/bin/sh

if [ -z "${EXPECTED_SYSTEM_PYTHON:-}" ]; then
  exit 0
fi

selected_python=
downloads_disabled=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --python) selected_python=$2; shift 2 ;;
    --no-python-downloads) downloads_disabled=true; shift ;;
    *) shift ;;
  esac
done
if [ "$selected_python" = "$EXPECTED_SYSTEM_PYTHON" ] && "$downloads_disabled"; then
  exit 0
fi
echo 'error: No interpreter found for Python 3.12.13' >&2
exit 2
