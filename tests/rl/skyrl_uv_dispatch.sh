#!/bin/sh
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -eu
python_preference=managed
while [ "$#" -gt 0 ]; do
    case "$1" in
        --python-preference) python_preference="$2"; shift 2 ;;
        --python)
            "$MARIN_TEST_UV_EXECUTABLE" python find "$2" --python-preference "$python_preference" --no-python-downloads >/dev/null
            shift 2
            ;;
        uv) shift; exec "$MARIN_TEST_UV_EXECUTABLE" "$@" ;;
        *) shift ;;
    esac
done
