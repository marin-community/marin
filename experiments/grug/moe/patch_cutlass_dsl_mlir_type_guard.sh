#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PYTHON [CUTLASS_PACKAGE]" >&2
  exit 2
fi

python="$1"
cutlass_package="${2:-}"
if [[ -z "$cutlass_package" ]]; then
  cutlass_package="$("$python" -c \
    'from pathlib import Path; import cutlass; print(Path(cutlass.__file__).parent)')"
fi
private_marker="$cutlass_package/.marin_private_copy"
arith="$cutlass_package/_mlir/dialects/arith.py"

if [[ ! -f "$arith" ]] || ! grep -Fq "def _isa(" "$arith"; then
  exit 0
fi
if grep -Fq "return isinstance(obj, cls)" "$arith"; then
  exit 0
fi

if [[ ! -e "$private_marker" ]]; then
  private_package="${cutlass_package}.private.$$"
  shared_package="${cutlass_package}.shared"
  test ! -e "$shared_package"
  cp -aL "$cutlass_package" "$private_package"
  mv "$cutlass_package" "$shared_package"
  mv "$private_package" "$cutlass_package"
  touch "$private_marker"
fi

patch --batch --forward -p0 -d "$(dirname "$cutlass_package")" \
  < "$(dirname "$0")/cutlass_dsl_mlir_type_guard.patch"
grep -Fq "return isinstance(obj, cls)" "$arith"
