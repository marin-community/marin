#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Cross-cluster reachability: TCP connect to the kubelet port (10250, always
# listening) on a node of each other cluster. Prints REACHED or BLOCKED per target.
for target in "$@"; do
  if timeout 6 bash -c "echo > /dev/tcp/${target%:*}/${target#*:}" 2>/dev/null; then r=REACHED; else r=BLOCKED; fi
  echo "[xc] from $(hostname -I | awk '{print $1}') to $target: $r"
done
