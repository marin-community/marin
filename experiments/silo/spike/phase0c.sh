#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Phase 0c: exit-code and stderr fidelity of `nerdctl exec` vs `ctr task exec`.
#
# Width test 01 found nerdctl 2.1.2 exec collapsing every non-zero exit to 1 and
# appending `level=fatal msg="exec failed with exit code N"` to the command's
# stderr. This measures whether containerd's own `ctr task exec` propagates the
# code exactly and adds nothing, under runc and runsc, including stdin.
set -u -o pipefail
WORK=/tmp/silo-0c; mkdir -p "$WORK/bin"; export PATH="$WORK/bin:$PATH"
T="timeout 120"
out() { printf '[0c] %s\n' "$*"; }

curl -fsSL -o "$WORK/n.tgz" https://github.com/containerd/nerdctl/releases/download/v2.1.2/nerdctl-full-2.1.2-linux-amd64.tar.gz
mkdir -p "$WORK/n" && tar -C "$WORK/n" -xzf "$WORK/n.tgz" && cp "$WORK/n"/bin/* "$WORK/bin/"
RB=https://storage.googleapis.com/gvisor/releases/release/20260714.0/x86_64
(cd "$WORK/bin" && curl -fsSL -O "$RB/runsc" && chmod 0755 runsc)
mkdir -p "$WORK/c/"{root,state}
printf 'version = 2\nroot = "%s"\nstate = "%s"\n[grpc]\n  address = "%s"\n' "$WORK/c/root" "$WORK/c/state" "$WORK/c/sock" > "$WORK/c.toml"
containerd --config "$WORK/c.toml" > "$WORK/containerd.log" 2>&1 &
for _ in $(seq 1 30); do [ -S "$WORK/c/sock" ] && break; sleep 1; done
N="nerdctl --address $WORK/c/sock --namespace silo"
C="ctr --address $WORK/c/sock --namespace silo"
$N pull -q docker.io/library/alpine:3.20 >/dev/null 2>&1


for rt in runc runsc; do
  flag=""; [ "$rt" = runsc ] && flag="--runtime $WORK/bin/runsc"
  name="fid-$rt"
  # shellcheck disable=SC2086
  $T $N run -d $flag --network none --name "$name" docker.io/library/alpine:3.20 sleep 3600 >/dev/null 2>&1
  # containerd knows the container by nerdctl's full ID, not its --name.
  cid=$($N inspect --format '{{.ID}}' "$name")
  out "$rt container id: $cid"
  for code in 0 3 124 137; do
    $T $N exec "$name" /bin/sh -c "echo out; echo err >&2; exit $code" > "$WORK/o" 2> "$WORK/e"; rc=$?
    out "$rt nerdctl exit=$code -> rc=$rc stdout=$(tr '\n' '|' < "$WORK/o") stderr=$(tr '\n' '|' < "$WORK/e")"
    $T $C task exec --exec-id "x$code$$" "$cid" /bin/sh -c "echo out; echo err >&2; exit $code" > "$WORK/o" 2> "$WORK/e"; rc=$?
    out "$rt ctr     exit=$code -> rc=$rc stdout=$(tr '\n' '|' < "$WORK/o") stderr=$(tr '\n' '|' < "$WORK/e")"
  done
  # cwd, env via busybox-style `env`, and stdin
  $T $C task exec --exec-id "cwd$$" --cwd /tmp "$cid" /bin/sh -c 'pwd' > "$WORK/o" 2>&1; out "$rt ctr --cwd /tmp -> $(tr '\n' '|' < "$WORK/o") rc=$?"
  $T $C task exec --exec-id "env$$" "$cid" env A=1 /bin/sh -c 'echo A=$A' > "$WORK/o" 2>&1; out "$rt ctr env A=1 -> $(tr '\n' '|' < "$WORK/o")"
  head -c 1048576 /dev/urandom > "$WORK/blob"; want=$(sha256sum < "$WORK/blob" | cut -d' ' -f1)
  $T $C task exec --exec-id "in$$" "$cid" /bin/sh -c 'cat > /tmp/b' < "$WORK/blob"; rc=$?
  got=$($T $C task exec --exec-id "outb$$" "$cid" /bin/sh -c 'cat /tmp/b' | sha256sum | cut -d' ' -f1)
  out "$rt ctr stdin 1MiB rc=$rc sha_match=$([ "$want" = "$got" ] && echo yes || echo NO)"
  $T $N rm -f "$name" >/dev/null 2>&1
done
out "done"
