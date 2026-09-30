#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Phase 0b: the gVisor half of the spike, plus the measurements the first run
# could not make cleanly.
#
#   Q4   runsc as a nested runtime, and its cost against runc
#   Q5'  isolation proof with a POSITIVE CONTROL: same image, same probes, only
#        --network differs. A probe that cannot succeed proves nothing when it fails.
#   Q6'  cgroup telemetry as the guest sees it, under runc AND runsc
#   new  cpuset: does --cpuset-cpus make nproc report the allotment?
#   new  pure lifecycle latency: create / exec / stdin-upload / delete, warm image
set -u -o pipefail

OUT="${IRIS_OUTPUT_DIR:-/tmp}/phase0b-receipt.json"
WORK=/tmp/silo-spike
mkdir -p "$WORK/bin"
RESULTS="$WORK/results"; : > "$RESULTS"
export PATH="$WORK/bin:$PATH"

log() { printf '[spike] %s\n' "$*" >&2; }
record() { printf '%s\t%s\n' "$1" "$2" >> "$RESULTS"; printf '[rec] %s=%s\n' "$1" "$2" >&2; }
rstr() {
  local v="$2"
  v="${v//\\/\\\\}"; v="${v//\"/\\\"}"; v="${v//$'\n'/\\n}"; v="${v//$'\t'/\\t}"; v="${v//$'\r'/}"
  printf '%s\t"%s"\n' "$1" "$v" >> "$RESULTS"
  printf '[rec] %s="%s"\n' "$1" "${v:0:300}" >&2
}
yn() { if [ "$1" -eq 0 ]; then echo true; else echo false; fi; }
now_ms() { date +%s%3N; }

# ---- install nerdctl-full + pinned gVisor (sha512-verified) ----------------
NERDCTL_VERSION=2.1.2
RUNSC_VERSION=20260714.0   # same pin as lib/iris/.../gcp/worker_bootstrap.py
RB="https://storage.googleapis.com/gvisor/releases/release/${RUNSC_VERSION}/x86_64"

curl -fsSL -o "$WORK/nerdctl-full.tgz" \
  "https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/nerdctl-full-${NERDCTL_VERSION}-linux-amd64.tar.gz"
mkdir -p "$WORK/nerdctl" && tar -C "$WORK/nerdctl" -xzf "$WORK/nerdctl-full.tgz" && cp "$WORK/nerdctl"/bin/* "$WORK/bin/"

runsc_ok=1
(
  cd "$WORK/bin" \
  && curl -fsSL -O "$RB/runsc" && curl -fsSL -O "$RB/runsc.sha512" \
  && curl -fsSL -O "$RB/containerd-shim-runsc-v1" && curl -fsSL -O "$RB/containerd-shim-runsc-v1.sha512" \
  && sha512sum -c runsc.sha512 && sha512sum -c containerd-shim-runsc-v1.sha512 \
  && chmod 0755 runsc containerd-shim-runsc-v1
) > "$WORK/runsc-install.log" 2>&1 && runsc_ok=0
record runsc_installed "$(yn $runsc_ok)"
rstr runsc_install_log "$(tail -c 500 "$WORK/runsc-install.log")"
rstr runsc_version "$(runsc --version 2>&1 | head -2 | tr '\n' ' ')"

mkdir -p "$WORK/containerd/"{root,state}
cat > "$WORK/containerd.toml" <<EOF
version = 2
root = "$WORK/containerd/root"
state = "$WORK/containerd/state"
[grpc]
  address = "$WORK/containerd/containerd.sock"
EOF
containerd --config "$WORK/containerd.toml" > "$WORK/containerd.log" 2>&1 &
CPID=$!
T="timeout 150"
N="nerdctl --address $WORK/containerd/containerd.sock --namespace silo"
for _ in $(seq 1 30); do $N info >/dev/null 2>&1 && break; sleep 1; done

IMG=docker.io/library/alpine:3.20
$N pull -q "$IMG" >/dev/null 2>&1
record image_pulled "$(yn $?)"

RUNSC_RT=io.containerd.runsc.v1

# ---- isolation probe: every tool is busybox, so absence is not failure -----
# Each line is TOOL=verdict where verdict is one of ok|failed|missing.
PROBE='
p() { name=$1; shift; if ! command -v "$1" >/dev/null 2>&1; then echo "$name=missing"; return; fi
      if "$@" >/dev/null 2>&1; then echo "$name=ok"; else echo "$name=failed"; fi; }
echo "IFACES=$(ls /sys/class/net 2>/dev/null | tr "\n" ",")"
echo "ROUTES=$(cat /proc/net/route 2>/dev/null | tail -n +2 | wc -l)"
p DNS        nslookup -timeout=3 github.com
p TCP_IP     nc -w 3 1.1.1.1 443
p HTTP_IP    wget -q -T 3 -O /dev/null http://1.1.1.1/
p HTTPS_NAME wget -q -T 5 -O /dev/null https://github.com/
echo "CPUMAX=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo absent)"
echo "MEMMAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo absent)"
echo "MEMCUR=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo absent)"
echo "MEMPEAK=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo absent)"
echo "CGV1CPU=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo absent)"
echo "CGV1PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo absent)"
echo "CGV1MEMLIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo absent)"
echo "CGV1MEMUSAGE=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo absent)"
echo "CGV1MEMMAXUSAGE=$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo absent)"
echo "CGROOT=$(ls /sys/fs/cgroup 2>/dev/null | tr "\n" ",")"
echo "NPROC=$(nproc 2>/dev/null)"
'
parse_into() {  # parse_into PREFIX <<< output
  local prefix="$1" k v
  while IFS='=' read -r k v; do
    [ -z "$k" ] && continue
    rstr "${prefix}_$(echo "$k" | tr 'A-Z' 'a-z')" "$v"
  done
}
probe() {  # probe LABEL RUNTIME NETWORK [extra args...]
  local label="$1" rt="$2" net="$3"; shift 3
  local rtflag=""; [ -n "$rt" ] && rtflag="--runtime $rt"
  local out rc
  # shellcheck disable=SC2086
  out=$($T $N run --rm $rtflag --network "$net" --cpus 2 --memory 1g "$@" "$IMG" /bin/sh -c "$PROBE" 2>"$WORK/$label.err")
  rc=$?
  record "${label}_ok" "$(yn $rc)"
  rstr "${label}_stderr" "$(tail -c 300 "$WORK/$label.err")"
  parse_into "$label" <<< "$out"
}

if [ $runsc_ok -eq 0 ]; then
  RUNSC_RT="$WORK/bin/runsc"
  log "smoke: runsc (runc shim route) true"
  t0=$(now_ms)
  $T $N run --rm --runtime "$RUNSC_RT" --network none "$IMG" true > "$WORK/smoke.log" 2>&1
  rc=$?; t1=$(now_ms)
  record runsc_smoke_ok "$(yn $rc)"; record runsc_smoke_ms "$((t1-t0))"
  rstr runsc_smoke_log "$(tail -c 600 "$WORK/smoke.log")"
  [ $rc -ne 0 ] && runsc_ok=1
fi
log "isolation: runc, network none vs host"
probe iso_runc_none ""        none
probe iso_runc_host ""        host      # positive control: the probes CAN succeed
if [ $runsc_ok -eq 0 ]; then
  log "isolation: runsc, network none vs host"
  probe iso_runsc_none "$RUNSC_RT" none
  probe iso_runsc_host "$RUNSC_RT" host
fi

log "cpuset: does nproc follow --cpuset-cpus?"
probe cpuset_runc  ""        none --cpuset-cpus 0-1
[ $runsc_ok -eq 0 ] && probe cpuset_runsc "$RUNSC_RT" none --cpuset-cpus 0-1

# ---- lifecycle latency, warm image, 5 iterations per runtime ---------------
lifecycle() {  # lifecycle LABEL RUNTIME
  local label="$1" rt="$2" rtflag="" i t0 t1 id
  [ -n "$rt" ] && rtflag="--runtime $rt"
  local c=() e=() u=() d=()
  for i in 1 2 3 4 5; do
    id="lat-$label-$i-$$"
    t0=$(now_ms)
    # shellcheck disable=SC2086
    $T $N run -d $rtflag --network none --cpus 2 --memory 1g --name "$id" "$IMG" sleep 3600 >/dev/null 2>&1 || { log "create failed $id"; continue; }
    t1=$(now_ms); c+=($((t1-t0)))
    t0=$(now_ms); $T $N exec "$id" /bin/sh -c 'echo hi' >/dev/null 2>&1; t1=$(now_ms); e+=($((t1-t0)))
    t0=$(now_ms); head -c 1048576 /dev/urandom | $T $N exec -i "$id" /bin/sh -c 'cat > /tmp/up.bin' >/dev/null 2>&1; t1=$(now_ms); u+=($((t1-t0)))
    t0=$(now_ms); $T $N rm -f "$id" >/dev/null 2>&1; t1=$(now_ms); d+=($((t1-t0)))
  done
  rstr "lat_${label}_create_ms" "${c[*]:-}"
  rstr "lat_${label}_exec_ms"   "${e[*]:-}"
  rstr "lat_${label}_upload_1mb_ms" "${u[*]:-}"
  rstr "lat_${label}_delete_ms" "${d[*]:-}"
}
log "lifecycle latency"
lifecycle runc ""
[ $runsc_ok -eq 0 ] && lifecycle runsc "$RUNSC_RT"

# ---- upload/download integrity through exec stdin/stdout -------------------
integ() {
  local label="$1" rt="$2" rtflag=""
  local id="io-$label-$$"
  [ -n "$rt" ] && rtflag="--runtime $rt"
  head -c 4194304 /dev/urandom > "$WORK/blob"
  local want got
  want=$(sha256sum "$WORK/blob" | cut -d' ' -f1)
  # shellcheck disable=SC2086
  $T $N run -d $rtflag --network none --name "$id" "$IMG" sleep 3600 >/dev/null 2>&1
  $T $N exec -i "$id" /bin/sh -c 'cat > /tmp/blob' < "$WORK/blob" 2>/dev/null
  got=$($T $N exec "$id" /bin/sh -c 'cat /tmp/blob' 2>/dev/null | sha256sum | cut -d' ' -f1)
  $T $N rm -f "$id" >/dev/null 2>&1
  [ "$want" = "$got" ]; record "io_${label}_roundtrip_4mb_sha_match" "$(yn $?)"
}
integ runc ""
[ $runsc_ok -eq 0 ] && integ runsc "$RUNSC_RT"

# ---- Q4: cost. Syscall-bound (many small files) and CPU-bound (shell loop) ---
bench() {  # bench LABEL RUNTIME SCRIPT
  local label="$1" rt="$2" script="$3" rtflag="" t0 t1
  [ -n "$rt" ] && rtflag="--runtime $rt"
  local id="bench-$label-$$"
  # shellcheck disable=SC2086
  $T $N run -d $rtflag --network none --cpus 2 --memory 2g --name "$id" "$IMG" sleep 3600 >/dev/null 2>&1
  t0=$(now_ms); $T $N exec "$id" /bin/sh -c "$script" >/dev/null 2>&1; local rc=$?; t1=$(now_ms)
  $T $N rm -f "$id" >/dev/null 2>&1
  record "bench_${label}_ok" "$(yn $rc)"
  record "bench_${label}_ms" "$((t1-t0))"
}
SYSCALL='mkdir -p /b && cd /b && i=0; while [ $i -lt 5000 ]; do echo x > f$i; i=$((i+1)); done; tar czf /tmp/b.tgz . && rm -rf /b'
CPU='i=0; while [ $i -lt 3000000 ]; do i=$((i+1)); done'
log "bench"
bench syscall_runc ""  "$SYSCALL"
bench cpu_runc     ""  "$CPU"
if [ $runsc_ok -eq 0 ]; then
  bench syscall_runsc "$RUNSC_RT" "$SYSCALL"
  bench cpu_runsc     "$RUNSC_RT" "$CPU"
fi

kill $CPID 2>/dev/null

mkdir -p "$(dirname "$OUT")"
{
  echo '{'; echo '  "schema": "silo-phase0b-spike-v1",'
  printf '  "taken_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "node": "%s"' "$(hostname)"
  while IFS=$'\t' read -r k v; do [ -z "$k" ] && continue; printf ',\n  "%s": %s' "$k" "$v"; done < "$RESULTS"
  echo; echo '}'
} > "$OUT"
echo "=== SILO PHASE0B RECEIPT ==="; cat "$OUT"; echo "=== END SILO PHASE0B RECEIPT ==="
