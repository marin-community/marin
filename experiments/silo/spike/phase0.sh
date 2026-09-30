#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Phase 0 feasibility spike for the Iris-backed sandbox provider.
#
# Answers the seven yes/no questions that gate the whole design. Runs inside a
# privileged Iris job on cw-us-east-02a and emits a JSON receipt on stdout and
# into $IRIS_OUTPUT_DIR (federated logs vanish at terminal state, so the receipt
# has to survive somewhere else too).
#
# Every probe is wrapped: a failing probe records a false answer, it does not
# abort the run. We want all seven answers from one job, not the first failure.

set -u -o pipefail

OUT="${IRIS_OUTPUT_DIR:-/tmp}/phase0-receipt.json"
WORK=/tmp/silo-spike
mkdir -p "$WORK"
RESULTS="$WORK/results"
: > "$RESULTS"

log() { printf '[spike] %s\n' "$*" >&2; }

# record KEY VALUE  -- VALUE is emitted as a raw JSON scalar
record() { printf '%s\t%s\n' "$1" "$2" >> "$RESULTS"; }
# rstr KEY STRING -- emitted as a JSON string, newlines and quotes escaped
rstr() {
  local v="$2"
  v="${v//\\/\\\\}"; v="${v//\"/\\\"}"; v="${v//$'\n'/\\n}"; v="${v//$'\t'/\\t}"; v="${v//$'\r'/}"
  printf '%s\t"%s"\n' "$1" "$v" >> "$RESULTS"
}
yn() { if [ "$1" -eq 0 ]; then echo true; else echo false; fi; }
now_ms() { date +%s%3N 2>/dev/null || echo 0; }

log "starting; uname=$(uname -a)"
rstr kernel "$(uname -r)"
rstr arch "$(uname -m)"
rstr hostname "$(hostname)"
rstr whoami "$(id -u):$(id -g)"

############################################################################
# Q1 — is the container actually privileged?
############################################################################
CAP_EFF=$(awk '/^CapEff/{print $2}' /proc/self/status 2>/dev/null || echo "")
rstr cap_eff "$CAP_EFF"
# A privileged container holds the full bounding set. The exact width varies by
# kernel, so test a capability we care about rather than matching a constant.
CAP_BND=$(awk '/^CapBnd/{print $2}' /proc/self/status 2>/dev/null || echo "")
rstr cap_bnd "$CAP_BND"
# CAP_SYS_ADMIN is bit 21.
have_sys_admin=1
if [ -n "$CAP_EFF" ]; then
  # shellcheck disable=SC2004
  if [ $(( 0x$CAP_EFF >> 21 & 1 )) -eq 1 ]; then have_sys_admin=0; fi
fi
record q1_privileged "$(yn $have_sys_admin)"
log "Q1 privileged(CAP_SYS_ADMIN)=$(yn $have_sys_admin)"

############################################################################
# Q2 — cgroup v2 delegation: can we create a child cgroup and set limits?
#      THE highest-risk item. Without it nested cgroup limits are unenforceable.
############################################################################
CG=/sys/fs/cgroup
cg_version=unknown
[ -f "$CG/cgroup.controllers" ] && cg_version=v2
[ -d "$CG/cpu" ] && [ ! -f "$CG/cgroup.controllers" ] && cg_version=v1
rstr cgroup_version "$cg_version"
rstr cgroup_mount "$(awk '$2=="/sys/fs/cgroup"{print $3" "$4}' /proc/self/mounts | head -1)"
rstr cgroup_controllers "$(cat $CG/cgroup.controllers 2>/dev/null || echo '')"

cg_writable=1; cg_subtree=1; cg_setlimit=1
TESTCG="$CG/silo-spike"
if mkdir -p "$TESTCG" 2>/dev/null; then
  cg_writable=0
  # Delegating cpu+memory to children requires writing the PARENT's subtree_control.
  if echo "+cpu +memory" > "$CG/cgroup.subtree_control" 2>/dev/null; then cg_subtree=0; fi
  if echo "400000 100000" > "$TESTCG/cpu.max" 2>/dev/null \
     && echo "1073741824" > "$TESTCG/memory.max" 2>/dev/null; then
    cg_setlimit=0
    rstr cgroup_child_cpu_max "$(cat "$TESTCG/cpu.max" 2>/dev/null)"
    rstr cgroup_child_memory_max "$(cat "$TESTCG/memory.max" 2>/dev/null)"
  fi
  rmdir "$TESTCG" 2>/dev/null
fi
record q2_cgroup_writable "$(yn $cg_writable)"
record q2_cgroup_subtree_control "$(yn $cg_subtree)"
record q2_cgroup_child_limits "$(yn $cg_setlimit)"
rstr cgroup_self_cpu_max "$(cat $CG/cpu.max 2>/dev/null || echo '')"
rstr cgroup_self_memory_max "$(cat $CG/memory.max 2>/dev/null || echo '')"
log "Q2 cgroup writable=$(yn $cg_writable) subtree=$(yn $cg_subtree) limits=$(yn $cg_setlimit)"

############################################################################
# Q3 — namespaces, mounts, overlayfs, /dev/fuse
############################################################################
unshare -Urm true >/dev/null 2>&1; record q3_unshare_userns "$(yn $?)"
unshare -m true   >/dev/null 2>&1; record q3_unshare_mountns "$(yn $?)"
[ -e /dev/fuse ]; record q3_dev_fuse "$(yn $?)"

ov=1
mkdir -p "$WORK/ov"/{lower,upper,work,merged}
echo hello > "$WORK/ov/lower/f"
if mount -t overlay overlay \
      -o "lowerdir=$WORK/ov/lower,upperdir=$WORK/ov/upper,workdir=$WORK/ov/work" \
      "$WORK/ov/merged" 2>"$WORK/ov.err"; then
  if [ "$(cat "$WORK/ov/merged/f" 2>/dev/null)" = "hello" ]; then ov=0; fi
  umount "$WORK/ov/merged" 2>/dev/null
fi
record q3_overlayfs "$(yn $ov)"
rstr overlayfs_error "$(head -c 300 "$WORK/ov.err" 2>/dev/null || echo '')"
rstr rootfs_fstype "$(stat -f -c %T / 2>/dev/null || echo unknown)"
rstr tmp_fstype "$(stat -f -c %T /tmp 2>/dev/null || echo unknown)"
rstr disk_avail "$(df -h /tmp 2>/dev/null | awk 'NR==2{print $2" total, "$4" avail"}')"
log "Q3 unshare/overlay/fuse done"

############################################################################
# Fetch a container runtime. nerdctl-full bundles containerd + runc + buildkit
# + CNI in one tarball, which is the smallest thing that gives us pull, run,
# exec, cgroups and (later) builds. runsc comes from gVisor's own release bucket.
############################################################################
export PATH="$WORK/bin:$PATH"
mkdir -p "$WORK/bin"
NERDCTL_VERSION=2.1.2
GVISOR_URL=https://storage.googleapis.com/gvisor/releases/release/latest/x86_64

t0=$(now_ms)
curl -fsSL -o "$WORK/nerdctl-full.tgz" \
  "https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/nerdctl-full-${NERDCTL_VERSION}-linux-amd64.tar.gz" \
  2>"$WORK/nerdctl.err"
nerdctl_dl=$?
t1=$(now_ms)
record fetch_nerdctl_ok "$(yn $nerdctl_dl)"
record fetch_nerdctl_ms "$((t1-t0))"
if [ $nerdctl_dl -eq 0 ]; then
  mkdir -p "$WORK/nerdctl"
  tar -C "$WORK/nerdctl" -xzf "$WORK/nerdctl-full.tgz" 2>/dev/null
  cp "$WORK/nerdctl"/bin/* "$WORK/bin/" 2>/dev/null
  rstr nerdctl_version "$(nerdctl --version 2>&1 | head -1)"
  rstr containerd_version "$(containerd --version 2>&1 | head -1)"
else
  rstr nerdctl_fetch_error "$(head -c 300 "$WORK/nerdctl.err")"
fi

t0=$(now_ms)
runsc_dl=1
if curl -fsSL -o "$WORK/bin/runsc" "$GVISOR_URL/runsc" 2>/dev/null \
   && curl -fsSL -o "$WORK/bin/containerd-shim-runsc-v1" "$GVISOR_URL/containerd-shim-runsc-v1" 2>/dev/null; then
  chmod 0755 "$WORK/bin/runsc" "$WORK/bin/containerd-shim-runsc-v1"
  runsc_dl=0
  rstr runsc_version "$(runsc --version 2>&1 | head -1)"
fi
t1=$(now_ms)
record fetch_runsc_ok "$(yn $runsc_dl)"
record fetch_runsc_ms "$((t1-t0))"

############################################################################
# Start containerd. Keep every path under /tmp (an emptyDir) so we are not
# fighting the pod's own overlay root.
############################################################################
mkdir -p "$WORK/containerd/"{root,state} /etc/containerd
cat > /etc/containerd/config.toml <<EOF
version = 2
root = "$WORK/containerd/root"
state = "$WORK/containerd/state"
[grpc]
  address = "$WORK/containerd/containerd.sock"
[plugins."io.containerd.grpc.v1.cri"]
  disable_cgroup = false
[plugins."io.containerd.runtime.v1.linux"]
  no_shim = false
[proxy_plugins]
EOF
cat >> /etc/containerd/config.toml <<'EOF'
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runsc]
  runtime_type = "io.containerd.runsc.v1"
EOF

containerd --config /etc/containerd/config.toml > "$WORK/containerd.log" 2>&1 &
CONTAINERD_PID=$!
export CONTAINERD_ADDRESS="$WORK/containerd/containerd.sock"
NERDCTL="nerdctl --address $CONTAINERD_ADDRESS --namespace silo"

cd_ready=1
for _ in $(seq 1 30); do
  if $NERDCTL info >/dev/null 2>&1; then cd_ready=0; break; fi
  sleep 1
done
record containerd_started "$(yn $cd_ready)"
rstr containerd_log_tail "$(tail -c 600 "$WORK/containerd.log" 2>/dev/null)"
rstr containerd_snapshotter "$($NERDCTL info 2>/dev/null | awk -F': *' '/Snapshotter/{print $2}' | head -1)"
log "containerd started=$(yn $cd_ready)"

############################################################################
# Q7 — image pull, cold vs warm. Also: is our own registry reachable?
############################################################################
IMG="docker.io/library/alpine:3.20"
pull_ok=1; pull_ms=0
if [ $cd_ready -eq 0 ]; then
  t0=$(now_ms); $NERDCTL pull -q "$IMG" >"$WORK/pull.log" 2>&1; pull_ok=$?; t1=$(now_ms)
  pull_ms=$((t1-t0))
fi
record q7_pull_ok "$(yn $pull_ok)"
record q7_pull_cold_ms "$pull_ms"
rstr q7_pull_error "$(tail -c 300 "$WORK/pull.log" 2>/dev/null)"

# Our registry: a 401 from /v2/ means reachable and requiring auth, which is the
# healthy answer. A connection error means the host cannot see it at all.
ENVREG=envreg.208261-marin-gpu.coreweave.app
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$ENVREG/v2/" 2>/dev/null || echo 000)
record q7_envreg_http_code "$code"
rstr q7_envreg_verdict "$([ "$code" = "401" ] && echo reachable_auth_required || ([ "$code" = "200" ] && echo reachable_anonymous || echo "unreachable_or_$code"))"

############################################################################
# Q5/Q6 — run a nested container and interrogate it.
#   run_probe RUNTIME LABEL
# Checks, inside the child: cgroup cpu.max / memory.max finite, and that the
# network is genuinely absent (DNS, raw TCP, HTTPS all must fail).
############################################################################
GUEST_PROBE='
echo "CPUMAX=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
echo "MEMMAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
echo "NPROC=$(nproc 2>/dev/null)"
echo "IFACES=$(ls /sys/class/net 2>/dev/null | tr "\n" "," )"
getent hosts github.com >/dev/null 2>&1 && echo "DNS=resolved" || echo "DNS=failed"
(timeout 5 sh -c "echo > /dev/tcp/1.1.1.1/443") >/dev/null 2>&1 && echo "TCP=connected" || echo "TCP=failed"
if command -v wget >/dev/null 2>&1; then
  timeout 8 wget -q -O /dev/null https://github.com >/dev/null 2>&1 && echo "HTTPS=ok" || echo "HTTPS=failed"
else
  echo "HTTPS=skipped_no_wget"
fi
'

run_probe() {
  local runtime="$1" label="$2" extra=""
  [ -n "$runtime" ] && extra="--runtime $runtime"
  local t0 t1 rc out
  t0=$(now_ms)
  # shellcheck disable=SC2086
  out=$($NERDCTL run --rm $extra \
        --network none \
        --cpus 2 --memory 1g \
        --name "silo-$label-$$" \
        "$IMG" /bin/sh -c "$GUEST_PROBE" 2>"$WORK/$label.err")
  rc=$?
  t1=$(now_ms)
  record "run_${label}_ok" "$(yn $rc)"
  record "run_${label}_ms" "$((t1-t0))"
  rstr "run_${label}_stderr" "$(tail -c 400 "$WORK/$label.err" 2>/dev/null)"
  rstr "run_${label}_raw" "$out"
  local k v
  while IFS='=' read -r k v; do
    [ -z "$k" ] && continue
    rstr "run_${label}_$(echo "$k" | tr 'A-Z' 'a-z')" "$v"
  done <<< "$out"
}

if [ $cd_ready -eq 0 ] && [ $pull_ok -eq 0 ]; then
  log "running runc probe"
  run_probe "" runc
  log "running warm runc probe (image already local)"
  run_probe "" runc_warm
  if [ $runsc_dl -eq 0 ]; then
    log "running runsc probe"
    run_probe runsc runsc
  fi
fi

############################################################################
# Q4 — gVisor cost on a syscall-heavy workload, measured against runc.
# A tar of many small files is syscall-bound, which is where runsc hurts most.
############################################################################
BENCH='
mkdir -p /b && cd /b
i=0; while [ $i -lt 2000 ]; do echo x > f$i; i=$((i+1)); done
tar czf /tmp/b.tgz . 2>/dev/null
echo BENCH_DONE
'
bench() {
  local runtime="$1" label="$2" extra="" t0 t1 rc
  [ -n "$runtime" ] && extra="--runtime $runtime"
  t0=$(now_ms)
  # shellcheck disable=SC2086
  $NERDCTL run --rm $extra --network none --cpus 2 --memory 1g \
      "$IMG" /bin/sh -c "$BENCH" >/dev/null 2>&1
  rc=$?
  t1=$(now_ms)
  record "bench_${label}_ok" "$(yn $rc)"
  record "bench_${label}_ms" "$((t1-t0))"
}
if [ $cd_ready -eq 0 ] && [ $pull_ok -eq 0 ]; then
  bench "" runc
  [ $runsc_dl -eq 0 ] && bench runsc runsc
fi

############################################################################
# Control: the HOST must have network. If the host also fails these, the
# sandbox result proves nothing about the sandbox boundary.
############################################################################
getent hosts github.com >/dev/null 2>&1; record host_dns "$(yn $?)"
curl -s -o /dev/null --max-time 10 https://github.com; record host_https "$(yn $?)"

kill $CONTAINERD_PID 2>/dev/null

############################################################################
# Emit the receipt.
############################################################################
mkdir -p "$(dirname "$OUT")"
{
  echo '{'
  echo '  "schema": "silo-phase0-spike-v1",'
  printf '  "taken_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  first=1
  while IFS=$'\t' read -r k v; do
    [ -z "$k" ] && continue
    [ $first -eq 0 ] && echo ','
    first=0
    printf '  "%s": %s' "$k" "$v"
  done < "$RESULTS"
  echo
  echo '}'
} > "$OUT"

echo "=== SILO PHASE0 RECEIPT ==="
cat "$OUT"
echo "=== END SILO PHASE0 RECEIPT ==="
