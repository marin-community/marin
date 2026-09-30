# Builder measurement recipe

This recipe lets a task builder collect the three clean-build and three resource
samples required by the construction contract with the staged Daytona helper. It
does not replace the independent semantic review: retain the raw helper output,
the helper call log, immutable inputs, and the small per-cell receipt so that a
reviewer can check the claim.

The commands below use `tools/daytona/dt.sh`, which is staged in every Daytona
enabled construction workspace. It runs `dt.py` through the worker's Daytona
interpreter. The CLI supports `snapshot build`, `sandbox create`, `upload`,
`download`, and `exec`; every call appends a JSON line to `DT_LOG`. `dt exec`
uses `--` to delimit one or more command words, so pass a compound command as
one quoted argument after that delimiter.

## Establish the input closure first

A clean build starts from only the immutable build inputs, not from an earlier
`task/`, Harbor package, bundle, virtual environment, or cache. Before running
any cell, write `evidence/clean-build-inputs.json` with the exact input paths,
SHA-256 values, declared build command, expected immutable output paths, and
the pinned build snapshot name plus its `dt snapshot get` JSON receipt. This is
task-specific: a builder must name the actual source files and toolchain needed
by its build command rather than copying its whole construction workspace.

Use a distinct directory containing just that closure for each cell. The local
side may prepare those directories from the same verified manifest, but each
remote cell must receive it into a new sandbox and build into a new empty output
directory. A terminal package is an expected-output oracle only; it must never
be uploaded as a build input.

For a task whose build requires a runtime not already present in a pinned build
snapshot, first make a task-owned, immutable Dockerfile and retain its SHA-256
and `snapshot build` receipt. Daytona snapshots have no Docker build context:
the Dockerfile cannot rely on `COPY` or `ADD`. Once the snapshot is built, all
three cells use that same named snapshot. The example assumes variables with
unique, task-owned values:

```bash
DT="$PWD/tools/daytona/dt.sh"
export DT_LOG="$PWD/task/evidence/dt_calls.jsonl"
SNAPSHOT="task-build-<unique-pinned-name>"
BUILD_INPUTS="$PWD/clean-inputs"       # no prior outputs or caches
REMOTE_ROOT=/work/clean-build
BUILD_COMMAND='cd /work/clean-build/input && uv run --frozen --offline build.py'
```

The build snapshot must contain the command and its declared offline dependency
closure. If the task needs network access to build, freeze downloaded source and
packages before these cells; a networked rebuild is not reproducible evidence.

## Three isolated build cells

Repeat this cell for `N=1`, `N=2`, and `N=3`, changing the sandbox purpose and
local receipt names. The helper's JSON is deliberately retained before parsing
the sandbox ID.

```bash
mkdir -p task/evidence/clean-build task/evidence/resource
"$DT" sandbox create --snapshot "$SNAPSHOT" --no-network \
  --purpose "clean-build-$N" >"task/evidence/clean-build/$N-create.json"
SANDBOX_ID="$(jq -er '.id' "task/evidence/clean-build/$N-create.json")"

"$DT" upload "$SANDBOX_ID" "$BUILD_INPUTS" "$REMOTE_ROOT/input" \
  >"task/evidence/clean-build/$N-upload.json"

"$DT" exec "$SANDBOX_ID" --cwd / --timeout 1800 --json -- \
  "set -eu
   rm -rf '$REMOTE_ROOT/output'
   mkdir -p '$REMOTE_ROOT/output'
   start_ns=\$(date +%s%N)
   $BUILD_COMMAND
   end_ns=\$(date +%s%N)
   printf 'build_start_ns=%s\\nbuild_end_ns=%s\\n' \"\$start_ns\" \"\$end_ns\"
   mkdir -p '$REMOTE_ROOT/receipt'
   if find '$REMOTE_ROOT/output' -type l -print -quit | grep -q .; then
     echo 'unexpected output symlink' >&2; exit 1
   fi
   find '$REMOTE_ROOT/output' -printf '%y %m %p\\n' | LC_ALL=C sort \
     >'$REMOTE_ROOT/receipt/output-tree.txt'
   find '$REMOTE_ROOT/output' -type f -exec sha256sum {} + | LC_ALL=C sort \
     >'$REMOTE_ROOT/receipt/output-hashes.txt'" \
  >"task/evidence/clean-build/$N-build.json"

"$DT" download "$SANDBOX_ID" "$REMOTE_ROOT/receipt/output-hashes.txt" \
  "task/evidence/clean-build/$N-output-hashes.txt"
"$DT" download "$SANDBOX_ID" "$REMOTE_ROOT/receipt/output-tree.txt" \
  "task/evidence/clean-build/$N-output-tree.txt"
"$DT" sandbox delete "$SANDBOX_ID" \
  >"task/evidence/clean-build/$N-delete.json"
```

Replace the illustrative `build.py` and output path with the task's actual build
command and output root. `output-hashes.txt` is a measurement receipt outside
that output root, so it does not alter the artifacts being compared.

The build receipt must record the create response's `create_s`, the measured
build start/end timestamps, the exact output file hash inventory, the output
tree's path types and modes, and the expected-output comparison. The raw `dt
exec --json` response includes the command's measured wall seconds. Hash the
downloaded receipts and the local
`DT_LOG`; copy the raw JSON files and those digests into private
`task/evidence/`. Treat a failed upload, command, download, or deletion as a
failed cell, never as an omitted detail.

`sha256sum` is an explicit build-snapshot dependency in this pattern. A builder
must preflight it (`command -v sha256sum`) or use a pinned equivalent and record
that choice. The staged helper does not provide a hashing subcommand.

## Resource sample within each fresh sandbox

Measure the task's actual startup and bounded reference/oracle workload in a
fresh sandbox, once per cell. Create a new sandbox and upload the required inputs
before the example below, and retain its create and delete receipts as above.
Keep startup and workload endpoints separate:
`create_s` measures sandbox provisioning; a task may additionally measure its
own readiness command between two wall-clock timestamps. The following compound
command produces a bounded text receipt from inside the sandbox. Substitute the
declared startup and workload commands and the task's trace directory.

```bash
"$DT" exec "$SANDBOX_ID" --cwd / --timeout 1800 --json -- \
  "set -eu
   root='$REMOTE_ROOT/input'
   trace_dir=\"\$root/task-trace\"
   ready_start_ns=\$(date +%s%N)
   <DECLARED_STARTUP_COMMAND>
   ready_end_ns=\$(date +%s%N)
   run_start_ns=\$(date +%s%N)
   <DECLARED_REFERENCE_OR_ORACLE_COMMAND>
   run_end_ns=\$(date +%s%N)
   memory_peak=unknown
   for p in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
     if [ -r \"\$p\" ]; then memory_peak=\$(cat \"\$p\"); break; fi
   done
   disk_kib=\$(du -sk \"\$root\" | awk '{print \$1}')
   trace_bytes=0
   if [ -d \"\$trace_dir\" ]; then
     trace_bytes=\$(find \"\$trace_dir\" -type f -exec sh -c 'for f; do wc -c <\"\$f\"; done' sh {} + | awk '{s+=\$1} END {print s+0}')
   fi
   printf 'ready_start_ns=%s\\nready_end_ns=%s\\nrun_start_ns=%s\\nrun_end_ns=%s\\nmemory_peak_bytes=%s\\nworkdir_disk_kib_end=%s\\ntrace_bytes=%s\\n' \
     \"\$ready_start_ns\" \"\$ready_end_ns\" \"\$run_start_ns\" \"\$run_end_ns\" \"\$memory_peak\" \"\$disk_kib\" \"\$trace_bytes\"" \
  >"task/evidence/resource/$N-exec.json"
```

Set a numeric budget before executing the cells and compare each reported value
to it. Record an unavailable cgroup file as `unknown`, not as zero. The normal
Daytona adapter independently collects the same cgroup locations at startup and
before deletion in `daytona-environment.json`; cite that controller receipt when
it corresponds to the measured workload, along with the builder's raw command
receipt.

## What these tools do and do not measure

* **Startup and runtime:** available. `sandbox create` returns `create_s`; `dt
  exec` returns `seconds`; task readiness and workload timestamps give narrower
  endpoints. These are wall-clock measurements, not CPU-time measurements.
* **Memory peak:** available when cgroup v2 `memory.peak` or cgroup v1
  `memory.max_usage_in_bytes` is readable. It is an observed sandbox cgroup
  value. The helper records requested CPU/memory/disk separately; requested
  capacity is not proof that the provider enforced that request.
* **Workdir disk at the end:** available with `du -sk`. This is an end-state
  measurement for the named directory.
* **Peak disk use:** not directly available from the staged helper or the
  Daytona receipts. A final `du` cannot establish a transient peak. A builder
  may add a bounded periodic `du` sampler to report an *observed sampled
  maximum*, but must label it as sampled and cannot call it an exact peak.
* **Trace size:** available when the task declares its trace directory or trace
  file set. Sum the bytes of that declared set; do not substitute helper log size
  or model tokens for task trace bytes.
* **No-tool tasks:** solver-side process/disk metrics can be inapplicable, with
  a concrete binding-based explanation. The private build and verifier work
  still require applicable measurements. A no-tool task can use the same
  isolated build cells because builder execution is not solver execution.

The helper does not create a generic build manifest, know a task's canonical
build command, hash arbitrary outputs, expose exact peak disk usage, or prove a
provider quota was enforced. Those are task-specific evidence obligations, not
claims that should be filled with a capacity request or a copied prior output.
