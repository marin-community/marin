# Debugging log for EP25 MXFP8 CUTLASS environment

Make the shipped GPU bundle resolve a deterministic CUDA 13 CUTLASS DSL
toolchain and compile the vendored grouped MXFP8 kernels on GB200.

## Initial status

`ep25d2-mxfp8-numerics-20260725` failed after 21.21 seconds, before
numerical execution, with `CompilerDiagnosticError`: libNVVM could not compile
generated device IR for `sm_100a`.

The frozen Levanter GPU export from this worktree contained both CUDA runtime
variants:

```text
nvidia-cutlass-dsl-libs-cu12==4.6.0
nvidia-cutlass-dsl-libs-cu13==4.6.0
```

CUTLASS DSL 4.6 installs `libs-cu12` by default and `[cu13]` adds
`libs-cu13`. The shared `libs-base` and `libs-core` packages are required, but
shipping both CUDA-flavor payloads leaves compiler/runtime selection dependent
on installation and loader behavior.

## Hypothesis 1

The bundled sync selected the CUDA 12 implementation over CUDA 13. This
matches the deterministic root cause established by
`research/mcwitt/7282-uniform-mxfp8` commit `d064dc173`: the default CUDA
payload is unconditional and `[cu13]` adds, rather than replaces, the CUDA 13
wheel.

The reference 4.5.2 fix overrides `nvidia-cutlass-dsl-libs-base`, which then
contained the default CUDA 12 payload, with an impossible platform marker.
CUTLASS DSL 4.6 split the shared code into `libs-base`/`libs-core` and moved
the default CUDA payload to `libs-cu12`, so the equivalent 4.6 fix must
override `libs-cu12` while retaining base/core.

The vendored kernel bodies already match the final uniform-MXFP8 branch
byte-for-byte apart from file-level Ruff exclusions. The local adapter also
contains commit `4876d9670`'s generic-to-gmem pointer normalization, so no
additional addrspace source change is currently indicated.

## Changes to make

- Add `nvidia-cutlass-dsl-libs-cu12==4.6.0 ; sys_platform == 'never'` to the
  root UV overrides.
- Regenerate the lock and verify a frozen `marin-levanter[gpu]` export contains
  `libs-base`, `libs-core`, and `libs-cu13`, but not `libs-cu12`.
- Keep the existing generic-address-space normalization unchanged.

## Results

`uv lock --check --offline` resolves all 608 locked packages successfully. The
corrected frozen Levanter GPU export is:

```text
nvidia-cutlass-dsl==4.6.0 ; sys_platform == 'linux'
nvidia-cutlass-dsl-libs-base==4.6.0 ; sys_platform == 'linux'
nvidia-cutlass-dsl-libs-core==4.6.0 ; sys_platform == 'linux'
nvidia-cutlass-dsl-libs-cu13==4.6.0 ; sys_platform == 'linux'
```

`nvidia-cutlass-dsl-libs-cu12` is absent. The final CPU regression reports 51
passed and 6 skipped in 37.47 seconds.

## Future work

- [ ] Recover the g8 job metadata or approve new 4.5.2-isolation/4.6.0-port
      work before another GB200 submission.

## Attempt 3 result

`ep25d2-mxfp8-numerics-20260725-v2` failed after 12.29 seconds with the same
libNVVM failure for `sm_100a`. Excluding the 4.6.0 CUDA 12 payload did not
clear the compiler gate. The job did not print the loaded CUTLASS extension
or libNVVM path, so the original payload-selection hypothesis is not proven
for the shipped bundle.

## Final reconstruction audit

The first 2.2 PF/s green run was `/mwittmann/mxfp8-002-g8` at commit
`42f7d9fa2`. Its record identifies CUTLASS DSL 4.5.2, the one-GB200 submit
shape, and the mutable `ghcr.io/marin-community/iris-task:latest` default. It
does not identify the pulled image digest, expanded toolkit environment,
libNVVM path, or installed wheel hashes. No explicit `CUDA_TOOLKIT_PATH` or
manual wheel-install step was recorded.

The current Levanter GPU environment cannot directly resolve that dependency
set because QuACK 0.6.1 pins CUTLASS DSL 4.6.0. Creating an isolated 4.5.2
environment or porting the kernel/toolchain to 4.6.0 would be new integration
work.

The numerical entry point now emits `CUTLASS_ENV_SENTINEL` before compilation.
The JSON record identifies the imported CUTLASS module, loaded `_cutlass_ir`
extension and its owning distribution, CUTLASS dist-info directories,
`CUDA_TOOLKIT_PATH`, `LD_LIBRARY_PATH`, and the libNVVM selected by
`cuda.pathfinder`.

## Stop decision

No v3 job is emitted because copying the green environment verbatim is not
possible from the retained record. `EP25_D2_MXFP8_SCOPING.md` lists the
missing artifacts and recovery estimates. The stale EP4 and rack commands
were removed from `EP25_D2_RELAY_COMMANDS.md`.

## Checkpoint escalation: exact 4.5.2 graph recovered

The coordinator compared this worktree with the known-green
`research/mcwitt/7282-uniform-mxfp8` worktree and recovered the missing
dependency resolution. The green bundle used:

```text
nvidia-cutlass-dsl[cu13]>=4.5.2,<4.6
nvidia-cutlass-dsl-libs-base==4.5.2 ; sys_platform == 'never'
```

Both CUTLASS packages use the explicit `https://pypi.nvidia.com/` index. The
green graph also resolves transitive `quack-kernels==0.5.0`; the later direct
QuACK 0.6.1 pin was removed because it forces CUTLASS DSL 4.6.0.

The root, Levanter, and Marin dependency declarations now match the green
constraints. The CUTLASS DSL/base/cu13 and QuACK sections of `uv.lock` were
regenerated to the green 4.5.2 resolution. `uv lock --check --offline`
resolves 606 packages, and the CPU regression reports 52 passed and 6 skipped
in 37.51 seconds. The vendored adapter also imports successfully against the
cached CUTLASS 4.5.2/cu13 payload.

`EP25_D2_RELAY_COMMANDS.md` now emits only the sentinel-enabled v3 numerical
job. A failed v3 remains a checkpoint escalation and does not close the MXFP8
direction under the amended round-6 fleet policy.
