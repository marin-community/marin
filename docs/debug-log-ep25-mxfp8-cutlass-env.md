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

- [ ] Rerun the GB200 numerical ladder after the corrected bundle is shipped.
