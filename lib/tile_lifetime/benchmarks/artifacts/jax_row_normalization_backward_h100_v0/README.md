# JAX-owned RMS reverse Fold on H100

> **Performance withdrawal:** the matched XLA function closed over benchmark
> arrays, so XLA constant-folded the algebra into constants and copies. The
> `1.955629x` ratio below is invalid. Raw samples are preserved for audit, and
> the correctness/determinism evidence remains valid. See
> `../jax_row_normalization_backward_h100_components_corrected_v1` for the
> corrected runtime-input measurement.

This artifact records one low-priority H100 replay of the generic generated
row-normalization reverse Fold. Ordinary JAX owns automatic differentiation;
Shuttle imports the exported StableHLO reverse program, erases the RMS name,
and generates the two deterministic axis-Fold CUDA bodies through typed FFI.

The source under test was revision
`9e1a556477a38a4b73922a83aa8514539939e58a` plus the generic versioned-library
linker repair committed with this artifact.

## Result

At `rows=2048`, `hidden=4096`, BF16 inputs, and explicit FP32 reverse algebra,
30 counterbalanced samples give:

| Path | Median | Minimum |
| --- | ---: | ---: |
| Generated typed FFI | 0.079698 ms | 0.078855 ms |
| Matched XLA algebra | 0.040753 ms | 0.035220 ms |

The originally reported generated/XLA ratio was `1.955629x`. It is withdrawn
because the XLA side was constant-folded and is not a compute baseline. This
remains a clean executable and correctness proof only.

The generated result is bitwise deterministic across repeated executions. Its
maximum/mean errors against the matched explicit FP32 algebra are:

| Cotangent | Maximum | Mean |
| --- | ---: | ---: |
| Input | `9.536743e-7` | `9.838027e-10` |
| Feature scale | `2.288818e-5` | `3.701312e-6` |

The natural JAX VJP after the BF16 output cast is retained as a diagnostic, not
the accepted reference. Its reduction tree and cast order differ from the
generated deterministic-tree policy: input and feature-scale maximum errors
are `0.0625` and `1.0`, respectively. Source-ordered BF16 equivalence is not
claimed.

## Linker repair

The pip CUDA installation contains versioned `libcudart.so.13` without an
unversioned symlink. The original revision found the correct file but passed it
to NVCC as a positional input, which NVCC rejected. The generic toolchain now
forwards each exact shared-library path through `-Xlinker` and embeds the CUDA
library directory as an rpath.

The successful replay ran with `LIBRARY_PATH`, `LD_LIBRARY_PATH`, and
`CUDA_HOME` unset. There were no symlinks in the pip CUDA library directory
before or after the replay. `generated-library-ldd.txt` shows the generated
library resolving the exact versioned cuDART path. The preserved
`original-positional-link-failure.log` records the pre-fix failure.

See `summary.json` for all raw timing samples, hashes, semantic provenance, and
numerical results. `generated_axis_fold_ffi.cu` is the exact generated source.
