# H100 evidence image disassembler closure

## Problem

The second reviewed H100 evidence launch authenticated its source capsule,
selected the frozen runtime, passed the H100 and tool preflight, and compiled
the first generated candidate's shared library, PTX, and cubin. The first SASS
collection then failed before kernel execution:

```text
cuobjdump fatal : Could not find executable file 'nvdisasm'; you can try adding path to environment variables PATH or NVDISASM_PATH
```

The preflight had validated `cuobjdump --version`, but that does not exercise
its SASS-decoding helper.

## Package audit

NVIDIA packages the CUDA disassembler separately from `cuobjdump`. The Debian
12 amd64 package index identifies this CUDA 13.2 package:

- URL: `https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-nvdisasm-13-2_13.2.86-1_amd64.deb`
- Size: `4,284,630` bytes
- SHA-256: `9d3ba750108356723313fa6e42d396a50fff8b00fc6f092a7b098537ac430b79`
- Package: `cuda-nvdisasm-13-2`, version `13.2.86-1`, architecture `amd64`
- Installed executable: `/usr/local/cuda-13.2/bin/nvdisasm`

The downloaded package matched the index size and digest. Its Debian control
metadata describes it as the CUDA disassembler, and its payload contains the
executable at the path above.

## Fix

The closed image manifest now records an absolute NVIDIA Debian 12 URL, byte
size, and SHA-256 for every package. The image rejects records outside that
repository, malformed or nonpositive sizes, size mismatches, and digest
mismatches before extracting a package. It adds the exact `nvdisasm` package
and exposes the CUDA toolkit bin directory through both `PATH` and
`NVDISASM_PATH`.

The build smoke now compiles a tiny `sm_90a` cubin with the pinned CUDA 13.2
NVCC, disassembles it through both `cuobjdump --dump-sass` and `nvdisasm`, and
requires the expected kernel and instruction records in both outputs. These
steps compile and inspect a file only; they do not load a CUDA driver, query a
device, or execute a kernel.

Policy coverage derives the `cuobjdump --dump-sass` requirement from the
runner's checked-in compile plan and binds it to the `nvdisasm` package,
environment, and functional image smoke.

## Validation boundary

This checkpoint contains source and local policy-test evidence only. It does
not include an image build, workflow dispatch, GPU query, or H100 relaunch. A
future immutable image build must pass the new compile/disassembly smoke before
it can be considered for another reviewed launch.
