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

## Image build attempt

[Workflow run 31445598155](https://github.com/marin-community/marin/actions/runs/31445598155)
built exact source `b42357de95f322e6e1a9ce2eb435dc3a1c0bc08f` with
`image_set=h100-evidence`. All five legacy image jobs were skipped, and only
job `93639015620` ran.

The closed package step downloaded `cuda-nvdisasm-13-2_13.2.86-1_amd64.deb`,
matched its 4,284,630-byte size and SHA-256, extracted it, and reported CUDA
13.2.86 for NVCC, `ptxas`, `cuobjdump`, and `nvdisasm`. The functional smoke
then completed the `sm_90a` cubin compile, poisoned-`PATH` `cuobjdump
--dump-sass`, direct `nvdisasm`, and both kernel-name checks.

The first instruction check failed because the single-quoted grep ERE used two
backslashes around each literal `*`:

```text
grep -Eq '/\\*[[:xdigit:]]+\\*/.*[A-Z][A-Z0-9.]+' /tmp/h100-evidence-cuobjdump.sass
```

The extra backslashes changed the ERE instead of matching an address comment
such as `/*0000*/`. The build stopped at step 4 of 6. The absolute Python CUDA
library probe, CPU-only JAX import probe, image push, registry inspection, and
OCI digest were not reached. No image was published. The downloaded raw job log
was 87,069 bytes with SHA-256
`232a031477b2b48e6358ad821cbdad4a846b9b16fef55a1c1763144ad95d2859`.

## Parser repair

The image build now bind-mounts a stdlib-only validator instead of rendering an
ERE in the Docker shell. The validator accepts one exact expected kernel,
requires bounded UTF-8 output and at least one address-bearing instruction,
and rejects empty output, warning or error diagnostics, malformed records,
unexpected symbols, and duplicate or descending addresses. It validates the
tool-specific `Function` or `.global`/label anchors for `cuobjdump` and
`nvdisasm`. The bind mount does not persist the validator source in the final
image.

The regression uses representative lines from NVIDIA's CUDA binary-utilities
output for both tools. It also executes the previous double-escaped grep
pattern and confirms that it rejects the same valid `cuobjdump` text accepted
by the validator.

## Scoped nvdisasm repair

[Workflow run 31447297229](https://github.com/marin-community/marin/actions/runs/31447297229)
built exact source `298e090a961c90b5857c4f86424ea4546571a706` with
`image_set=h100-evidence`. All five legacy image jobs were skipped, and only
job `93644153924` ran.

The build again matched every closed package size and digest. The `sm_90a`
compile, poisoned-`PATH` `cuobjdump --dump-sass`, and direct `nvdisasm` calls
succeeded. The closed `cuobjdump` validation accepted 24 instructions. The
first failure was the `nvdisasm` validation rejecting this real addressed data
record before the expected function:

```text
/*0000*/ \t.byte\t0xff, 0xff, 0xff, 0xff, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
```

The record belongs to nvdisasm's pre-function data, not the
`.text.h100_evidence_smoke` instruction body. The parser had classified every
address comment in the file as an instruction candidate, so it rejected the
valid data directive before reaching the exact function label.

The build stopped at step 4 of 6. The absolute Python CUDA-library probe,
CPU-only JAX import probe, image push, registry inspection, and OCI digest were
not reached. No image was published. The downloaded raw job log was 89,407
bytes with SHA-256
`22e2094101073b952da4c80a2b21f00cfa9937b73ff7980102fdcf58931643d0`.

The nvdisasm validator now locates one exact expected `.global`, verifies its
exact function label and `.text.<kernel>` section, and validates addressed
records only within that function body. A following section or function closes
the body. Addressed data outside the body is ignored; malformed addressed
records inside it still fail. Missing or duplicate anchors, a wrong text
section, and instructions found only in a trailing function also fail.

## Validation boundary

This checkpoint contains source, local policy-test evidence, and the two failed
image-build records above. It does not include another workflow run, published
image, OCI digest, GPU query, or H100 relaunch. A future immutable image build
must pass the scoped compile/disassembly smoke and the two remaining runtime
probes before it can be considered for another reviewed launch.
