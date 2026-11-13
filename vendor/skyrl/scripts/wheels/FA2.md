# FlashAttention 2 for Torch 2.14.1

`package_fa2.py` produces an FA2-only package from the exact published Torch
2.14.1/CUDA 13.2.2 native builds on ARM and x86. It excludes the 21 optional
`flash_attn/cute` Python files. Stock Megatron Core 0.18 probes that bundled FA4
code even when using FA2; the old FA4 code cannot import with CUTLASS DSL 4.7.
Core's `megatron/core/transformer/attention.py` catches the missing module's
`ImportError` and sets `HAVE_FA4`
to false. Exclusion takes that fallback without importing the bundled FA4 code.
Core can then select the installed FA2 backend without changes to Core itself.

The input wheels come from
[trainer-native-cu132-torch2141-20261004-db1a32ca](https://github.com/marin-community/MarinSkyRL/releases/tag/trainer-native-cu132-torch2141-20261004-db1a32ca).
Their [frozen build recipe, environment and patch](https://github.com/marin-community/MarinSkyRL/tree/604f2c1463032ffc1f47da114f43cf1d550dd473/scripts/wheels)
pin the upstream source and CUDA compiler cohort. The FlashAttention patch only
changes four C++17 build flags to C++20 for Torch 2.14 headers. It does not change
kernel arithmetic. The release also retains each architecture's `BUILD_INFO`
and `BUILD_REQUIREMENTS.txt`.

The output distribution version is `2.8.3.post2+marin.fa2.cu132torch2141.1`.
Download the wheel for each architecture from that release, then run:

```bash
python scripts/wheels/package_fa2.py \
  flash_attn-2.8.3.post1+marin.cu132torch2141.1-cp312-cp312-linux_aarch64.whl dist
python scripts/wheels/package_fa2.py \
  flash_attn-2.8.3.post1+marin.cu132torch2141.1-cp312-cp312-linux_x86_64.whl dist
```

The script accepts only checksum-pinned inputs. It changes the distribution
version and wheel records, and preserves every remaining file byte for byte,
including the FA2 extension. Each output has a JSON proof with input/output
hashes, excluded files and hashes of every preserved file. Output ZIP metadata
is fixed so the same inputs and Python/zlib versions reproduce archive bytes.
Publish the wheels, proofs and this recipe together. Install the output wheels
as packages; do not remove files from installed environments. This package does
not provide FA4. Native imports and FA2 forward/backward require qualification
on each target before adoption.
