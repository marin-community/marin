# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-011: dissect serialized jit_train_step executables from the JAX persistent cache.

The 16-node cute-producer runs fail executable load with cuModuleLoadData
CUDA_ERROR_INVALID_VALUE, but they still write their compiled executable to the
persistent compilation cache before the load attempt. This tool downloads cache
entries (failing cute twin + passing xla twin), decompresses them, carves out
every embedded ELF (cubins) and PTX blob, and reports structural facts:
sizes, ELF machine/flags (SM arch), section/symbol counts, and kernel-name
samples. Run on an iris CPU pod (has S3 creds); pass cache-entry S3 paths as
argv.
"""

import struct
import sys
import zlib

import fsspec

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
GZIP_MAGIC = b"\x1f\x8b"
ELF_MAGIC = b"\x7fELF"
EM_CUDA = 190


def decompress(raw: bytes) -> bytes:
    if raw[:4] == ZSTD_MAGIC:
        import zstandard  # noqa: PLC0415

        return zstandard.ZstdDecompressor().decompress(raw, max_output_size=2**33)
    if raw[:2] == GZIP_MAGIC:
        return zlib.decompress(raw, wbits=31)
    try:
        return zlib.decompress(raw)
    except zlib.error:
        return raw


def elf_span(buf: bytes, off: int) -> dict | None:
    """Parse an ELF64 header at `off`; return dict of facts incl. total size."""
    if buf[off : off + 4] != ELF_MAGIC or off + 64 > len(buf):
        return None
    is64 = buf[off + 4] == 2
    if not is64:
        return None
    (e_machine,) = struct.unpack_from("<H", buf, off + 18)
    (e_flags,) = struct.unpack_from("<I", buf, off + 48)
    (e_shoff,) = struct.unpack_from("<Q", buf, off + 40)
    e_shentsize, e_shnum = struct.unpack_from("<HH", buf, off + 58)
    size = e_shoff + e_shentsize * e_shnum
    if e_shoff == 0 or size <= 64 or off + size > len(buf) + 16:
        # header claims sections beyond the buffer -> corrupt or truncated
        return {
            "off": off,
            "machine": e_machine,
            "flags": hex(e_flags),
            "shnum": e_shnum,
            "shoff": e_shoff,
            "size": None,
            "truncated": True,
        }
    return {
        "off": off,
        "machine": e_machine,
        "flags": hex(e_flags),
        "shnum": e_shnum,
        "shoff": e_shoff,
        "size": size,
        "truncated": False,
    }


def kernel_names(buf: bytes, off: int, size: int, limit: int = 12) -> list[str]:
    """Crude symbol sample: printable strings that look like kernel symbols."""
    import re  # noqa: PLC0415

    window = buf[off : off + min(size or 1 << 22, 1 << 22)]
    names = re.findall(rb"[A-Za-z_][A-Za-z0-9_$]{6,120}", window)
    keep = []
    for n in names:
        s = n.decode()
        if any(t in s for t in ("fusion", "triton", "kernel", "wrapped", "cute", "concat", "slice", "while", "cond")):
            if s not in keep:
                keep.append(s)
        if len(keep) >= limit:
            break
    return keep


def analyze(path: str) -> None:
    with fsspec.open(path, "rb") as f:
        raw = f.read()
    print(f"ENTRY {path}")
    print(f"  raw_bytes={len(raw)} magic={raw[:4].hex()}")
    buf = decompress(raw)
    print(f"  decompressed_bytes={len(buf)}")
    offs = []
    i = buf.find(ELF_MAGIC)
    while i != -1:
        offs.append(i)
        i = buf.find(ELF_MAGIC, i + 4)
    print(f"  elf_count={len(offs)}")
    for off in offs:
        info = elf_span(buf, off)
        if info is None:
            print(f"  ELF off={off} unparseable(32-bit or short)")
            continue
        tag = "CUDA" if info["machine"] == EM_CUDA else f"m{info['machine']}"
        print(
            f"  ELF off={info['off']} machine={tag} flags={info['flags']} "
            f"shnum={info['shnum']} size={info['size']} truncated={info['truncated']}"
        )
        if info["machine"] == EM_CUDA and info["size"]:
            print(f"    names={kernel_names(buf, off, info['size'])}")
    ptx = buf.find(b".version ")
    print(f"  ptx_marker_off={ptx}")
    for marker in (b"nvJitLink", b"fatbin", b"__cudaFatFormat", b"CuteDSLRT"):
        print(f"  marker {marker.decode()}: {buf.find(marker)}")


def main() -> None:
    for path in sys.argv[1:]:
        analyze(path)
        print()


if __name__ == "__main__":
    main()
