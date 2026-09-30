# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dump the buffers live at the compiler's memory peak for main and the candidate in the layer scan.

Compiles the rematted scan of `scan_compare.py` for the named variants with XLA's HLO dump on, then
prints each module's peak-buffer report from the buffer-assignment dump.

Usage (GB200x4): python autoresearch/loop-260930-mfu30/b/scan_memory.py control candidate
"""

import glob
import os
import re
import shutil
import sys

DUMP_ROOT = "/tmp/m30b_scan_memory"


def main():
    names = sys.argv[1:]
    for name in names:
        dump = f"{DUMP_ROOT}/{name}"
        shutil.rmtree(dump, ignore_errors=True)
    base_flags = os.environ.get("XLA_FLAGS", "")
    for name in names:
        # Each variant compiles in a fresh process so the dump flag applies to its module alone.
        pid = os.fork()
        if pid == 0:
            os.environ["XLA_FLAGS"] = f"{base_flags} --xla_dump_to={DUMP_ROOT}/{name} --xla_dump_hlo_as_text"
            import scan_compare as sc

            mesh = sc._mesh()
            inp = sc._inputs(mesh)
            local_fn, policy, backward = sc.VARIANTS[name]
            exe, _args = sc._build(mesh, inp, local_fn, policy, backward)
            stats = exe.memory_analysis()
            print(f"== {name} temp_bytes={stats.temp_size_in_bytes}", flush=True)
            os._exit(0)
        os.waitpid(pid, 0)
        for path in sorted(glob.glob(f"{DUMP_ROOT}/{name}/*loss*buffer-assignment.txt")):
            text = open(path).read()
            match = re.search(r"Peak buffers:\n(.*?)(\n\n|\Z)", text, re.S)
            print(f"== {name} {os.path.basename(path)}", flush=True)
            if not match:
                print("   no peak report", flush=True)
                continue
            for line in match.group(1).splitlines()[:120]:
                if "value:" in line or "Buffer" in line or "size" in line:
                    print("   " + line.strip()[:260], flush=True)


if __name__ == "__main__":
    main()
