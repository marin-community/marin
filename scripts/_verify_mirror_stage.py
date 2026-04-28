# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-region smoke test for `_stage_mirror_to_local`.

Submitted as an iris job pinned to a region that does NOT hold the target
checkpoint, to prove that MirrorFileSystem copies it in and TensorStore can
open the resulting concrete URL.
"""

import asyncio
import logging
import sys
import time

import fsspec
import tensorstore as ts

from levanter.checkpoint import _stage_mirror_to_local
from levanter.tensorstore_serialization import _create_ocdbt_spec
from rigging.filesystem import marin_prefix, mirror_budget

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

CKPT_REL = "checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915"
MIRROR_URL = f"mirror://{CKPT_REL}"


async def _count_ocdbt_keys(resolved_url: str) -> int:
    kvstore = await ts.KvStore.open(_create_ocdbt_spec(resolved_url, array_path=None)["kvstore"])
    return len(await kvstore.list())


def main() -> int:
    print(f"[verify] marin_prefix = {marin_prefix()}")
    print(f"[verify] mirror URL   = {MIRROR_URL}")

    t0 = time.time()
    # Matches the budget `experiments/exp_delphi_math_10b_midtrain.py` sets
    # on the 1e20 `mirrored(...)` value (30 GB). The real training run runs
    # the same helper inside the executor's own `mirror_budget()` context.
    with mirror_budget(30.0):
        resolved = _stage_mirror_to_local(MIRROR_URL)
    stage_s = time.time() - t0
    print(f"[verify] staged in {stage_s:.1f}s")
    print(f"[verify] resolved    = {resolved}")

    if not resolved.startswith(marin_prefix().rstrip("/")):
        print("[verify] FAIL: resolved URL does not live under local marin prefix")
        return 1

    fs, fspath = fsspec.core.url_to_fs(resolved)
    expected = ("metadata.json", "manifest.ocdbt")
    for rel in expected:
        full = f"{fspath}/{rel}"
        if not fs.exists(full):
            print(f"[verify] FAIL: missing {rel} at {full}")
            return 1
        print(f"[verify]   {rel}: OK")

    try:
        num_keys = asyncio.run(_count_ocdbt_keys(resolved))
    except Exception as exc:
        print(f"[verify] FAIL: TensorStore OCDBT open failed: {exc!r}")
        return 1
    print(f"[verify]   OCDBT keys   = {num_keys}")
    if num_keys == 0:
        print("[verify] FAIL: OCDBT returned zero keys")
        return 1

    print("[verify] SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
