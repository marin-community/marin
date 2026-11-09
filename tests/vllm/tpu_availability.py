# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Skip marker for vLLM tests that need a real TPU."""

from pathlib import Path

import pytest


def _tpu_is_available() -> bool:
    """Whether a Cloud TPU is present, via its numbered VFIO IOMMU groups (/dev/vfio/0, ...).

    Match only a numbered group, not any /dev/vfio entry: the bare /dev/vfio/vfio control device is
    present wherever the vfio module is loaded (e.g. GitHub ubuntu-latest runners) with no TPU bound,
    so ``any(iterdir())`` false-positives there. Numbered groups are the real-TPU signal the iris
    worker and the GrugMoE e2e key off.
    """
    vfio = Path("/dev/vfio")
    return vfio.is_dir() and any(p.name.isdigit() for p in vfio.iterdir())


# Skip a test that needs a real TPU when none is present, so pointing pytest at the file on a
# TPU host runs it while a CPU run reports a clean skip.
skip_if_no_tpu = pytest.mark.skipif(not _tpu_is_available(), reason="no TPU available")
