# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference interpreter for `SegmentPlan` transfers.

Executes every device's plan with host copies — what the `put_segments`
kernel should produce. Shared by the metadata tests, the GPU kernel tests,
and the multi-node smoke.
"""

import numpy as np


def execute_plans(plans, sources, out_rows: int) -> list[np.ndarray]:
    """Return each destination's receive buffer after all plans execute."""
    hidden = sources[0].shape[1]
    outputs = [np.zeros((out_rows, hidden), sources[0].dtype) for _ in range(len(plans))]
    for device, plan in enumerate(plans):
        dest_ids, entry_start = np.asarray(plan.dest_ids), np.asarray(plan.entry_start)
        src_lo, dst_lo, rows = (np.asarray(a) for a in (plan.src_lo, plan.dst_lo, plan.rows))
        for k, dest in enumerate(dest_ids):
            for e in range(entry_start[k], entry_start[k + 1]):
                outputs[dest][dst_lo[e] : dst_lo[e] + rows[e]] = sources[device][src_lo[e] : src_lo[e] + rows[e]]
    return outputs
