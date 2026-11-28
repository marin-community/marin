# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print the expert-axis alignment outcome for the SFT exp3 pair (debug job)."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from readers import configure_object_storage
from registry import CHECKPOINTS, SAMPLE
import measure
import numpy as np
from expert_perm import router_expert_permutation, ROUTER_TENSOR

configure_object_storage()
a_ckpt = CHECKPOINTS["hero-base-393491"]
b_ckpt = CHECKPOINTS["exp3-v5-native-2100"]
side_a = measure.make_side(a_ckpt, SAMPLE, None)
side_b = measure.make_side(b_ckpt, SAMPLE, None)
try:
    alignment = router_expert_permutation(side_a, side_b)
    print("ALIGNMENT OK: identity=%s mean_sim=%.4f first=%s" % (
        alignment["identity"], alignment["mean_similarity"], alignment["perm"][:8].tolist()))
    remap = None if alignment["identity"] else alignment["perm"]
    side_b2 = measure.make_side(b_ckpt, SAMPLE, None, remap)
    req = [r for r in measure.simple_requests(SAMPLE) if r.tensor == ROUTER_TENSOR][0]
    arr_a = side_a.read(req)
    arr_b = side_b2.read(req)
    st = measure.slice_stats(arr_a, arr_b, "as-stored")
    print("ROUTER rel with remap: %.3e" % ((st.sum_d2 / st.sum_w2) ** 0.5))
except Exception as error:
    import traceback
    traceback.print_exc()
    print("ALIGNMENT FAILED:", error)
