# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Both reference sweeps must lower. ``lower()`` fingerprints every step, and a
``build_config`` that reads ``ctx.resolved()`` unconditionally raises at fingerprint
time — so these DAGs could not be lowered (let alone run) before the fix.
"""

from experiments.references import reference_hyperparameter_sweep, reference_scaling_suite


def test_reference_scaling_suite_lowers():
    for step in reference_scaling_suite.build():
        spec = step.lower()
        assert spec.fingerprint_payload


def test_reference_hyperparameter_sweep_lowers():
    terminal = reference_hyperparameter_sweep.build(num_loops=2)
    spec = terminal.lower()
    assert spec.fingerprint_payload
