# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.long_loop.spec_pipeline import sft_accounting_supported


def test_accounting_settings_without_measured_step_seconds_do_not_support_the_step():
    assert not sft_accounting_supported({"pinned_evidence": {}, "source_loss_metadata": {"source": "s3://x"}})
    assert not sft_accounting_supported({})


def test_accounting_settings_with_a_measured_step_time_support_the_step():
    assert sft_accounting_supported({"measured_step_seconds": 110.0})
