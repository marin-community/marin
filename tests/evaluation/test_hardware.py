# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""select_accelerator's ``--accelerator`` override path: the single-host TPU guard.

Iris expands a multi-host TPU job into one task per VM, so serving on a multi-host TPU shape would
start N independent servers fighting over one endpoint name. select_accelerator must accept only the
single-host slices the cluster actually provisions and reject every other TPU shape.
"""

import pytest

from experiments.evaluation.hardware import SERVABLE_TPU_SLICES, Platform, select_accelerator
from experiments.evaluation.models import EvalModelConfig

# select_accelerator ignores the model entirely once an override is given; only its shape matters here.
_MODEL = EvalModelConfig(name="test-model", location="test/test", hbm_gb=16, apply_chat_template=True)


@pytest.mark.parametrize("tpu_type", SERVABLE_TPU_SLICES)
def test_select_accelerator_accepts_single_host_tpu_override(tpu_type: str) -> None:
    choice = select_accelerator(_MODEL, Platform.TPU, override=tpu_type)
    assert choice.platform == Platform.TPU
    assert choice.tpu_type == tpu_type


@pytest.mark.parametrize("tpu_type", ["v6e-16", "v5p-16", "v5litepod-16"])
def test_select_accelerator_rejects_multi_host_tpu_override(tpu_type: str) -> None:
    with pytest.raises(ValueError, match=tpu_type):
        select_accelerator(_MODEL, Platform.TPU, override=tpu_type)
