# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review-job roots must land on a cluster that places their 4xGB200 serving.

Only ROOT jobs federate in Iris: a job's whole tree stays on the root's peer, so a
``target_cluster`` on the child serving request is ignored. The submission-time check
refuses roots on (or unpinned and able to federate onto) cw-us-east-02a, whose Kueue
topology gates 4xGB200 silently.
"""

from pathlib import Path

import pytest
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model

from experiments.post_training.task_curation.full_qc import SAMPLE_REVIEWER_VERSION, _local_endpoint, local_reviewer
from experiments.post_training.task_curation.judges import require_review_serving_root
from experiments.post_training.task_curation.sample_qc_artifacts import SAMPLE_REVIEWER_SHAS

V3_MODEL_CONFIG = Path("experiments/evaluation/serve/models/zai-org/GLM-5.3-full-qc-v3.yaml")


def test_unpinned_review_root_is_refused():
    """An unpinned root could federate onto the gated cluster; nothing would ever review."""
    with pytest.raises(ValueError, match="must root on a cluster that places it"):
        require_review_serving_root(None)


def test_review_root_on_the_gated_cluster_is_refused():
    with pytest.raises(ValueError, match="cw-us-east-02a"):
        require_review_serving_root("cw-us-east-02a")


def test_review_root_on_a_placable_cluster_is_accepted():
    assert require_review_serving_root("cw-us-east-08a") == "cw-us-east-08a"


def test_child_only_serving_pin_does_not_satisfy_the_root_check():
    """A child target_cluster is ignored (only roots federate): it cannot cure a gated root."""
    launcher = _local_endpoint(V3_MODEL_CONFIG, 524288)
    # The launcher no longer carries a child placement pin at all - the root decides.
    assert launcher.accelerator.target_cluster is None
    with pytest.raises(ValueError, match="only roots federate"):
        require_review_serving_root("cw-us-east-02a")


def test_pinned_reviewer_keeps_its_identity_and_leaves_placement_to_the_root():
    """The pinned SAMPLE reviewer's identity stays byte-stable; the root governs placement."""
    spec = local_reviewer(SAMPLE_REVIEWER_VERSION)
    assert spec.cache_revision == SAMPLE_REVIEWER_SHAS[SAMPLE_REVIEWER_VERSION]
    assert spec.endpoint.local.accelerator.target_cluster is None
    assert _local_endpoint(V3_MODEL_CONFIG, 524288).accelerator.target_cluster is None


def test_non_gb200_serving_requests_still_lower_cleanly():
    accelerator = AcceleratorChoice(Platform.GPU, gpu_type="H100", gpu_count=1)
    assert accelerator.target_cluster is None


def test_judge_serving_requests_non_preemptible_capacity():
    """Long reviewer serving must not sit on preemptible capacity."""
    endpoint = _local_endpoint(V3_MODEL_CONFIG, 524288)
    assert endpoint.accelerator.preemptible is False

    model = load_model_config(V3_MODEL_CONFIG)
    config = inference_config_for_model(
        model,
        endpoint.accelerator,
        env_vars=endpoint.env_vars,
        priority=endpoint.priority,
        api_model="glm-5.3",
    )
    assert config.iris.worker_resources.preemptible is False
