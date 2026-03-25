# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from levanter.distributed import DistributedConfig, _square_brace_expand


def test_square_brace_expand():
    custom_sequence = "node[001-004,007]suffix"
    expanded_nodes = _square_brace_expand(custom_sequence)
    assert expanded_nodes == ["node001suffix", "node002suffix", "node003suffix", "node004suffix", "node007suffix"]

    custom_sequence_2 = "prefix[001-002]node[005-006]suffix"
    expanded_nodes_2 = _square_brace_expand(custom_sequence_2)
    assert expanded_nodes_2 == [
        "prefix001node005suffix",
        "prefix001node006suffix",
        "prefix002node005suffix",
        "prefix002node006suffix",
    ]

    custom_sequence_3 = "node[1-11]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)]

    custom_sequence_3 = "node[1-11,21]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)] + ["node21suffix"]


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.initialize_jax")
@patch("iris.cluster.client.job_info.get_job_info")
@patch("levanter.distributed.DistributedConfig._is_distributed", return_value=False)
def test_distributed_config_initializes_via_iris_when_iris_job_present(
    mock_is_distributed,
    mock_get_job_info,
    mock_initialize_iris_jax,
    mock_jax_initialize,
):
    """When Iris job info is present and no other distributed env is detected,
    DistributedConfig delegates to iris.runtime.jax_init.initialize_jax."""
    mock_get_job_info.return_value = object()

    DistributedConfig().initialize()

    mock_initialize_iris_jax.assert_called_once_with()
    mock_jax_initialize.assert_not_called()
