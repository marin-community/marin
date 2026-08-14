# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap
from pathlib import Path

from levanter.distributed import _square_brace_expand


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


def test_distributed_state_is_closed_at_process_exit(tmp_path: Path):
    state_path = tmp_path / "distributed-state"
    script = textwrap.dedent(
        """
        import atexit
        import socket
        import sys
        from pathlib import Path

        import jax

        from levanter.distributed import DistributedConfig


        state_path = Path(sys.argv[1])
        atexit.register(lambda: state_path.write_text(str(jax.distributed.is_initialized())))

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        DistributedConfig(
            coordinator_address=f"127.0.0.1:{port}",
            num_processes=1,
            process_id=0,
            local_device_ids=0,
        ).initialize()
        assert jax.distributed.is_initialized()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(state_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=50,
    )

    assert result.returncode == 0, result.stderr
    assert state_path.read_text() == "False"
