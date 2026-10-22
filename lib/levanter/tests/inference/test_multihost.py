# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.timeout(180)
@pytest.mark.parametrize("terminal", ["shutdown", "execution-error"])
def test_malformed_requests_do_not_strand_workers_before_valid_scoring_and_shutdown(tmp_path, terminal):
    script = tmp_path / "multihost.py"
    script.write_text(
        textwrap.dedent(
            """
        import sys
        import haliax as hax
        import jax
        import numpy as np
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.inference.engine import InferenceEngineConfig, Request
        from levanter.inference.multihost import MultiHostInferenceEngine
        from levanter.inference.jit_scheduler import SeqDecodingParams
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

        rank = int(sys.argv[1])
        jax.distributed.initialize(coordinator_address=sys.argv[2], num_processes=2,
                                   process_id=rank, local_device_ids=[rank],
                                   initialization_timeout=60, heartbeat_timeout_seconds=30)
        try:
            mesh = compact_grug_mesh(replica_axis_size=1, expert_axis_size=jax.device_count())
            with jax.set_mesh(mesh), hax.axis_mapping({}):
                cfg = SnowballConfig(vocab_size=32, hidden_dim=8, intermediate_dim=8,
                    shared_expert_intermediate_dim=8, num_experts=4, num_layers=1,
                    num_heads=2, num_kv_heads=1, num_experts_per_token=2, head_dim=8,
                    max_seq_len=16, attention_implementation="reference")
                model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), cfg, key=jax.random.key(17))
                config = InferenceEngineConfig(max_seq_len=16, page_size=4, max_pages=4,
                    max_seqs=1, max_seqs_in_prefill=1, max_prefill_size=2,
                    max_queued_tokens=16, max_rounds=1)
                class FailingEngine(MultiHostInferenceEngine):
                    def _dispatch(self, command):
                        raise RuntimeError("injected post-broadcast execution failure")
                engine_class = FailingEngine if sys.argv[3] == "execution-error" else MultiHostInferenceEngine
                engine = engine_class.from_model_with_config(model, None, config, axis_resources={})
                if rank == 0:
                    for invalid in (lambda: engine.generate([]),
                                    lambda: engine.score_token_logprobs([0], 0),
                                    lambda: engine.score_token_logprobs([0] * 17, 1)):
                        try:
                            invalid()
                        except ValueError:
                            pass
                        else:
                            raise AssertionError("Invalid request accepted")
                    scores = engine.score_token_logprobs([0, 1, 2, 3], 1)
                    assert len(scores.token_logprobs) == 4
                    assert np.isfinite(scores.token_logprobs).all()
                    request = Request([0, 1], 0, SeqDecodingParams(
                        max_num_tokens=np.array(4, np.int32), stop_tokens=None,
                        temperature=np.array(0, np.float32), top_p=np.array(1, np.float32),
                        key=np.array([0, 19311], np.uint32)), 1)
                    result = engine.generate([request])
                    assert result.total_generated > 0
                    engine.shutdown_workers()
                else:
                    engine.worker_loop()
                print("RANK_OK", rank, flush=True)
        finally:
            jax.distributed.shutdown()
    """
        )
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "JAX_NUM_CPU_DEVICES"):
        env.pop(key, None)
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), address, terminal],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(2)
    ]
    try:
        for rank, process in enumerate(processes):
            output, _ = process.communicate(timeout=150)
            if terminal == "execution-error":
                assert process.returncode == 1, output
                assert "injected post-broadcast execution failure" in output, output
            else:
                assert process.returncode == 0, output
                assert f"RANK_OK {rank}" in output, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
