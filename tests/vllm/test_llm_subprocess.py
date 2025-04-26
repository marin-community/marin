"""Test running vllm im a subprocess which is useful for tasks
such as lm_eval_harness which use vllm in a subprocess to generate
completions.
"""

import subprocess

import pytest
import ray

from tests.conftest import model_config


@ray.remote(resources={"TPU-v6e-8-head": 1})
def _test_llm_func(model_config):
    model_path = model_config.ensure_downloaded("/tmp/test-llama-eval")

    import time

    # Start the vllm serve process
    command = ["vllm", "serve", model_path]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    start_time = time.time()
    try:
        # Stream output in real time for up to 3 minutes
        while True:
            line = process.stdout.readline()
            if line:
                print(f"[vllm serve] {line.rstrip()}", flush=True)
            # Check if process has exited
            if process.poll() is not None:
                # Drain any remaining output
                for line in process.stdout:
                    print(f"[vllm serve] {line.rstrip()}", flush=True)
                break
            # Stop after 3 minutes (180 seconds)
            if time.time() - start_time > 180:
                print("Stopping vllm serve after 3 minutes.", flush=True)
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    model_config.destroy()


@pytest.mark.skip(reason="Takes too long and doesn't provide enough value (we don't call vllm in a subprocess anymore)")
def test_local_llm_inference(ray_tpu_cluster):
    ray.get(_test_llm_func.remote(model_config))
