# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test fast TPU bootstrap via tpu-inference fork.

Submits two Iris jobs on v6e-4:
1. Baseline: default vllm serve loading (runai_streamer)
2. Fast bootstrap: load_format=dummy + tpu_bootstrap abstract_dummy

Compare startup times to validate the fork's fast bootstrap path.
"""

import json
import time
import traceback

from fray.v1.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VLLM_NATIVE_PIP_PACKAGES, VllmEnvironment
from marin.utils import remove_tpu_lockfile_on_exit

MODEL = "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"
TPU_TYPE = "v6e-4"
PROMPT = "What is 2 + 2? Answer in one word."


def run_test(*, label: str, engine_kwargs: dict) -> None:
    """Start vLLM, send one query, report timing."""
    with remove_tpu_lockfile_on_exit():
        model = ModelConfig(
            name="bootstrap-test",
            path=MODEL,
            engine_kwargs={"max_model_len": 4096, **engine_kwargs},
        )
        env = VllmEnvironment(
            model=model,
            host="127.0.0.1",
            port=8000,
            timeout_seconds=1800,
            mode="native",
        )
        t0 = time.time()
        try:
            import requests

            with env:
                t_ready = time.time() - t0
                print(f"[{label}] Server ready in {t_ready:.1f}s")

                response = requests.post(
                    f"{env.server_url}/chat/completions",
                    json={
                        "model": env.model_id,
                        "messages": [{"role": "user", "content": PROMPT}],
                        "temperature": 0.0,
                        "max_tokens": 32,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                t_total = time.time() - t0
                output = response.json()["choices"][0]["message"]["content"]
                print(f"[{label}] First response in {t_total:.1f}s: {output}")
        except Exception:
            elapsed = time.time() - t0
            print(f"[{label}] FAILED after {elapsed:.1f}s")
            traceback.print_exc()
            raise


def launch_job(*, label: str, engine_kwargs: dict, job_suffix: str) -> str:
    cluster = current_cluster()

    def _run():
        run_test(label=label, engine_kwargs=engine_kwargs)

    job_request = JobRequest(
        name=f"fast-bootstrap-{job_suffix}",
        entrypoint=Entrypoint.from_callable(_run),
        resources=ResourceConfig.with_tpu(TPU_TYPE),
        environment=EnvironmentConfig.create(
            extras=["eval", "tpu", "vllm"],
            pip_packages=VLLM_NATIVE_PIP_PACKAGES,
            env_vars={"MARIN_VLLM_MODE": "native"},
        ),
    )
    job_id = cluster.launch(job_request)
    print(f"Launched {label}: {job_id}")
    return job_id


def main() -> int:
    # Job 1: Baseline (default loading)
    baseline_id = launch_job(
        label="baseline",
        engine_kwargs={},
        job_suffix="baseline",
    )

    # Job 2: Fast bootstrap
    fast_id = launch_job(
        label="fast-bootstrap",
        engine_kwargs={
            "load_format": "dummy",
            "num_gpu_blocks_override": 128,
            "model_loader_extra_config": json.dumps(
                {
                    "tpu_bootstrap": {
                        "model_bootstrap": "abstract_dummy",
                        "prefer_jax_for_bootstrap": True,
                    }
                }
            ),
        },
        job_suffix="fast",
    )

    cluster = current_cluster()
    print(f"\nBaseline job: {baseline_id}")
    print(f"Fast bootstrap job: {fast_id}")
    print("\nWaiting for both jobs...")

    for job_id, label in [(fast_id, "fast-bootstrap"), (baseline_id, "baseline")]:
        try:
            cluster.wait(job_id, raise_on_failure=True)
            print(f"{label}: PASSED")
        except Exception as e:
            print(f"{label}: FAILED - {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
