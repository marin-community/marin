# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full GrugMoE canary export plus installed-vLLM generation smoke."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.grug.moe import vllm_tpu_parity as parity

_SERVE_CODE = r"""
import importlib.metadata as md
import importlib.util
import os
from pathlib import Path
import sys

if os.environ.get("PYTHONPATH"):
    raise SystemExit(
        f"PYTHONPATH unexpectedly set in serving subprocess: {os.environ['PYTHONPATH']}"
    )

artifact_dir = Path(sys.argv[1])
print("serve_artifact_dir=" + str(artifact_dir))
for package in ("vllm", "tpu-inference"):
    dist = md.distribution(package)
    direct_url = dist.read_text("direct_url.json")
    print(f"serve_{package}_direct_url=" + (direct_url.strip() if direct_url else ""))
print(
    "serve_grugmoe_spec="
    + repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe"))
)

from vllm import LLM, SamplingParams

prompt_ids = [1, 42, 128, 2048, 17, 3072, 5, 63]
llm = LLM(
    model=str(artifact_dir),
    tokenizer=str(artifact_dir),
    runner="generate",
    skip_tokenizer_init=True,
    trust_remote_code=False,
    dtype="bfloat16",
    enforce_eager=True,
    max_model_len=16,
    max_num_batched_tokens=16,
    max_num_seqs=1,
)
print("llm_initialized=True")
outputs = llm.generate(
    [prompt_ids],
    SamplingParams(max_tokens=3, temperature=0.0),
    use_tqdm=False,
)
result = [
    (list(o.prompt_token_ids), list(o.outputs[0].token_ids), o.outputs[0].text)
    for o in outputs
]
print("installed_full_canary_generated=" + repr(result))
print("installed_path_result=works:full_canary_generate")
"""


def _direct_url(package: str) -> str:
    direct_url = md.distribution(package).read_text("direct_url.json")
    return direct_url.strip() if direct_url else ""


def _print_runtime_header() -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")

    print("remote_cwd=" + os.getcwd())
    try:
        marin_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        marin_sha = f"unavailable:{exc!r}"
    print("marin_sha=" + marin_sha)
    for package in ("vllm", "tpu-inference"):
        print(f"{package}_direct_url=" + _direct_url(package))
    print("vllm_version=" + md.version("vllm"))
    print("tpu-inference_version=" + md.version("tpu-inference"))
    print("jax_version=" + md.version("jax"))
    print("libtpu_version=" + md.version("libtpu"))
    print("grugmoe_spec=" + repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe")))


def run(output_dir: Path, *, max_shard_size: int, generation_tokens: int) -> None:
    _print_runtime_header()

    import tpu_inference.models.jax.grugmoe as tpu_grugmoe

    if output_dir.exists():
        shutil.rmtree(output_dir)
    parity.check_realistic_training_state_roundtrip(
        tpu_grugmoe,
        config_name="canary",
        output_dir=output_dir,
        max_shard_size=max_shard_size,
        generation_tokens=generation_tokens,
    )

    artifact_dir = output_dir / "grugmoe-inference"
    print("full_canary_artifact_dir=" + str(artifact_dir))
    print("full_canary_artifact_bytes=" + str(parity._directory_size_bytes(artifact_dir)))
    print("full_canary_shard_count=" + str(parity._shard_count(artifact_dir)))

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["VLLM_TARGET_DEVICE"] = "tpu"
    subprocess.run(
        [sys.executable, "-c", _SERVE_CODE, str(artifact_dir)],
        check=True,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/grugmoe-installed-vllm-full-canary"),
    )
    parser.add_argument("--max-shard-size", type=int, default=268435456)
    parser.add_argument("--generation-tokens", type=int, default=3)
    args = parser.parse_args()
    run(
        args.output_dir,
        max_shard_size=args.max_shard_size,
        generation_tokens=args.generation_tokens,
    )


if __name__ == "__main__":
    main()
