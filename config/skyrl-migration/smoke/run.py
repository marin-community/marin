# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise an installed migration wheel against the pinned Math500 evaluation data."""

import argparse
import hashlib
import importlib.resources
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from skyrl_gym.envs.aime import verifier
from skyrl_gym.envs.aime.utils import BOXED_ANSWER_INSTRUCTION, normalize_final_answer
from skyrl_gym.envs.aime.verifier import AIMEVerifier
from skyrl_gym.verification import RolloutEvidence

PAYLOAD = Path(__file__).resolve().parent


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def inputs() -> tuple[list[dict], dict]:
    """Return the verified Math500 rows and the candidate's identity receipt."""
    receipt = json.loads((PAYLOAD / "receipt.json").read_text())
    data = (PAYLOAD / "math500.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == receipt["dataset_sha256"]
    provenance = json.loads(importlib.resources.files("marinskyrl").joinpath("_vendor_provenance.json").read_text())
    assert provenance == receipt["provenance"]
    assert Path(verifier.__file__).is_relative_to(sys.prefix), verifier.__file__
    rows = json.loads(data)
    assert len(rows) == 500
    return rows, receipt


def control(item: tuple[int, dict]) -> dict:
    index, row = item
    answer = normalize_final_answer(row["answer"])
    judge = AIMEVerifier(answer, strict_box_verify=True)
    positive = judge.verify(RolloutEvidence(response=f"\\boxed{{{answer}}}", stop_reason="stop"))
    negative = judge.verify(RolloutEvidence(response="", stop_reason="stop"))
    return {"row": index, "id": row["unique_id"], "positive": positive.passed, "negative": negative.passed}


def controls(output: Path) -> None:
    rows, receipt = inputs()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = []
        for result in pool.map(control, enumerate(rows)):
            results.append(result)
            if len(results) % 25 == 0:
                print(f"Checked {len(results)}/500 reference pairs", flush=True)
    write_json(output, {"receipt": receipt, "results": results, "verifier": verifier.__file__, "python": sys.executable})
    assert all(row["positive"] is True and row["negative"] is False for row in results)
    print("All 500 positive and 500 negative verifier controls passed", flush=True)


def generate(limit: int) -> None:
    # These dependencies are available only in the separately installed GPU profile.
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    import ray  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from skyrl_train.config.utils import get_default_config  # noqa: PLC0415
    from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint  # noqa: PLC0415
    from skyrl_train.utils.utils import initialize_ray, validate_generator_cfg  # noqa: PLC0415

    rows, receipt = inputs()
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "receipt.json", receipt | {"python": sys.executable, "verifier": verifier.__file__})
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}; Torch: {torch.__version__}", flush=True)
    model = snapshot_download(repo_id=receipt["model"], revision=receipt["model_revision"])
    records = [
        {
            "data_source": "math500",
            "prompt": [{"role": "user", "content": row["problem"] + BOXED_ANSWER_INSTRUCTION}],
            "env_class": "aime",
            "reward_model": {"ground_truth": normalize_final_answer(row["answer"])},
            "extra_info": {"split": "test", "index": index},
        }
        for index, row in enumerate(rows[:limit])
    ]
    parquet = output / "math500.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet)
    cfg = get_default_config()
    overrides = {
        "data.train_data": [],
        "data.val_data": [str(parquet)],
        "trainer.policy.model.path": model,
        "trainer.policy.model.revision": receipt["model_revision"],
        "trainer.policy.model.tokenizer_path": model,
        "trainer.policy.model.tokenizer_revision": receipt["model_revision"],
        "trainer.strategy": "megatron",
        "trainer.algorithm.use_kl_loss": False,
        "trainer.placement.colocate_all": True,
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.ref_num_gpus_per_node": 1,
        "trainer.max_prompt_length": 2048,
        "trainer.eval_interval": 1,
        "trainer.eval_num_prompts": None,
        "trainer.eval_batch_size": 32,
        "trajectory_runner.max_concurrent_tasks": 32,
        "trainer.logger": "console",
        "trainer.export_path": str(output),
        "trainer.dump_eval_results": True,
        "environment.task_sessions.aime.strict_box_verify": True,
        "generator.num_inference_engines": 1,
        "generator.inference_engine_tensor_parallel_size": 1,
        "generator.n_samples_per_prompt": 1,
        "generator.gpu_memory_utilization": 0.5,
        "generator.max_num_seqs": 32,
        "generator.model_loading": "stage_local",
        "generator.engine_init_kwargs.max_model_len": 3072,
        "generator.sampling_params.max_generate_length": 1024,
        "generator.eval_sampling_params.temperature": 0.0,
        "generator.eval_sampling_params.max_generate_length": 1024,
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    try:
        metrics = EvalOnlyEntrypoint(cfg).run()
    finally:
        ray.shutdown()
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    assert metrics["eval/all/num_attempted"] == limit, metrics
    assert metrics["eval/all/num_scored"] == limit, metrics
    assert metrics["eval/all/verifier_score_coverage"] == 1.0, metrics
    assert metrics["eval/all/pass_at_1"] > 0.0, metrics
    assert all(math.isfinite(value) for value in metrics.values()), metrics
    print(f"MATH500_SMOKE_PASSED rows={limit}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("controls")
    check.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("generate")
    evaluate.add_argument("--limit", type=int, choices=(32, 500), required=True)
    commands.add_parser("megatron-result")
    args = parser.parse_args()
    if args.command == "controls":
        controls(args.output)
    elif args.command == "megatron-result":
        suite = ET.parse(Path(os.environ["IRIS_OUTPUT_DIR"]) / "megatron.xml").getroot()
        cases = suite.findall(".//testcase")
        assert len(cases) == 3, ET.tostring(suite)
        assert not suite.findall(".//failure"), ET.tostring(suite)
        assert not suite.findall(".//error"), ET.tostring(suite)
        assert not suite.findall(".//skipped"), ET.tostring(suite)
        print("MEGATRON_SMOKE_PASSED: three retained GPU tests; no failures or skips", flush=True)
    else:
        generate(args.limit)


if __name__ == "__main__":
    main()
