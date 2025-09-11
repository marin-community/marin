import json
import os

import fsspec
import ray

from marin.processing.classification.config.inference_config import (
    InferenceConfig,
    RuntimeConfig,
)
from marin.processing.classification.inference import (
    run_inference,
)


def _make_text(n_chars: int = 300) -> str:
    base = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed \n"
        "do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    )
    s = (base * ((n_chars // len(base)) + 1))[:n_chars]
    return s


def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    fs, _ = fsspec.core.url_to_fs(path)
    fs.makedirs(os.path.dirname(path), exist_ok=True)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# @pytest.fixture(scope="session", autouse=True)
# def _enable_tpu_cluster_env():
#     os.environ["START_RAY_TPU_CLUSTER"] = "true"


# @pytest.mark.usefixtures("ray_tpu_cluster")
def test_inference_and_reaper_integration(current_date_time, gcsfuse_mount_model_path):
    timestamp = current_date_time
    input_base = f"gs://marin-us-central2/scratch/test-cleanup/{timestamp}/in"
    output_base = f"gs://marin-us-central2/scratch/test-cleanup/{timestamp}/out"

    # Clean up any existing output directory to start fresh
    fs, _ = fsspec.core.url_to_fs(output_base)
    if fs.exists(output_base):
        fs.rm(output_base, recursive=True)

    # Create 25 input files, each 5 rows with id,text
    num_files = 25
    num_rows = 5
    for i in range(num_files):
        rows = []
        for j in range(num_rows):
            rows.append(
                {
                    "id": f"file{i}-row{j}",
                    "text": _make_text(300),
                }
            )
        _write_jsonl_gz(f"{input_base}/part-{i:02d}.jsonl.gz", rows)

    # Build inference config using vLLM small model; tensor_parallel_size=1
    engine_kwargs = {"tensor_parallel_size": 1, "enforce_eager": True, "max_model_len": 1024}
    generation_kwargs = {"max_tokens": 16}

    cfg = InferenceConfig(
        input_path=input_base,
        model_name=gcsfuse_mount_model_path,
        attribute_name="quality",
        model_type="vllm",
        output_path=output_base,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        filetype="jsonl.gz",
        batch_size=4,
        resume=False,
        classifier_kwargs={
            "template": "You are a rater. Rate the quality from 0 to 5. Text: {text}. Answer with a single number.",
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "apply_chat_template": False,
            "prompt_column": "text",
            "generated_text_column_name": "generated_text",
        },
    )

    # Run distributed inference over the directory
    ray.get(run_inference.remote(cfg))

    # Validate outputs exist
    # fs, _ = fsspec.core.url_to_fs(output_base)
    # found = fs.glob(f"{output_base}/*.jsonl.gz")
    # assert len(found) == num_files

    # # Now directly run one file with a queue to capture the worker PID and invoke the Reaper
    # q = Queue()
    # input_one = f"{input_base}/part-00.jsonl.gz"
    # output_one = f"{output_base}/single-part-00.jsonl.gz"

    # # Submit a single task with the same settings
    # ref = process_file_ray.options(
    #     memory=16 * 1024 * 1024 * 1024,
    #     resources={"TPU": 1},
    # ).remote(
    #     input_one,
    #     output_one,
    #     gcsfuse_mount_model_path,
    #     "quality",
    #     "vllm",
    #     "jsonl.gz",
    #     {
    #         "template": "You are a rater. Rate the quality from 0 to 5. Text: {text}. Answer with a single number.",
    #         "engine_kwargs": engine_kwargs,
    #         "generation_kwargs": generation_kwargs,
    #         "apply_chat_template": False,
    #         "prompt_column": "text",
    #         "generated_text_column_name": "generated_text",
    #     },
    #     4,
    #     False,
    #     q,
    # )
    # ray.get(ref)

    # # Get worker info and run reaper on the same node
    # info = q.get()
    # reaper = Reaper.options(
    #     scheduling_strategy=NodeAffinitySchedulingStrategy(info["node_id"], soft=False)
    # ).remote()
    # result = ray.get(reaper.kill_if_holding_accel.remote(info["pid"]))

    # # If any processes are holding accelerator FDs, they should be killed.
    # if result["holding"]:
    #     assert set(result["killed"]) >= set(result["holding"]) and len(result["failed"]) == 0
    # else:
    #     # Nothing was holding accelerator FDs; still a valid outcome
    #     assert len(result["killed"]) == 0
