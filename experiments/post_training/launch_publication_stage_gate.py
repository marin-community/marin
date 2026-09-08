# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preview or submit bounded native weight-sync and pause/continue gates."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

NODEID = "skyrl-train/tests/gpu/test_megatron_worker.py::test_megatron_policy_weight_sync[publication_stages_two_gpu]"
CHAT_NODEID = (
    "skyrl-train/tests/gpu/gpu_ci/test_pause_and_continue_generation.py"
    "::test_continue_generation_vllm_engine_chat_completion"
)
SOURCES = (
    "uv.lock",
    "pyproject.toml",
    "cloud/iris/bootstrap_runtime.sh",
    "skyrl-train/tests/gpu/test_megatron_worker.py",
    "skyrl-train/skyrl_train/weight_sync/publication_timing.py",
    "skyrl-train/skyrl_train/workers/megatron/megatron_worker.py",
    "skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py",
    "skyrl-train/skyrl_train/inference_engines/inference_engine_client.py",
    "skyrl-train/skyrl_train/fully_async_trainer.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--marin-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--native-chat", action="store_true", help="Run the exact native HTTP pause/continue gate")
    parser.add_argument("--native-chat-prerequisite", action="store_true", help="Qualify native chat inputs on CPU only")
    args = parser.parse_args()
    assert not args.native_chat_prerequisite or args.native_chat
    assert len(args.marin_commit) == 40 and all(c in "0123456789abcdef" for c in args.marin_commit)
    assert args.job_name.startswith("async-rl-v2-publication-stage-")
    marin_root = Path(__file__).resolve().parents[2]
    assert (
        subprocess.check_output(["git", "-C", str(marin_root), "rev-parse", "HEAD"], text=True).strip()
        == args.marin_commit
    )
    assert not subprocess.check_output(["git", "-C", str(marin_root), "status", "--porcelain"], text=True)
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sources = SOURCES + (
        (
            "skyrl-train/tests/gpu/gpu_ci/test_pause_and_continue_generation.py",
            "skyrl-train/tests/gpu/gpu_ci/test_inference_engine_client_http_endpoint.py",
            "skyrl-train/tests/gpu/utils.py",
            "skyrl-train/skyrl_train/inference_engines/inference_engine_client_http_endpoint.py",
            "skyrl-train/examples/gsm8k/gsm8k_dataset.py",
        )
        if args.native_chat
        else ()
    )
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}
    nodeid = CHAT_NODEID if args.native_chat else NODEID
    timeout = 700 if args.native_chat and not args.native_chat_prerequisite else 1800
    prepare_chat = (
        """
/tmp/oa-publication-env/bin/python "$publication_root/skyrl-train/examples/gsm8k/gsm8k_dataset.py"
/tmp/oa-publication-env/bin/python - <<'PROMPTS'
import hashlib,json
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
from tests.gpu.utils import get_test_prompts,TEST_DATA_PATH
from tests.gpu.gpu_ci.test_pause_and_continue_generation import MODEL
from tests.gpu.gpu_ci.test_inference_engine_client_http_endpoint import get_test_actor_config
prompts=get_test_prompts(MODEL,num_samples=1)
assert len(prompts)==1 and prompts[0] and all(m['content'] for m in prompts[0])
# Execute the actual native test's prompt-token expression with the real tokenizer.
import ast
from types import SimpleNamespace
from transformers import AutoTokenizer
native=Path('skyrl-train/tests/gpu/gpu_ci/test_pause_and_continue_generation.py')
function=next(n for n in ast.parse(native.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='test_continue_generation_vllm_engine_chat_completion')
assignment=next(n for n in ast.walk(function) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt_tokens' for t in n.targets))
tokenizer=AutoTokenizer.from_pretrained(MODEL)
actual=eval(compile(ast.Expression(assignment.value),str(native),'eval'),{'client':SimpleNamespace(tokenizer=tokenizer),'messages':prompts[0]})
mapping=tokenizer.apply_chat_template(prompts[0],add_generation_prompt=True,tokenize=True,return_dict=True)
assert isinstance(actual,list) and actual==mapping['input_ids'] and len(actual)>2
print('NATIVE_CHAT_TOKENIZER_EXPRESSION_PASS',json.dumps({'mapping_fields':len(mapping),'prompt_token_ids':len(actual),'native_test_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}),flush=True)

cfg=get_test_actor_config(num_inference_engines=2,model=MODEL)
assert cfg.generator.num_inference_engines==2 and cfg.generator.inference_engine_tensor_parallel_size==1
path=Path(TEST_DATA_PATH)
cached=try_to_load_from_cache(MODEL,'tokenizer_config.json')
snapshot=Path(cached).parent.name if isinstance(cached,str) else None
print('NATIVE_CHAT_PREREQUISITE_PASS',json.dumps({'path':str(path),'parquet_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prompt_count':len(prompts),'model':MODEL,'tokenizer_snapshot':snapshot}),flush=True)
PROMPTS
"""
        if args.native_chat
        else ""
    )
    finish = (
        """
/tmp/oa-publication-env/bin/python - <<'CPU'
import torch
assert torch.cuda.device_count()==0
print('NATIVE_CHAT_CPU_ONLY_QUALIFICATION_PASS',flush=True)
CPU
"""
        if args.native_chat_prerequisite
        else f"""
/tmp/oa-publication-env/bin/python - <<'GPU'
import torch
assert torch.cuda.device_count()==2
assert all('H100' in torch.cuda.get_device_name(i) for i in range(2))
print('PUBLICATION_TWO_H100_PREFLIGHT_PASS',flush=True)
GPU
exec /tmp/oa-publication-env/bin/python -m pytest -s -q '{nodeid}'
"""
    )
    bootstrap = (
        "UV_PROJECT_ENVIRONMENT=/tmp/oa-publication-env uv sync --quiet --frozen "
        '--project "$publication_root" --python python3.12 --no-python-downloads '
        "--link-mode symlink --extra cpu --extra telemetry --group dev --group harbor-test"
        if args.native_chat_prerequisite
        else 'bash "$publication_root/cloud/iris/bootstrap_runtime.sh" "$publication_root" '
        "/tmp/oa-publication-env /tmp/oa-publication-runtime megatron development\n"
        "source /tmp/oa-publication-runtime"
    )
    assert "publication_stage_walls" in Path(SOURCES[-1]).read_text(), "driver trace hook must be present"
    body = f"""set -euo pipefail
publication_root="$PWD"
export PYTHONPATH="$publication_root/skyrl-train:$publication_root/skyrl-gym:$publication_root"
export HF_HOME=/tmp/oa-cache/huggingface
export UV_CACHE_DIR=/tmp/oa-cache/uv
export WANDB_MODE=disabled
export LOGURU_LEVEL=INFO
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
python3 - <<'VERIFY'
import hashlib,json
from pathlib import Path
expected=json.loads({json.dumps(json.dumps(hashes))})
for name,digest in expected.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('PUBLICATION_SOURCE_VERIFIED',json.dumps({{'msr':{commit!r},'marin':{args.marin_commit!r},'sha256':expected}}),flush=True)
VERIFY
{bootstrap}
{prepare_chat}
{finish}
"""
    command = [
        os.environ["IRIS"],
        "--cluster",
        "marin",
        "job",
        "run",
        "--job-name",
        args.job_name,
        "--target-cluster",
        "cw-us-east-02a",
        "--priority",
        "batch",
        *([] if args.native_chat_prerequisite else ["--gpu", "H100x2"]),
        "--replicas",
        "1",
        "--cpu",
        "8",
        "--memory",
        "64GB",
        "--disk",
        "100GB",
        "--enable-extra-resources",
        "--max-retries",
        "0",
        "--timeout",
        str(timeout),
        "--no-sync",
        "--no-wait",
        "--",
        "bash",
        "-c",
        body,
    ]
    preview = {
        "msr": commit,
        "marin": args.marin_commit,
        "sha256": hashes,
        "nodeid": nodeid,
        "model_revision": None if args.native_chat else "c1899de289a04d12100db370d81485cdf75e47ca",
        "model": "Qwen/Qwen2.5-0.5B-Instruct" if args.native_chat else "Qwen/Qwen3-0.6B",
        "profile": (
            "cpu/telemetry + frozen dev/harbor-test groups"
            if args.native_chat_prerequisite
            else "megatron/vllm/telemetry + frozen dev group"
        ),
        "max_task_gpu_hours": 0 if args.native_chat_prerequisite else 2 * timeout / 3600,
        "prerequisite_only": args.native_chat_prerequisite,
        "storage": "Qwen model and canonical GSM8K fixture in pod cache; no bucket writes; durable Iris logs",
        "command": command,
    }
    print(json.dumps(preview, indent=2), flush=True)
    if args.execute:
        assert os.environ["IRIS_USER"] == "atqamar"
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
