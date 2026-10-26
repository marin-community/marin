# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one arm of the milestone-16 spending gate: bind, serve on cw-rno2a H100x8, score, release.

``preflight`` (no GPU) binds the arm's model and writes its provenance:

- candidate: ``<run root>/sft-result.json`` must report ``completed_updates == 16``, a native checkpoint
  and ``hf_path == <run root>/hf/step-15``; its ``trained-precision-audit.json`` hash is recorded;
- both arms: metadata file hashes (the baseline's must equal the LR0 source hashes; the candidate's
  template, tokenizer and generation config must equal the baseline's, except that ``tokenizer_config.json``
  may carry an embedded copy of the baseline ``chat_template.jinja`` and ``generation_config.json`` may
  differ in JSON formatting only, as the SFT export audit accepts; see ``accept_candidate_metadata``),
  shard sizes and ETags, and every
  safetensors header: 502 tensors, the 26 ``mlp.router.bias`` F32 and the other 476 BF16.

``run`` repeats the preflight, checks for an idle whole H100 node on cw-rno2a, submits the serve in
the production band with the fixed SERVE_ARGS, waits for the endpoint, writes the capability to an
owner-only file outside Dropbox, scores with ``m16_gate.score_arm``, records engine evidence from
the startup logs, deletes the capability and cancels its own serve. Capability URLs and exception
text are never printed.
"""

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from marin.inference.iris_cli import _wait_for_endpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.timing import Duration
from transformers import AutoTokenizer

from experiments.interp.snowball_identity import identity_eval as shared
from experiments.interp.snowball_identity import m16_gate
from experiments.post_training.identity_context_distill.precision_source_audit import check_export_auxiliary_metadata

BASELINE_MODEL = "s3://marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a"
BASELINE_REVISION = "adc0078a19a4383bc998fcdb43867b3360baf5f3"
# LR0 SOURCE_FILE_HASHES (experiments/post_training/identity_context_distill/lr0_probe.py at 625cfe38).
BASELINE_FILE_HASHES = {
    "config.json": "c7e1e7ac08e7b8b712d2728596765d2a5f954a7372ba1090cf87709e5fa9fbd5",
    "generation_config.json": "fef404c3c02c0c5120a18a55820caac75222b507f27e1aef0e4cd482ba887d73",
    "chat_template.jinja": "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de",
    "model.safetensors.index.json": "7737eed0ad5faa748e709ae9fb7dfb1baded1e5a18b2d811cf13f69776b4869d",
    "tokenizer.json": "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45",
    "tokenizer_config.json": "7c11be094d34bd4bb5cf704584fb71b1164220bcfcc5963321bc6ec03d7ec853",
}
TENSORS, ROUTER_BIASES = 502, 26
VLLM = {
    "source": "marin-fork",
    "release_tag": "marin-vllm-gpu-20260924-01911be34fac",
    "source_commit": "01911be34fac8715347962a8d791ca34a879e01e",
    "x86_64_wheel_sha256": "d04fa6af4a7161f8995154cbdbbc8dddc61b9a9ff198b3947985b2d998ef60e0",
    "router_bias_load": {
        "evidence": "source at the vLLM pin + safetensors headers + engine log markers; not a runtime readback",
        "summary": (
            "router bias FP32 (loader lines cited, not a runtime readback); router dot as served has FP32 input "
            "and FP32 weight parameter (whether cuBLAS uses TF32 for it is not verified); identical for both arms"
        ),
        "citations": {
            "vllm/model_executor/models/grugmoe.py:592-601": (
                "router = ReplicatedLinear(bias=True, params_dtype=torch.float32)"
            ),
            "vllm/model_executor/models/grugmoe.py:701-716": (
                "served CUDA forward: router_logits = _apply_grug_linear(self.router, x_flat.float()) (line 713); "
                "line 666 is the CPU-only reference path"
            ),
            "vllm/model_executor/layers/linear.py:371-384": (
                "ReplicatedLinear passes params_dtype to create_weights and to the bias Parameter, "
                "so the router weight is also an FP32 parameter (BF16 checkpoint values upcast exactly on copy_)"
            ),
            "vllm/model_executor/models/grugmoe.py:559": "routing adds router_logits.float() + self.bias.float()",
            "vllm/model_executor/models/grugmoe.py:1427-1459": "load_weights: param weight_loader(param, loaded_weight)",
            "vllm/model_executor/layers/linear.py:396-404": (
                "ReplicatedLinear.weight_loader: shape assert, param.data.copy_"
            ),
        },
        "file_sha256": {
            "vllm/model_executor/models/grugmoe.py": "ce919b5292ee636699c2145e4ff588696925dfb99a74a8c7c8f96748fbae92b9",
            "vllm/model_executor/layers/linear.py": "a3943e520e55c12b5048b821183978c60901ba78619ca21629692f403e1f3b4f",
        },
    },
}
SERVE_ARGS = [
    "--cluster", "marin", "--backend", "vllm", "--gpu", "H100x8", "--target-cluster", "cw-rno2a",
    "--vllm-source", "marin-fork", "--dtype", "bfloat16", "--tensor-parallel-size", "1",
    "--vllm-arg=--data-parallel-size=8", "--vllm-arg=--enable-expert-parallel",
    "--vllm-arg=--reasoning-parser=grug", "--vllm-arg=--max-num-seqs=256",
    "--max-model-len", "32768", "--max-num-batched-tokens", "2048",
    "--cpu", "32", "--memory", "512g", "--disk", "512g", "--timeout-hours", "1",
    "--max-retries-preemption", "0", "--no-cache", "--no-wait", "--priority-band", "production",
]  # fmt: skip
ENGINE_LOG_MARKERS = (
    "v0.0.0.dev20260924+marin.01911be34fac", "dtype=torch.bfloat16", "max_seq_len=32768",
    "tensor_parallel_size=1", "data_parallel_size=8", "quantization=None", "reasoning_parser='grug'",
    "use_ep=True", "FLASH_ATTN", "max_num_seqs': 256",
)  # fmt: skip
ENDPOINT_WAIT_SECONDS = 1800
# The iris CLI of this checkout, in this interpreter (the package has no __main__).
IRIS = [sys.executable, "-c", "import sys; from iris.cli import main; sys.argv[0] = 'iris'; main()", "--cluster=marin"]


def read(path: str) -> bytes:
    fs, p = filesystem_for(path)
    with fs.open(p, "rb") as f:
        return f.read()


def safetensors_dtypes(path: str) -> dict[str, str]:
    fs, p = filesystem_for(path)
    with fs.open(p, "rb", block_size=1 << 20) as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return {name: spec["dtype"] for name, spec in header.items() if name != "__metadata__"}


def bind_candidate(run_root: str) -> dict:
    result = json.loads(read(f"{run_root}/sft-result.json"))
    if result.get("completed_updates") != 16 or not result.get("native_checkpoint"):
        raise click.ClickException("sft-result.json is not a completed 16-update run with a native checkpoint")
    if result["hf_path"].rstrip("/") != f"{run_root}/hf/step-15":
        raise click.ClickException("sft-result.json hf_path is not <run root>/hf/step-15")
    audit = read(f"{result['hf_path']}/trained-precision-audit.json")
    return {
        "run_root": run_root,
        "sft_result_sha256": hashlib.sha256(read(f"{run_root}/sft-result.json")).hexdigest(),
        "completed_updates": result["completed_updates"],
        "native_checkpoint": result["native_checkpoint"],
        "hf_path": result["hf_path"].rstrip("/"),
        "trained_precision_audit_sha256": hashlib.sha256(audit).hexdigest(),
    }


def model_receipt(model: str) -> tuple[dict, dict[str, bytes]]:
    """The model's receipt, and the metadata bytes its file hashes were computed from."""
    metadata = {name: read(f"{model}/{name}") for name in BASELINE_FILE_HASHES}
    files = {name: hashlib.sha256(data).hexdigest() for name, data in metadata.items()}
    index = json.loads(read(f"{model}/model.safetensors.index.json"))
    fs, root = filesystem_for(model)
    shards = {}
    dtypes: dict[str, str] = {}
    for shard in sorted(set(index["weight_map"].values())):
        info = fs.info(f"{root}/{shard}")
        shards[shard] = {"size": info["size"], "etag": str(info.get("ETag") or info.get("etag")).strip('"')}
        dtypes.update(safetensors_dtypes(f"{model}/{shard}"))
    biases = {k: v for k, v in dtypes.items() if k.endswith(".mlp.router.bias")}
    others = {k: v for k, v in dtypes.items() if not k.endswith(".mlp.router.bias")}
    if len(dtypes) != TENSORS or len(biases) != ROUTER_BIASES or set(biases.values()) != {"F32"}:
        raise click.ClickException("Expected 502 tensors with 26 F32 router biases")
    if set(others.values()) != {"BF16"}:
        raise click.ClickException("Expected every non-bias tensor stored BF16")
    return {
        "model": model,
        "file_sha256": files,
        "shards": shards,
        "header_dtypes": {"tensors": len(dtypes), "router_bias_F32": len(biases), "BF16": len(others)},
    }, metadata


def accept_candidate_metadata(
    baseline: dict, candidate: dict, baseline_bytes: dict[str, bytes], candidate_bytes: dict[str, bytes]
) -> dict:
    """SHARED_MODEL_FILES must be byte-identical, except two differences the SFT export audit accepts.

    Tokenizer saving may embed the source ``chat_template.jinja`` in ``tokenizer_config.json``. A candidate whose
    ``tokenizer_config.json`` differs passes when ``check_export_auxiliary_metadata`` accepts it against the
    baseline with exactly that addition. The export may also rewrite ``generation_config.json`` with different
    JSON formatting; it passes when its parsed value equals the baseline's, as in the SFT audit. The check runs on
    local copies of the bytes hashed in the two receipts, so it reads the same routed objects. Returns the
    provenance fields that ``decide`` re-checks.
    """
    differ = [
        name for name in m16_gate.SHARED_MODEL_FILES if candidate["file_sha256"][name] != baseline["file_sha256"][name]
    ]
    raw_only = [name for name in differ if name not in ("tokenizer_config.json", m16_gate.REFORMATTED_FILE)]
    if raw_only:
        raise click.ClickException(f"Candidate {raw_only[0]} differs from the baseline")
    reformatted: list[str] = []
    if m16_gate.REFORMATTED_FILE in differ:
        name = m16_gate.REFORMATTED_FILE
        if json.loads(candidate_bytes[name]) != json.loads(baseline_bytes[name]):
            raise click.ClickException(f"Candidate {name} differs from the baseline")
        reformatted = [name]
    additions: list[str] = []
    if "tokenizer_config.json" in differ:
        with tempfile.TemporaryDirectory() as tmp:
            roots = {}
            for arm, files in (("baseline", baseline_bytes), ("candidate", candidate_bytes)):
                roots[arm] = Path(tmp) / arm
                roots[arm].mkdir()
                for name in m16_gate.SHARED_MODEL_FILES:
                    (roots[arm] / name).write_bytes(files[name])
            try:
                additions = check_export_auxiliary_metadata(str(roots["baseline"]), str(roots["candidate"]))
            except ValueError as e:
                raise click.ClickException(f"Candidate metadata refused: {e}") from None
        if additions != [m16_gate.TEMPLATE_ADDITION]:
            raise click.ClickException("Candidate tokenizer_config.json differs from the baseline")
    return {
        "accepted_metadata_additions": additions,
        "accepted_metadata_reformatted": reformatted,
        "metadata_acceptance": {
            "baseline_model": baseline["model"],
            "baseline_chat_template_sha256": baseline["file_sha256"]["chat_template.jinja"],
            "accepted_tokenizer_config_sha256": candidate["file_sha256"]["tokenizer_config.json"],
            "baseline_generation_config_sha256": baseline["file_sha256"][m16_gate.REFORMATTED_FILE],
            "accepted_generation_config_sha256": candidate["file_sha256"][m16_gate.REFORMATTED_FILE],
        },
    }


def bind_merge(model: str, audit_path: str, readback_path: str) -> dict:
    """A declared non-SFT candidate: a merged export, bound by its audit and readback receipts.

    It replaces only the SFT binding. The export is then checked like an SFT candidate (headers, metadata
    acceptance), served with the same SERVE_ARGS and scored by the same ``score_arm``; ``decide`` is unchanged.
    """
    receipts = {}
    for key, path in (("merge_audit", audit_path), ("merge_readback", readback_path)):
        try:
            data = read(path)
        except FileNotFoundError:
            raise click.ClickException(f"{key} receipt not found") from None
        receipts[key] = path
        receipts[f"{key}_sha256"] = hashlib.sha256(data).hexdigest()
    return {"candidate_kind": "merge", "hf_path": model.rstrip("/"), **receipts}


def preflight_arm(arm: str, run_root: str | None, merge: dict | None = None) -> dict:
    baseline, baseline_bytes = model_receipt(BASELINE_MODEL)
    if baseline["file_sha256"] != BASELINE_FILE_HASHES:
        raise click.ClickException("Baseline mirror metadata differs from the LR0 source hashes")
    if arm == "baseline":
        return {"arm": arm, "revision": BASELINE_REVISION, **baseline, "vllm": VLLM, "serve_args": SERVE_ARGS}
    if arm == "merge":
        if not merge:
            raise click.ClickException("--model, --merge-audit and --merge-readback are required for a merge arm")
        binding = bind_merge(merge["model"], merge["audit"], merge["readback"])
    elif run_root is None:
        raise click.ClickException("--run-root is required for the candidate arm")
    else:
        binding = bind_candidate(run_root.rstrip("/"))
    candidate, candidate_bytes = model_receipt(binding["hf_path"])
    accepted = accept_candidate_metadata(baseline, candidate, baseline_bytes, candidate_bytes)
    kind = "merge" if arm == "merge" else "sft"
    provenance = {"arm": "candidate", "candidate_kind": kind, "binding": binding, **candidate, **accepted}
    return {**provenance, "vllm": VLLM, "serve_args": SERVE_ARGS}


def cancel_record(returncode: int) -> dict:
    """The serve release as recorded in the receipt: cancelled only when the cancel command succeeded."""
    return {"cancel_rc": returncode, "cancelled_after_scoring": returncode == 0}


def idle_whole_h100_nodes() -> int:
    out = subprocess.run(
        [*IRIS, "rpc", "controller", "list-peers"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    data = json.loads(out[out.index("{") :])
    peer = next(p for p in data["peers"] if p["peer_id"] == "cw-rno2a")
    backend = peer["backends"][0]
    age = time.time() - int(backend["availability"]["observation_epoch_ms"]) / 1000
    if not peer["reachable"] or not 0 <= age < 60:
        raise click.ClickException("cw-rno2a capacity observation is stale or unreachable")
    nodes = backend["detail"]["kubernetes"]["nodes"]
    return sum(
        n["gpu_count"] == 8 and "H100" in n.get("gpu_model", "") and n["ready"] and n["schedulable"]
        and n["running_pods"] == 0
        for n in nodes
    )  # fmt: skip


@click.group()
def cli():
    pass


def merge_options(fn):
    fn = click.option("--model", default=None, help="Merged export path (merge arm).")(fn)
    fn = click.option("--merge-audit", default=None, help="Merge audit receipt (merge arm).")(fn)
    fn = click.option("--merge-readback", default=None, help="Merge readback receipt (merge arm).")(fn)
    return click.option("--label", default=None, help="Output label for a merge arm.")(fn)


def merge_spec(arm: str, model, merge_audit, merge_readback, label) -> dict | None:
    if arm != "merge":
        return None
    if not (model and merge_audit and merge_readback and label):
        raise click.ClickException("a merge arm needs --model, --merge-audit, --merge-readback and --label")
    return {"model": model, "audit": merge_audit, "readback": merge_readback, "label": label}


@cli.command()
@click.option("--arm", type=click.Choice(["baseline", "candidate", "merge"]), required=True)
@click.option("--run-root", default=None, help="SFT run root (candidate arm).")
@merge_options
@click.option("--out", required=True, type=click.Path(path_type=Path))
def preflight(arm, run_root, model, merge_audit, merge_readback, label, out):
    """Bind the arm and write its provenance; no GPU."""
    m16_gate.write_json(out, preflight_arm(arm, run_root, merge_spec(arm, model, merge_audit, merge_readback, label)))
    print(json.dumps({"arm": arm, "provenance": str(out), "status": "ok"}))


@cli.command()
@click.option("--arm", type=click.Choice(["baseline", "candidate", "merge"]), required=True)
@click.option("--run-root", default=None)
@merge_options
@click.option("--gate", "gate_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--prompt-dir", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--tokenizer", required=True)
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
def run(arm, run_root, model, merge_audit, merge_readback, label, gate_path, prompt_dir, tokenizer, out_dir):
    """Serve one arm in the production band, score it and release the serve."""
    gate = m16_gate.load_gate(gate_path)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    m16_gate.check_frozen_inputs(prompt_dir, tok)
    merge = merge_spec(arm, model, merge_audit, merge_readback, label)
    provenance = preflight_arm(arm, run_root, merge)
    tag = f"merge-{label}" if merge else arm
    idle = idle_whole_h100_nodes()
    if idle < 1:
        raise click.ClickException("No idle whole H100 node on cw-rno2a; not submitting")
    name = f"m16gate-{tag}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    endpoint_name = f"/serve/{name}"
    argv = [sys.executable, "-m", "marin.inference.iris_cli", provenance["model"], *SERVE_ARGS]
    argv += ["--name", name, "--endpoint-name", endpoint_name]
    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    provenance["serve"] = {"job": f"/karan/{name}", "submitted_utc": time.time(), "idle_whole_h100_nodes": idle}
    cap = Path(f"/tmp/{name}-endpoint-url")
    try:
        with connect_controller(cluster_name="marin") as info:
            with IrisClient.remote(info.url, workspace=Path.cwd(), credentials=info.credentials) as client:
                job = client.job(JobName.from_wire(f"/karan/{name}"))
                _wait_for_endpoint(client, job, endpoint_name, ENDPOINT_WAIT_SECONDS)
                token = client.mint_endpoint_token(endpoint_name, ttl=Duration.from_hours(1))
                fd = os.open(cap, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(token.capability_url.rstrip("/") + "\n")
        provenance["serve"]["ready_utc"] = time.time()
        endpoint = shared.Endpoint(cap.read_text().strip())
        receipt = m16_gate.score_arm(endpoint, tok, prompt_dir, gate, provenance["arm"], provenance)
    except Exception as exc:
        print(json.dumps({"status": "failed", "arm": arm, "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
    finally:
        cap.unlink(missing_ok=True)
        logs = subprocess.run(
            [*IRIS, "job", "logs", "--no-tail", "--max-lines", "50000", f"/karan/{name}"],
            capture_output=True, text=True,
        ).stdout  # fmt: skip
        cancel_rc = subprocess.run(
            [*IRIS, "job", "cancel", f"/karan/{name}"],
            capture_output=True,
        ).returncode  # fmt: skip
    receipt["provenance"]["engine_log_markers"] = {marker: marker in logs for marker in ENGINE_LOG_MARKERS}
    receipt["provenance"]["serve"].update(cancel_record(cancel_rc))
    m16_gate.write_json(out_dir / f"m16-gate-{tag}.json", receipt)
    print(json.dumps({"arm": arm, "primary_wins": receipt["primary_wins"]["won"], "status": "scored"}))


if __name__ == "__main__":
    cli()
