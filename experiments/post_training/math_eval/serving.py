# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a frozen rating panel inside one externally time-bounded Iris GPU task."""

import hashlib
import importlib.metadata
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.serve import local_inference
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.audit_overlay import validated_statuses
from experiments.post_training.math_eval.pool import MODEL_TEMPLATES, canonical_json
from experiments.post_training.math_eval.pool_audit import verify_verifier_sources
from experiments.post_training.math_eval.rate import MODEL_PROFILES
from experiments.post_training.math_eval.scoring import SEMANTIC_DEPENDENCIES
from experiments.post_training.math_eval.serving_records import completion_request, completion_rows, serving_metrics

EAST_PREFIX = "s3://marin-us-east-02a/marin/"
SERVING_MODELS = {
    "qwen": {
        "weights": EAST_PREFIX + "users/ahmad/models/async-rl-qwen3-0.6b/2026.09.08.83/hf",
        "identity": "users/ahmad/models/async-rl-qwen3-0.6b@2026.09.08.83:8a30d2b5",
        "config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    },
    "snowball": {
        "weights": EAST_PREFIX + "exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm",
        "identity": "models/snowball-67b-a2b-sft-s2-thinking@2026.08.30:c6168770",
        "config_sha256": "7186b37670787e8e842f8fe0be20625f4b3bea6893f86f505d987babb3c4d9c0",
    },
}


@dataclass(frozen=True)
class RatingServingConfig:
    model: str
    pool_uri: str
    overlay_uri: str
    overlay_sha256: str
    manifest_sha256: str
    expected_ids: tuple[str, ...]
    samples: int
    output_uri: str
    source_commit: str
    tensor_parallel_size: int
    concurrency: int = 8
    startup_timeout_seconds: int = 600
    request_timeout_seconds: int = 600

    def __post_init__(self):
        if self.model not in SERVING_MODELS or self.samples not in (4, 8):
            raise ValueError("Unknown rating model or prescribed K")
        if self.model == "qwen" and self.samples != 8:
            raise ValueError("Qwen ratings require K=8")
        if any(not uri.startswith(EAST_PREFIX) for uri in (self.pool_uri, self.overlay_uri, self.output_uri)):
            raise ValueError("Rating artifacts must remain in east")
        if not self.expected_ids or len(set(self.expected_ids)) != len(self.expected_ids):
            raise ValueError("Rating membership must be nonempty and unique")
        if any(
            not re.fullmatch(r"[a-f0-9]{64}", value)
            for value in (*self.expected_ids, self.overlay_sha256, self.manifest_sha256)
        ):
            raise ValueError("Rating membership and artifact hashes must be explicit")
        if not re.fullmatch(r"[a-f0-9]{40}", self.source_commit):
            raise ValueError("Rating source must be a full commit")
        if self.tensor_parallel_size not in (1, 8) or (self.model == "snowball" and self.tensor_parallel_size != 8):
            raise ValueError("Rating tensor parallelism is outside the reviewed single-host profile")
        if not 1 <= self.concurrency <= 32 or min(self.startup_timeout_seconds, self.request_timeout_seconds) <= 0:
            raise ValueError("Rating concurrency and timeouts must be bounded")


def serving_configuration(config):
    """Construct explicit engine parameters without artifact reads or GPU imports."""
    if VLLM_GPU_RELEASE.source_commit != "f0d7cc7f587482e0ab771e3c9715e726eb914e60":
        raise ValueError("Serving engine changed; qualify its renderer before rating")
    profile = MODEL_PROFILES[config.model]
    model = ServedModelConfig(
        weights=SERVING_MODELS[config.model]["weights"],
        api_model=f"rating-{config.model}",
        dtype="bfloat16",
        max_model_len=profile["max_prompt_tokens"] + profile["max_response_tokens"],
        tensor_parallel_size=config.tensor_parallel_size,
    )
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        startup_timeout_seconds=config.startup_timeout_seconds,
        max_num_seqs=config.concurrency * config.samples,
        extra_args=("--seed", "17", "--generation-config", "vllm"),
    )
    return model, engine


def serving_specification(config):
    model, engine = serving_configuration(config)
    engine_record = asdict(engine) | {"extra_metric_families": sorted(engine.extra_metric_families)}
    return json.loads(json.dumps({"config": asdict(config), "model": asdict(model), "engine": engine_record}))


def verify_wheel_receipt(log_text, version):
    """Validate the installed URL/extension/ABI proof, not an unmeasured wheel digest."""
    marker = "MARIN_VLLM_WHEEL_VERIFIED="
    evidence = [json.loads(line.split(marker, 1)[1]) for line in log_text.splitlines() if marker in line]
    if len(evidence) != 1 or version != VLLM_GPU_RELEASE.version:
        raise ValueError("Native wheel proof or API runtime version is missing or ambiguous")
    wheel = next(item for item in VLLM_GPU_RELEASE.wheels if item.architecture == "x86_64")
    record = evidence[0]
    expected = {
        "release_tag": VLLM_GPU_RELEASE.release_tag,
        "source_commit": VLLM_GPU_RELEASE.source_commit,
        "version": VLLM_GPU_RELEASE.version,
        "wheel_sha256": wheel.sha256,
        "wheel_url": wheel.url,
        "sm_targets": list(wheel.sm_targets),
        "compute_capability": "9.0",
    }
    if any(record.get(key) != value for key, value in expected.items()) or not record.get("extension_path"):
        raise ValueError("Native serving runtime differs from the reviewed H100 wheel")
    return record | {"wheel_digest_verification": "promoted_asset_provenance_only", "api_version": version}


def load_rating_inputs(config):
    """Revalidate fixed question membership and accepted status before generation."""
    prefix = StoragePath(config.pool_uri)
    manifest = pq.read_table(pa.BufferReader((prefix / "manifest.parquet").read_bytes())).to_pylist()
    selection = json.loads((prefix / "selection.json").read_bytes())
    overlay = json.loads(StoragePath(config.overlay_uri).read_bytes())
    statuses, overlay_sha = validated_statuses(manifest, selection, overlay)
    if (
        hashlib.sha256(canonical_json(manifest).encode()).hexdigest() != config.manifest_sha256
        or selection["manifest_sha256"] != config.manifest_sha256
    ):
        raise ValueError("Rating manifest changed")
    if overlay_sha != config.overlay_sha256:
        raise ValueError("Rating acceptance overlay changed")
    lookup = {row["prompt_sha256"]: row for row in manifest}
    if any(digest not in lookup or statuses[digest] != "accept" for digest in config.expected_ids):
        raise ValueError("Rating membership includes an unknown or unaccepted question")
    return [lookup[digest] for digest in config.expected_ids]


def run_rating_serving(config):
    """Produce raw-token dumps; independent terminal/harness audit is still required.

    The submitting Iris envelope must set a native task timeout and zero retries.
    There are no child GPU jobs: model startup, requests and server cleanup all
    execute within the same task lifetime.
    """
    job = get_job_info()
    if job is None or job.attempt_id != 0:
        raise ValueError("Ratings require a first-attempt Iris task")
    description = iris_ctx().client.describe_task(job.task_id)
    status, resources = description.status, description.resources
    device = resources.device
    if (
        status.execution_cluster_id != "cw-us-east-02a"
        or str(status.task_id) != str(job.task_id)
        or status.current_attempt_number != 0
        or len(status.attempts) != 1
        or device is None
        or device.kind != "gpu"
        or device.variant != "H100"
        or device.count != config.tensor_parallel_size
    ):
        raise ValueError("Controller allocation differs from the reviewed east H100 task")
    allocation = {
        "cluster": status.execution_cluster_id,
        "resources": asdict(resources),
        "attempt_uid": status.attempts[0].attempt_uid,
        "started_at_ms": None if status.attempts[0].started_at is None else status.attempts[0].started_at.epoch_ms,
    }
    output = StoragePath(config.output_uri)
    if output.exists():
        raise ValueError("Refusing to overwrite any previous rating attempt")
    verify_verifier_sources()
    score_versions = {name: importlib.metadata.version(name) for name in (*SEMANTIC_DEPENDENCIES, "reasoning-gym")}
    if score_versions != (SEMANTIC_DEPENDENCIES | {"reasoning-gym": "0.1.25"}):
        raise ValueError("Serving scorer dependency versions differ from the frozen verifier")
    items = load_rating_inputs(config)
    model, engine = serving_configuration(config)
    model_config = (StoragePath(model.weights) / "config.json").read_bytes()
    tokenizer_bytes = (StoragePath(model.weights) / "tokenizer.json").read_bytes()
    if hashlib.sha256(model_config).hexdigest() != SERVING_MODELS[config.model]["config_sha256"]:
        raise ValueError("Downloaded model configuration differs from the audited export")
    profile = MODEL_PROFILES[config.model]
    if hashlib.sha256(tokenizer_bytes).hexdigest() != profile["tokenizer_sha256"]:
        raise ValueError("Downloaded tokenizer differs from the frozen contract")
    decoder = Tokenizer.from_str(tokenizer_bytes.decode())
    requests_by_question = [
        completion_request(item, decoder, model=config.model, samples=config.samples, api_model=model.model_id)
        for item in items
    ]
    # Standard serving handles the immutable object-store model path directly.
    specification = serving_specification(config)
    with local_inference(model, engine, num_chips=config.tensor_parallel_size) as session:
        endpoint = session.model.endpoint.base_url
        version_reply = requests.get(endpoint.removesuffix("/v1") + "/version", timeout=30)
        version_reply.raise_for_status()
        runtime = verify_wheel_receipt(session._served.environment.logs(), version_reply.json()["version"])
        native_command = list(session._served.environment._command)

        def generate(index):
            session.check_alive()
            response = requests.post(
                endpoint + "/completions", json=requests_by_question[index], timeout=config.request_timeout_seconds
            )
            response.raise_for_status()
            return response.json()

        # Only HTTP runs in worker threads. Keep scoring on the main thread,
        # where the semantic verifier stack can safely use POSIX timeouts.
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            rows = []
            for index, response in enumerate(executor.map(generate, range(len(items)))):
                rows.extend(
                    completion_rows(
                        items[index],
                        requests_by_question[index],
                        response,
                        decoder,
                        model=config.model,
                        question_index=index,
                    )
                )
        session.check_alive()
    metrics = serving_metrics(rows, samples=config.samples)
    dump = output / "dumped_evals" / "global_step_0_evals"
    dump.mkdirs(exist_ok=False)
    raw_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    (dump / "rows.jsonl").write_bytes(raw_bytes)
    (dump / "aggregated_results.jsonl").write_text(json.dumps(metrics, sort_keys=True))
    receipt = {
        "schema": "math_eval_serving_generation_v1",
        "specification": specification,
        "specification_sha256": audit.canonical_sha(specification),
        "runtime": runtime,
        "score_dependency_versions": score_versions,
        "native_command": native_command,
        "task_id": str(job.task_id),
        "attempt_id": job.attempt_id,
        "attempt_uid": allocation["attempt_uid"],
        "worker_region": allocation["cluster"],
        "controller_allocation": allocation,
        "bundle_id": job.bundle_id,
        "model_identity": SERVING_MODELS[config.model]["identity"],
        "model_config_sha256": hashlib.sha256(model_config).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "prompt_template_id": MODEL_TEMPLATES[config.model].template_id,
        "engine_global_seed": 17,
        "request_sampling_seed": None,
        "expected_ids_sha256": audit.canonical_sha(sorted(config.expected_ids)),
        "ordered_request_sha256": audit.canonical_sha(
            [audit.canonical_sha(request) for request in requests_by_question]
        ),
        "raw_rows_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "rows": len(rows),
        "metrics": metrics,
        "requires_independent_terminal_and_harness_audit": True,
    }
    (output / "generation.json").write_text(json.dumps(receipt, sort_keys=True))
    print(
        "RATING_GENERATION_COMPLETE " + json.dumps({"receipt_sha256": audit.canonical_sha(receipt), "rows": len(rows)})
    )
    return receipt
