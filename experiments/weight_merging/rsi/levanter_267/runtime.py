# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job-local model loading, synchronized serving, and cached item-303 capture."""

import hashlib
import json
import logging
import threading
import time
import urllib.request
import uuid

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.jax.init import initialize_jax
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.multihost import MultiHostInferenceEngine
from levanter.inference.openai import InferenceContext, InferenceServer, InferenceServerConfig
from levanter.models.snowball import GrugMoeHfConfig, SnowballConfig, SnowballLMHeadModel
from levanter.tokenizers import load_tokenizer
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.weight_merging.rsi.levanter_267.compile_profile import instrument_prefill
from experiments.weight_merging.rsi.levanter_267.protocol import RUN_PROVENANCE_PATH, inference_source_digest


class ProbeEngine(MultiHostInferenceEngine):
    """Retain raw IDs so the smoke can verify HTTP usage and stop stripping."""

    def generate(self, requests, step_callback=None):
        result = super().generate(requests, step_callback=step_callback)
        self.last_generated_tokens = result.tokens
        counts = getattr(self, "batch_counts", {})
        key = str(len(requests))
        counts[key] = counts.get(key, 0) + 1
        self.batch_counts = counts
        print(
            "PRODUCTION_BATCH", json.dumps(dict(size=len(requests), total_generated=result.total_generated)), flush=True
        )
        return result


def _post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def _save(output, name, value):
    data = json.dumps(value, indent=2).encode()
    (StoragePath(output) / name).write_bytes(data)
    print("ARTIFACT", str(StoragePath(output) / name), flush=True)


def _record_memory(output, stage):
    devices = [dict(device_id=device.id, stats=device.memory_stats()) for device in jax.local_devices()]
    _save(output, f"memory-{stage}-rank{jax.process_index()}.json", devices)
    print("DEVICE_MEMORY", stage, json.dumps(devices), flush=True)


def _load_model(mode, model_path, tokenizer, max_seq_len):
    if mode == "tiny":
        cfg = SnowballConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=128,
            intermediate_dim=128,
            shared_expert_intermediate_dim=128,
            num_experts=32,
            num_layers=5,
            max_seq_len=max_seq_len,
            num_heads=20,
            num_kv_heads=5,
            head_dim=128,
        )
        model = SnowballLMHeadModel.init(hax.Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(13))
        model = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) else x, model)
        # Router biases are fixed selection thresholds; retain their FP32 contract.
        blocks = tuple(
            eqx.tree_at(lambda b: b.mlp.router_bias, b, b.mlp.router_bias.astype(jnp.float32))
            for b in model.transformer.blocks
        )
        model = eqx.tree_at(lambda m: m.transformer.blocks, model, blocks)
    else:
        converter = HFCheckpointConverter(
            SnowballConfig,
            reference_checkpoint=RepoRef(model_path),
            tokenizer=tokenizer,
            HfConfigClass=GrugMoeHfConfig,
        )
        # Preserve checkpoint dtypes, particularly the fixed FP32 routing biases.
        model = converter.load_pretrained(SnowballLMHeadModel, dtype=None, axis_mapping={})
    return model


def _validate_numerical_input(tokenizer, item, output):
    text = item["prompt"] + item["completion"]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_sha = hashlib.sha256(text.encode()).hexdigest()
    matches = input_sha == item["expected_input_sha256"] and tokens == item["expected_token_ids"]
    _save(
        output,
        "input-validation.json",
        dict(
            input_sha256=input_sha,
            expected_input_sha256=item["expected_input_sha256"],
            token_ids=tokens,
            expected_token_count=len(item["expected_token_ids"]),
            oracle_sha256=item["oracle_sha256"],
            matched=matches,
        ),
    )
    if not matches or len(tokens) != 5279:
        raise ValueError("Item303 input hash or token IDs differ from the pinned oracle")
    return tokens


def _capture_numerics(engine, tokenizer, item, output, config, local):
    tokens = _validate_numerical_input(tokenizer, item, output)
    start = time.monotonic()
    result = engine.score_token_logprobs(tokens, 1)
    elapsed = time.monotonic() - start
    rows = []
    for position, (token, logprob, tops) in enumerate(
        zip(tokens, result.token_logprobs, result.top_token_logprobs, strict=True)
    ):
        top_id, top_value = max(tops.items(), key=lambda pair: pair[1])
        rows.append(
            dict(
                position=position,
                token_id=token,
                token=tokenizer.decode([token], skip_special_tokens=False),
                logprob=None if position == 0 else logprob,
                top1_token_id=None if position == 0 else top_id,
                top1_logprob=None if position == 0 else top_value,
            )
        )
    _save(
        output,
        "303.json",
        dict(
            id=303,
            input_sha256=item["expected_input_sha256"],
            tokens=rows,
            elapsed_seconds=elapsed,
            tokens_per_second=(len(tokens) - 1) / elapsed,
            usage=dict(prompt_tokens=len(tokens)),
            dropped_assignments=np.asarray(engine.gen_state.cache.dropped_assignments).tolist(),
        ),
    )
    _probe_long_decode(engine, tokenizer, tokens, result.top_token_logprobs[2050], output, config, local)


def _probe_long_decode(engine, tokenizer, tokens, next_token_logprobs, output, config, local):
    # Exercise the production scheduler's padded decode path after a >2048-token prompt too.
    long_prompt = tokenizer.decode(tokens[:2050], skip_special_tokens=False)
    if tokenizer.encode(long_prompt, add_special_tokens=False) != tokens[:2050]:
        raise ValueError("Long-decode prompt does not round-trip through the tokenizer")
    start = time.monotonic()
    long_smoke = _post(
        local + "/v1/completions",
        dict(
            model=config.model_name,
            prompt=long_prompt,
            max_tokens=2,
            temperature=0,
            seed=19311,
        ),
    )
    raw_ids = engine.last_generated_tokens[0]
    long_smoke["generated_token_ids"] = raw_ids
    expected_first = max(next_token_logprobs, key=next_token_logprobs.get)
    long_smoke["cached_first_token_id"] = expected_first
    long_smoke["first_token_matches_cached_score"] = bool(raw_ids and raw_ids[0] == expected_first)
    long_smoke["wall_seconds"] = time.monotonic() - start
    _save(output, "long-decode.json", long_smoke)
    count = long_smoke["usage"]["completion_tokens"]
    finish = long_smoke["choices"][0]["finish_reason"]
    legitimate_stop = count > 0 and raw_ids[-1] in config.eos_token_ids and finish == "stop"
    if (
        long_smoke["usage"]["prompt_tokens"] != 2050
        or count != len(raw_ids)
        or (count != 2 and not legitimate_stop)
        or not long_smoke["first_token_matches_cached_score"]
    ):
        raise RuntimeError("Long-prompt decode failed continuation count, EOS, or cached-first-token consistency")


def _server_config(trainer, model_path, max_seq_len, max_seqs):
    info = get_job_info()
    page_size = 128
    pages = (max_seq_len + page_size - 1) // page_size
    generation_config = json.loads((StoragePath(model_path) / "generation_config.json").read_text())
    eos_ids = generation_config["eos_token_id"]
    eos_ids = (eos_ids,) if isinstance(eos_ids, int) else tuple(eos_ids)
    print("PINNED_EOS_IDS", json.dumps(eos_ids), flush=True)
    return InferenceServerConfig(
        eos_token_ids=eos_ids,
        trainer=trainer,
        model_name="snowball-levanter",
        host="0.0.0.0",
        port=info.ports["http"],
        service=InferenceEngineConfig(
            max_seq_len=max_seq_len,
            page_size=page_size,
            max_pages=(pages + 2) * max_seqs,
            max_stop_seqs=8,
            max_seqs=max_seqs,
            max_seqs_in_prefill=max_seqs,
            max_prefill_size=256,
            max_queued_tokens=max_seq_len * max_seqs,
            max_tokens_per_round=16,
            max_rounds=8,
        ),
    )


def _serve_and_probe(engine, config, model_path, output, item, mode, source_commit):
    info = get_job_info()
    model, tokenizer = engine.model, engine.tokenizer
    context = InferenceContext(model, tokenizer, engine, config)
    app = InferenceServer._create_app(context)
    provenance = dict(
        run_id=uuid.uuid4().hex,
        source_commit=source_commit,
        serve_job=str(info.job_id),
        model_path=model_path,
        model_revision=model_path.rstrip("/").rsplit("@", 1)[-1],
        inference_source_sha256=inference_source_digest(),
        max_seq_len=config.service.max_seq_len,
        max_seqs=config.service.max_seqs,
        max_pages=config.service.max_pages,
    )
    _save(output, "run-provenance.json", provenance)

    @app.get(RUN_PROVENANCE_PATH)
    async def run_provenance():
        return dict(provenance, batch_counts=dict(getattr(engine, "batch_counts", {})))

    server = InferenceServer(config, context, app)
    context.start()
    http_thread = threading.Thread(target=server.serve, daemon=True)
    http_thread.start()
    local = f"http://127.0.0.1:{config.port}"
    registration = None
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if server._server is not None and server._server.started:
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("HTTP server did not start")
        registration = iris_ctx().registry.register(
            "levanter267-http",
            f"http://{info.advertise_host}:{config.port}",
            metadata={"model": config.model_name},
        )
        with urllib.request.urlopen(local + "/v1/models", timeout=30) as response:
            _save(output, "models.json", json.load(response))
        # Invalid deterministic requests must not broadcast and strand the followers.
        try:
            engine.score_token_logprobs([0], 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed scoring request was accepted")
        start = time.monotonic()
        smoke = _post(
            local + "/v1/completions",
            dict(
                model=config.model_name,
                prompt="The answer to 2 + 2 is",
                max_tokens=4,
                temperature=0,
                seed=19311,
            ),
        )
        _save(output, "first-token.json", dict(response=smoke, wall_seconds=time.monotonic() - start))
        print("FIRST_SERVED_TOKEN", json.dumps(smoke), flush=True)
        if item is not None:
            _capture_numerics(engine, tokenizer, item, output, config, local)
        _record_memory(output, "gates-complete")
        if mode == "serve":
            print("AWAITING_EXTERNAL_NUMERICS_REVIEW; evaluator requires an accepted gate report", flush=True)
            http_thread.join()
    finally:
        if registration is not None:
            iris_ctx().registry.unregister(registration)
        if server._server is not None:
            server._server.should_exit = True
        http_thread.join(timeout=10)
        if http_thread.is_alive():
            raise RuntimeError("HTTP server did not shut down; refusing to desynchronize workers")
        engine.shutdown_workers()


def run(mode, model_path, output, item, max_seq_len, source_commit, max_seqs=1):
    logging.basicConfig(level=logging.INFO)
    initialize_jax()
    if jax.process_count() != 4 or jax.device_count() != 16:
        raise RuntimeError(f"Expected 4 processes / 16 chips, got {jax.process_count()} / {jax.device_count()}")
    trainer = TrainerConfig(
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(
            axes={"data": 1, "context": 1, "expert": 16, "model": 1},
            dcn_axes={"replica_dcn": 1},
            batch_axis_name=None,
            param_mapping={},
        ),
    )
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        start = time.monotonic()
        tokenizer = load_tokenizer(model_path)
        if jax.process_index() == 0:
            files = (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "chat_template.jinja",
            )
            hashes = {name: hashlib.sha256((StoragePath(model_path) / name).read_bytes()).hexdigest() for name in files}
            _save(output, "tokenizer-provenance.json", dict(model_path=model_path, sha256=hashes))
        model = _load_model(mode, model_path, tokenizer, max_seq_len)
        jax.block_until_ready(model)
        print(
            "MODEL_READY",
            json.dumps(dict(seconds=time.monotonic() - start, mesh=dict(trainer.device_mesh.shape))),
            flush=True,
        )
        _record_memory(output, "model-loaded")
        config = _server_config(trainer, model_path, max_seq_len, max_seqs)
        instrument_prefill(output)
        engine = ProbeEngine.from_model_with_config(model, tokenizer, config.service, axis_resources={})
        if jax.process_index() != 0:
            engine.worker_loop()
            _record_memory(output, "complete")
            return
        _serve_and_probe(engine, config, model_path, output, item, mode, source_commit)
        _record_memory(output, "complete")
        _save(output, "complete.json", dict(mode=mode, success=True))
