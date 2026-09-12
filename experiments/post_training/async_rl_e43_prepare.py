# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose E4.3 without submission; fail closed on the two frozen source mismatches."""

# Each stage runs in a different environment with optional Marin or trainer dependencies.
# ruff: noqa: PLC0415

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import yaml

CACHE = Path("/home/ahmad/.cache/oa")
MARIN = Path("/home/ahmad/oa/worktrees/marin/async-v2-e43-compose")
MSR = Path("/home/ahmad/oa/worktrees/MarinSkyRL/async-v2-e43-compose")
MARIN_SHA = "a453edb92b14da9a8406c005f6bc4c76d72bc34b"
MSR_SHA = "65fa9170dae2493960f365fe237ea5c5c5c71545"
CORRECTIONS = ("behavior_clip", "regular_no_tis", "regular_tis", "regular_mask", "bc_mask", "regular_m2")
AGES = (1, 2)
SEEDS = (17, 29)
PREFIX = "s3://marin-us-east-02a/marin"
OUTPUT = CACHE / "async-v2-e43-composition-v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_pins():
    for root, expected in ((MARIN, MARIN_SHA), (MSR, MSR_SHA)):
        assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == expected
        assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()


def recipe_argv(correction, age, seed, cluster):
    argv = [
        "--version",
        "2026.09.09.259",
        "--cluster",
        cluster,
        "--runner",
        "async",
        "--scale",
        "screening",
        "--stage",
        "rl",
        "--completion",
        "model",
        "--pool-artifact",
        "math-eval-pool@1.0.0-qwen-bucket-s1",
        "--updates",
        "96",
        "--eval-updates",
        "32",
        "--minibatches",
        "2",
        "--seed",
        str(seed),
        "--staleness",
        str(age),
        "--weight-sync-interval",
        "1",
        "--no-kl-loss",
        "--epoch-seeded-shuffle",
        "--correction",
        correction,
        "--generation-workers",
        "64",
        "--first-token-admission",
        "--dataloader-workers",
        "0",
        "--response-tokens",
        "1024",
        "--eval-response-tokens",
        "1024",
        "--context-tokens",
        "2048",
        "--grad-cosine",
        "--grad-cosine-store",
        "gpu_fp32",
        "--lr",
        "2e-6",
        "--serial-engine-startup",
        "--timeout-seconds",
        "4800",
        "--dry-run",
    ]
    if cluster == "cw-rno2a":
        argv += ["--allow-cross-region-io"]
    return argv


def stage_recipe():
    from click.testing import CliRunner

    from experiments.post_training import async_rl

    assert Path(async_rl.__file__).is_relative_to(MARIN)
    assert_pins()
    n2_path = CACHE / "async-v2-n2-rno-fresh-training-envelope.json"
    n2 = json.loads(n2_path.read_text())["request"]
    coverage_path = Path("/home/ahmad/oa-data/captures/async-rl-v2-optimizer/qwen-input-coverage-receipt.json")
    rows = []
    rno_refusal = CliRunner().invoke(async_rl.main, recipe_argv("behavior_clip", 1, 17, "cw-rno2a"))
    assert rno_refusal.exit_code != 0
    assert "Cross-region I/O is restricted to Qwen screening RL jobs on cw-rno2a" in rno_refusal.output
    missing = CliRunner().invoke(async_rl.main, recipe_argv("regular_m2", 1, 17, "cw-us-east-02a"))
    assert missing.exit_code != 0 and "regular_m2" in missing.output and "not one of" in missing.output
    for correction in CORRECTIONS:
        for age in AGES:
            for seed in SEEDS:
                tag = f"{correction}-a{age}-seed{seed}"
                row = dict(
                    tag=tag,
                    correction=correction,
                    configured_age=age,
                    seed=seed,
                    wave=1 if seed == 17 else 2,
                    gpus=16,
                    updates=96,
                    minibatches=2,
                    expected_consumed_groups=6144,
                    expected_consumed_sequences=24576,
                    expected_prepared_cohorts=48,
                    eval_updates=[0, 32, 64, 96],
                )
                rows.append(row)
                if correction == "regular_m2":
                    row.update(status="blocked_missing_frozen_preset_and_runtime", envelope=None)
                    continue
                argv = recipe_argv(correction, age, seed, "cw-us-east-02a")
                result = CliRunner().invoke(async_rl.main, argv)
                assert result.exit_code == 0, (tag, result.output, result.exception)
                preview = json.loads(result.output)["training"]["request"]
                request = copy.deepcopy(preview)
                for key in ("model", "train_data", "validation_data"):
                    old_blocks = n2[key] if isinstance(n2[key], list) else [n2[key]]
                    new_blocks = preview[key] if isinstance(preview[key], list) else [preview[key]]
                    for old, new in zip(old_blocks, new_blocks, strict=True):
                        for field in ("identity", "local_path"):
                            assert old[field] == new[field], (tag, field, old, new)
                        if "relative_path" in old:
                            assert old["relative_path"] == new["relative_path"]
                    request[key] = copy.deepcopy(n2[key])
                guard_uri = (
                    f"{PREFIX}/users/ahmad/documents/math-eval-pool/1.0.0-candidate1"
                    f"/qualification/e43-v1/measurement/{tag}.json"
                )
                config = yaml.safe_load(request["config_yaml"])
                config["trainer"]["measurement_guard_uri"] = guard_uri
                config["trainer"]["measurement_guard_resume_step"] = None
                request["config_yaml"] = yaml.safe_dump(config, sort_keys=False)
                science = {
                    key: request[key]
                    for key in (
                        "config_yaml",
                        "runtime",
                        "model",
                        "train_data",
                        "validation_data",
                        "topology",
                        "seed",
                        "overrides",
                        "completion_mode",
                    )
                }
                science_sha = digest(science)
                version = f"1.0.0-e43-v1-{tag}"
                base = f"users/ahmad/checkpoints/async-rl/e43-{science_sha[:12]}-training"
                output_path = f"{PREFIX}/{base}/{version}"
                temporary_path = f"s3://marin-us-east-02a/tmp/ttl=14d/skyrl/marin-us-east-02a/marin/{base}/{version}"
                for key in ("output", "overrides"):
                    request[key] = json.loads(
                        json.dumps(request[key])
                        .replace("<temporary_output_path>", temporary_path)
                        .replace("<output_path>", output_path)
                    )
                request["run_id"] = base + "-" + version
                request["attempt_id"] = hashlib.sha256((science_sha + ":attempt0").encode()).hexdigest()[:12]
                assert request["completion_mode"] == "checkpoint"
                assert request["runtime"]["commit"] == MSR_SHA
                job_stem = f"async-rl-v2-qwen-k12-p8-i8-{correction}-a{age}-{science_sha[:8]}-v1"
                execution = dict(
                    cluster="cw-us-east-02a",
                    cluster_config="lib/iris/config/cw-us-east-02a.yaml",
                    cpu=16,
                    memory="128GB",
                    disk="2TB",
                    priority="batch",
                    max_retries=1,
                    target_cluster=None,
                    parent_cluster_config=None,
                    wandb_entity="atqamar-oa",
                    timeout_seconds=4800,
                    job_name=job_stem,
                )
                envelope = dict(schema_version=2, request=request, execution=execution)
                row.update(
                    status="east_cpu_composed_not_launch_ready",
                    recipe_argv=argv,
                    envelope=envelope,
                    envelope_sha256=digest(envelope),
                    scientific_sha256=science_sha,
                    measurement_guard_uri=guard_uri,
                    coordinator_job_id="/atqamar/" + job_stem + "-coordinator",
                    native_job_id="/atqamar/" + job_stem + "-coordinator/" + execution["job_name"],
                )
                print("E43_N2_ACTUAL_CLI_PASS " + tag, flush=True)
    assert len(rows) == 24
    assert len({r["tag"] for r in rows}) == 24
    manifest = dict(
        status="E43_COMPOSITION_BLOCKED",
        lever_class="R",
        group_id="e43_correction_head_to_head",
        marin_sha=MARIN_SHA,
        msr_sha=MSR_SHA,
        arms=rows,
        gpu_submissions=0,
        declared_member_count=24,
        native_composed_count=0,
        age_settings=list(AGES),
        waves={str(w): [r["tag"] for r in rows if r["wave"] == w] for w in (1, 2)},
        inference_family=dict(
            comparisons=45,
            endpoint_contrasts=30,
            degradation_contrasts=15,
            seeds=list(SEEDS),
            conditioning="two observed training seeds",
        ),
        inputs={str(n2_path): sha(n2_path), str(coverage_path): sha(coverage_path)},
        counter_cases=dict(rno_checkpoint_refusal=rno_refusal.output, missing_m2_refusal=missing.output),
        pending=[
            "REG+M2 absent from both approved pins; historical quality screen remains FAIL",
            "RNO checkpoint completion rejected by frozen Marin; east alternative composed",
            "Native/Hydra preview stage",
            "Fresh controller and regional output absence checks",
            "Bound coordinator and held-out export/evaluation packet",
            "Measured N2 realised-age distinction and external source/quality audit",
        ],
    )
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "E43_FROZEN_SOURCE_BLOCKERS_REPRODUCED missing_m2=true rno_checkpoint_rejected=true arms_preserved=24",
        flush=True,
    )


def stage_native():
    from cloud.iris import iris_backend as backend
    from cloud.iris import training_driver as driver
    from cloud.iris.protocol import job_spec
    from cloud.iris.rl_config_translation import apply_context_budget_overrides, build_skyrl_hydra_args, parse_rl_config
    from cloud.iris.runtime_bundle import resolve_launcher_source
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from skyrl_train.utils.utils import validate_cfg

    assert_pins()
    assert Path(backend.__file__).is_relative_to(MSR)
    assert resolve_launcher_source().commit == MSR_SHA
    config_dir = MSR / "skyrl-train/skyrl_train/config"
    packet = json.loads(OUTPUT.read_text())
    evidence = []
    for row in packet["arms"]:
        envelope = row["envelope"]
        if envelope is None:
            continue
        assert digest(envelope) == row["envelope_sha256"]
        request = envelope["request"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as handle:
            handle.write(request["config_yaml"])
            handle.flush()
            argv = backend.job_launch_argv(job_spec(envelope), handle.name)
            index = argv.index("--cluster-config") + 1
            argv[index] = str(MARIN / argv[index])
            args = backend.resolved_launch_args(argv)
            with patch.object(backend, "_build_task_shell", side_effect=lambda _a, command, _p: command):
                controller = backend.build_task_command(args)
            command = controller[controller.index("--") + 1 :]
            values = vars(driver.create_parser().parse_args(command[3:]))
            values.update(
                rl_config_path=handle.name,
                train_data=driver.parse_list_arg(values["train_data"]),
                val_data=driver.parse_list_arg(values["val_data"]),
                skyrl_overrides=values["skyrl_override"] or [],
            )
            cfg = driver.LocalRLConfig(
                **{f.name: values[f.name] for f in fields(driver.LocalRLConfig) if f.name in values}
            )
            runner = driver.LocalRLRunner(cfg)
            parsed, overrides = apply_context_budget_overrides(
                parse_rl_config(handle.name, model_override=cfg.model_path), cfg.skyrl_overrides
            )
            generated = (
                build_skyrl_hydra_args(
                    parsed,
                    runner._build_exp_args(),
                    driver._LocalHPCStub(gpus_per_node=cfg.gpus, cpus_per_node=cfg.cpus),
                )
                + overrides
            )
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                composed = compose(config_name="ppo_base_config", overrides=generated)
            validate_cfg(composed)
            t = composed.trainer
            assert t.max_steps == t.ckpt_interval == 96
            assert t.eval_interval == 32 and t.eval_before_train
            assert t.train_batch_size == 128 and t.policy_mini_batch_size == 64
            assert t.fully_async.first_token_admission and t.fully_async.weight_sync_interval == 1
            assert t.fully_async.max_staleness_steps == row["configured_age"]
            assert t.fully_async.num_parallel_generation_workers == 64
            assert t.measurement_guard_uri == row["measurement_guard_uri"]
            assert t.algorithm.grad_cosine.enabled and t.algorithm.loss_reduction == "token_mean"
            assert not t.algorithm.use_kl_loss and t.policy.optimizer_config.lr == 2e-6
            assert t.seed == row["seed"] and t.hf_hub_repo_id is None
            assert t.policy.megatron_config.tensor_model_parallel_size == 2
            assert composed.data.num_workers == 0 and composed.data.epoch_seeded_shuffle
            assert composed.generator.num_inference_engines == 8
            assert composed.generator.inference_engine_tensor_parallel_size == 1
            assert composed.generator.n_samples_per_prompt == 4
            assert t.update_epochs_per_batch == 1 and t.strategy == "megatron"
            assert parsed.entrypoint == "skyrl_train.entrypoints.fully_async"
            expected_loss = "behavior_clip" if row["correction"] in ("behavior_clip", "bc_mask") else "regular"
            assert t.algorithm.policy_loss_type == expected_loss
            assert t.algorithm.use_tis == (row["correction"] == "regular_tis")
            if t.algorithm.use_tis:
                assert t.algorithm.tis_imp_ratio_cap == 2.0
            masked = row["correction"] in ("regular_mask", "bc_mask")
            assert bool(OmegaConf.select(t, "algorithm.offpolicy_mask.enabled", default=False)) == masked
            if masked:
                assert t.algorithm.offpolicy_mask.low == 0.5 and t.algorithm.offpolicy_mask.high == 5.0
                assert t.algorithm.offpolicy_mask.ratio == "mismatch" and not t.algorithm.offpolicy_mask.renormalize
            evidence.append(
                dict(
                    tag=row["tag"],
                    envelope_sha256=row["envelope_sha256"],
                    hydra_args=generated,
                    composed=OmegaConf.to_container(composed, resolve=True),
                )
            )
            print("E43_N2_NATIVE_HYDRA_PASS " + row["tag"] + " args=" + str(len(generated)), flush=True)
    destination = CACHE / "async-v2-e43-native-compose-v1.json"
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    packet["native_composed_count"] = len(evidence)
    packet["pending"] = [item for item in packet["pending"] if item != "Native/Hydra preview stage"]
    packet["native_receipt"] = dict(path=str(destination), sha256=sha(destination))
    OUTPUT.write_text(json.dumps(packet, indent=2) + "\n")
    print("E43_PARTIAL_COMPOSITION_PASS configured=20 missing_method=4 declared=24 launch_ready=false", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("recipe", "native"), required=True)
    selected = parser.parse_args().stage
    stage_recipe() if selected == "recipe" else stage_native()
