# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline eval-only flow; only model and installed-process execution are scripted."""

import gzip
import importlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import ray
import yaml
from cloud.iris.launch import LaunchOutcome, execute_launch
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from omegaconf import OmegaConf
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import Prompt
from skyrl_gym.task_factories import episode_factories
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.workers import RolloutWorkerPool, RolloutWorkerResources
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, _orphan_reasons
from skyrl_train.trajectory_runners.trajectory_retention_config import parse_trajectory_retention_config
from taskcompendium.grader import verifyit_package
from taskcompendium.models import ConversationInput, TextMessage
from tests.cpu.task_specs import lowered_task, task_input, text_task
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verifyit.spec import NumericSpec


class ScriptedEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model_name = "cpu-fixture"

    async def tokenize(self, request):
        return {"tokens": self.tokenizer.encode("six plus six", add_special_tokens=False)}

    async def chat_completion(self, request):
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "12"},
                    "token_ids": [3],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -0.1}]},
                }
            ]
        }

    async def generate(self, request):
        ids = request["prompt_token_ids"]
        return {
            "responses": ["12"] * len(ids),
            "assistant_messages": [{"role": "assistant", "content": "12"}] * len(ids),
            "prompt_ids": ids,
            "response_ids": [[3]] * len(ids),
            "response_logprobs": [[-0.1]] * len(ids),
            "stop_reasons": ["stop"] * len(ids),
        }


class ScriptedInstalledProcess:
    async def startup(self):
        pass

    async def shutdown(self):
        pass

    def original_harness(self, lowered, session_id, settings, phase):
        async def agent(context, complete):
            await complete(Prompt(messages=({"role": "user", "content": "six plus six"},), options={}))

        return AgentHarness(agent)


@dataclass
class WorkerSpec:
    config: object

    def build(self, tokenizer, shard):
        cfg = self.config.generator
        return TaskRolloutWorker(
            cfg,
            WholeTaskProjection(cfg, tokenizer),
            DirectModelClient(ScriptedEngine(tokenizer)),
            episode_factories(),
            harbor=HarborTaskSettings.from_config(OmegaConf.create({})),
            installed_agents=ScriptedInstalledProcess(),
            shutdown_timeout=10,
        )


class TinyDataset:
    def __init__(self):
        self.rows = []
        for name, harness in [("native", ASSISTANT_PREFIX_HARNESS), ("installed", HARBOR_TRIAL_SESSION)]:
            task = text_task(name, prompt="six plus six", source=name).model_copy(
                update={
                    "grader": verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
                    "tags": ("harbor",) if name == "installed" else (),
                }
            )
            if name == "native":
                task = task.model_copy(
                    update={
                        "context": ConversationInput(
                            events=(*task.context.events, TextMessage(role="assistant", content="Answer:"))
                        )
                    }
                )
            self.rows.append({"input": task_input(lowered_task(task, harness, max_turns=1))})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    @staticmethod
    def collate_fn(rows):
        return rows


class LocalTracker:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def log(self, results, **kwargs):
        pass


class CPUEntry(EvalOnlyEntrypoint):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = tokenizer_from_config(cfg)
        self.eval_dataset = TinyDataset()

    def create_inference_engine_client(self, **kwargs):
        return ScriptedEngine(self.tokenizer)

    def get_trajectory_runner(self, cfg, tokenizer, client):
        return RolloutWorkerPool(WorkerSpec(cfg), RolloutWorkerResources(1, 1, 2, 90, 0))

    def get_tracker(self):
        return LocalTracker()


def configuration(root):
    tokenizer_core = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "12": 3}, unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core, eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
    )
    tokenizer.chat_template = "{% for m in messages %}{{m.content}} {% endfor %}"
    tokenizer.save_pretrained(root / "tokenizer")
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.policy.model.tokenizer_path = str(root / "tokenizer")
    cfg.trainer.policy.model.lora.adapter_path = None
    cfg.trainer.eval_num_prompts = None
    cfg.trainer.eval_batch_size = 2
    cfg.trainer.evaluation_runner = "training"
    cfg.trainer.dump_eval_results = True
    cfg.trainer.export_path = str(root / "exports")
    cfg.trajectory_runner.max_concurrent_tasks = 2
    cfg.generator.backend = "vllm"
    cfg.generator.max_input_length = 256
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.generator.sampling_params.max_generate_length = 16
    cfg.generator.eval_sampling_params.max_generate_length = 16
    cfg.generator.trajectory_retention = {
        "enabled": True,
        "required": True,
        "output_path": str(root / "traces"),
        "run_id": "cpu-eval",
        "phases": ["eval"],
        "sample_count_per_step": 0,
        "sample_fraction": 0,
        "eval_sample_fraction": 1,
        "always_retain_failures": False,
        "always_retain_non_terminating": False,
        "always_retain_loops": False,
        "max_bytes_per_step": None,
        "max_bytes_per_run": None,
        "model_path": "/committed/global_step_3/policy",
        "model_source_identity": "fixture-checkpoint-sha256",
        "model_checkpoint_step": 3,
    }
    return cfg


def verify_retention(root, cfg):
    records = []
    for path in Path(cfg.generator.trajectory_retention.output_path).rglob("*.zip"):
        assert "step=eval_only" in str(path)
        with zipfile.ZipFile(path) as archive:
            records.extend(
                json.loads(gzip.decompress(archive.read(name)))
                for name in archive.namelist()
                if name.endswith(".json.gz")
            )
    assert len(records) == 2
    assert {record["trajectory"]["environment_extras"]["data_source"] for record in records} == {"native", "installed"}
    config = parse_trajectory_retention_config(OmegaConf.to_container(cfg.generator.trajectory_retention))
    for record in records:
        assert record["global_step"] is None
        assert record["provenance"]["model_version_step"] == 3
        assert record["provenance"]["model_source_identity"] == "fixture-checkpoint-sha256"
        assert _orphan_reasons(record, config)
    sink = TrajectorySink(config, tokenizer_from_config(cfg))
    try:
        assert len(sink._ledger.records) == 2
    finally:
        sink.close()
    (root / "retention-verified.json").write_text(json.dumps(records, indent=2))


class LocalBackend:
    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg

    def launch(self, config_path):
        launch = OmegaConf.load(config_path)
        self.cfg.generator.trajectory_retention = launch.skyrl.generator.trajectory_retention
        assert self.cfg.generator.trajectory_retention.model_checkpoint_step == 3
        Path(launch.artifacts.resolved_config_uri).write_text(config_path.read_text())
        OmegaConf.save(self.cfg, self.root / "consumed-eval.yaml")
        ray.init(num_cpus=3, include_dashboard=False, log_to_driver=True)
        try:
            timing_events = []
            evaluation = importlib.import_module("skyrl_train.evaluate")
            original_record = evaluation.record_event
            evaluation.record_event = lambda name, fields: timing_events.append({"name": name, **fields})
            try:
                metrics = CPUEntry(self.cfg).run()
            finally:
                evaluation.record_event = original_record
            tasks = [event for event in timing_events if event["name"] == "evaluation_task"]
            assert len(tasks) == 2
            assert all(
                event["succeeded"]
                and event["submitted_unix_ms"] <= event["started_unix_ms"] <= event["finished_unix_ms"]
                for event in tasks
            )
            assert {event["benchmark"] for event in tasks} == {"installed", "native"}
            (self.root / "timing-events.json").write_text(json.dumps(timing_events, indent=2))
        finally:
            ray.shutdown()
        (self.root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        assert metrics["eval/all/num_attempted"] == 2
        assert metrics["eval/all/num_scored"] == 2
        assert metrics["eval/all/avg_verifier_score"] == 1
        verify_retention(self.root, self.cfg)
        return LaunchOutcome("cpu-eval", "succeeded", 0)

    def validate(self, config_path):
        raise AssertionError("Expected wait, not prepare")

    def export_terminal_policy(self, config_path):
        raise AssertionError("Eval-only must not export a model")


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    cfg = configuration(root)
    launch = raw_launch_config()
    launch["skyrl"].setdefault("generator", {})["trajectory_retention"] = OmegaConf.to_container(
        cfg.generator.trajectory_retention
    )
    del launch["skyrl"]["generator"]["trajectory_retention"]["output_path"]
    launch["run"]["export_hf"] = False
    launch["runtime"]["entrypoint"] = "skyrl_train.entrypoints.main_generate"
    launch["artifacts"] = {
        key: str(root / name)
        for key, name in {
            "checkpoint_root": "checkpoints",
            "export_root": "exports",
            "attempts_root": "attempts",
            "resolved_config_uri": "resolved.yaml",
            "terminal_manifest_uri": "terminal.json",
        }.items()
    }
    path = root / "launch.yaml"
    path.write_text(yaml.safe_dump(launch))
    result = execute_launch(path, backend=LocalBackend(root, cfg))
    assert result.state == "succeeded", result
    assert json.loads((root / "terminal.json").read_text())["result"]["state"] == "succeeded"
    print("CPU_EVAL_COMPLETE", result.state)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
