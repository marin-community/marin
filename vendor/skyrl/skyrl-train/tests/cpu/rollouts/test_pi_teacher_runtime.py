"""Launch-preflight and student-only evaluation wiring for Pi teacher SFT."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import json

import httpx
import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from skyrl_train.config.pi_training import validate_pi_training_config
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.pi_teacher_runtime import prepare_pi_teacher, start_pi_managed_teacher, student_eval_harbor_config
from skyrl_train.rollouts.harbor_settings import EnvironmentType
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.config.utils import get_default_config
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


@pytest.fixture
def teacher_config(tmp_path, monkeypatch):
    student = build_tiny_policy(tmp_path / "student")
    teacher = build_tiny_policy(tmp_path / "teacher")
    monkeypatch.setenv("PI_TEST_LAUNCH_TEACHER_KEY", "test-only-key")

    class Server(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            if self.headers.get("Authorization") != "Bearer test-only-key":
                self.send_error(401)
                return
            payload = json.dumps({"data": [{"id": "teacher", "max_model_len": 4096}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = get_default_config()
    cfg.trainer.policy.model.path = str(student)
    cfg.trainer.policy.model.tokenizer_path = str(student)
    cfg.trainer.evaluation_runner = "harbor"
    cfg.trainer.eval_interval = 1
    cfg.generator.n_samples_per_prompt = 1
    OmegaConf.update(cfg, "generator.eval_sampling_params.logprobs", 0, force_add=True)
    cfg.trainer.algorithm.policy_loss_type = "sft"
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    OmegaConf.update(cfg, "data.train_data", ["fixture-task"], force_add=True)
    OmegaConf.update(cfg, "data.terminal_bench_data", [], force_add=True)
    OmegaConf.update(cfg, "scenario", {"preset": "teacher_demo_pi_verified_base"}, force_add=True)
    OmegaConf.update(cfg, "terminal_bench_config.harbor.verifier_disable", False, force_add=True)
    OmegaConf.update(cfg, "terminal_bench_config.harbor.environment_type", "docker", force_add=True)
    OmegaConf.update(
        cfg,
        "terminal_bench_config.installed_agent",
        {
            "agent": {
                "name": "pi",
                "model_name": "hosted_vllm/teacher",
                "pi": {"mode": "teacher_trace"},
                "kwargs": {"thinking_format": "qwen-chat-template"},
            },
            "teacher": {
                "source": "external_vllm",
                "model": "teacher",
                "model_path": str(teacher),
                "revision": "local-pinned",
                "student_model": str(student),
                "max_context_tokens": 4096,
                "endpoint": f"http://127.0.0.1:{server.server_port}/v1",
                "auth": "env:PI_TEST_LAUNCH_TEACHER_KEY",
            },
            "output_path": str(tmp_path / "attempts"),
            "setup_timeout": 5,
            "run_timeout": 5,
            "cleanup_timeout": 5,
            "service_startup_timeout": 5,
            "service_shutdown_timeout": 5,
        },
        force_add=True,
    )
    try:
        yield cfg, student, teacher
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_external_teacher_launch_preflight_and_student_only_eval(teacher_config):
    cfg, student, _teacher = teacher_config
    validate_pi_training_config(cfg)
    prepared = prepare_pi_teacher(cfg)
    assert prepared is not None
    assert prepared.binding.model_path != cfg.trainer.policy.model.path
    assert prepared.tokenizer.name_or_path != str(student)
    eval_config = student_eval_harbor_config(cfg.terminal_bench_config, str(student))
    assert eval_config.installed_agent.agent.pi.mode == "literal"
    assert "teacher" not in eval_config.installed_agent
    student_engine = CPUInferenceEngine(str(student), seed=3)
    tokenizer = AutoTokenizer.from_pretrained(student)
    launcher = BasePPOExp.__new__(BasePPOExp)
    launcher.sessions = {}
    launcher.prepared_pi_teacher = prepared
    launcher.pi_teacher_engines = None
    inference = InferenceEngineClient([student_engine], tokenizer, cfg)
    train_spec = launcher.get_task_runner(cfg, inference)._spec
    eval_spec = launcher.get_task_runner(cfg, inference, harbor_config=eval_config, evaluation=True)._spec
    assert not cfg.data.terminal_bench_data
    assert train_spec.teacher_tokenizer is prepared.tokenizer
    assert train_spec.harbor_config.installed_agent.agent.pi.mode == "teacher_trace"
    assert eval_spec.teacher_tokenizer is None
    terminal_launcher = TerminalBenchExp.__new__(TerminalBenchExp)
    terminal_launcher.sessions = {}
    terminal_launcher.prepared_pi_teacher = prepared
    terminal_launcher.pi_teacher_engines = None
    terminal_spec = terminal_launcher.get_trajectory_runner(cfg, tokenizer, inference)._spec
    assert terminal_spec.teacher_tokenizer is prepared.tokenizer
    assert terminal_spec.harbor_config.installed_agent.agent.pi.mode == "teacher_trace"
    train_worker = train_spec.build(tokenizer, WorkerShard(0, 1))
    eval_worker = eval_spec.build(tokenizer, WorkerShard(0, 1))
    assert train_worker.harbor.environment.type is EnvironmentType.DOCKER
    assert train_worker.installed_agents is not None and not train_worker.installed_agents.student_only
    assert eval_worker.installed_agents is not None and eval_worker.installed_agents.student_only


def test_managed_teacher_provisions_independent_pinned_engine_role(teacher_config):
    cfg, student, teacher = teacher_config
    cfg.terminal_bench_config.installed_agent.teacher.source = "skyrl_managed"
    del cfg.terminal_bench_config.installed_agent.teacher.endpoint
    del cfg.terminal_bench_config.installed_agent.teacher.auth
    OmegaConf.update(
        cfg,
        "terminal_bench_config.installed_agent.teacher.resources",
        {"backend": "vllm", "num_engines": 1, "tensor_parallel_size": 1},
        force_add=True,
    )
    independent_engine = CPUInferenceEngine(str(teacher), seed=8)
    cfg.terminal_bench_config.installed_agent.teacher.max_context_tokens = independent_engine.max_model_len
    validate_pi_training_config(cfg)
    prepared = prepare_pi_teacher(cfg)
    assert prepared is not None
    received = []

    def allocate(**kwargs):
        received.append(kwargs)
        return [independent_engine]

    engines = start_pi_managed_teacher(cfg, prepared, engine_factory=allocate)
    assert engines == [independent_engine]
    assert received[0]["pretrain"] == str(teacher)
    assert received[0]["shared_pg"] is None
    assert received[0]["engine_init_kwargs"]["revision"] == "local-pinned"
    assert received[0]["engine_init_kwargs"]["served_model_name"] == "teacher"
    assert received[0]["engine_init_kwargs"]["max_model_len"] == independent_engine.max_model_len
    student_engine = CPUInferenceEngine(str(student), seed=3)
    launcher = BasePPOExp.__new__(BasePPOExp)
    launcher.sessions = {}
    launcher.prepared_pi_teacher = prepared
    launcher.pi_teacher_engines = engines
    tokenizer = AutoTokenizer.from_pretrained(student)
    train_spec = launcher.get_task_runner(cfg, InferenceEngineClient([student_engine], tokenizer, cfg))._spec
    assert train_spec.teacher_engines == engines
    worker = train_spec.build(tokenizer, WorkerShard(0, 1))
    assert worker.installed_agents is not None and not worker.installed_agents.student_only


def test_external_teacher_preflight_rejects_wrong_endpoint_model(teacher_config):
    cfg, _, _ = teacher_config
    cfg.terminal_bench_config.installed_agent.teacher.model = "mismatched-alias"
    cfg.terminal_bench_config.installed_agent.agent.model_name = "hosted_vllm/mismatched-alias"
    with pytest.raises(ValueError, match="alias or context limit"):
        prepare_pi_teacher(cfg)


def test_external_teacher_preflight_rejects_bad_credentials(teacher_config, monkeypatch):
    cfg, _, _ = teacher_config
    monkeypatch.setenv("PI_TEST_LAUNCH_TEACHER_KEY", "incorrect")
    with pytest.raises(httpx.HTTPStatusError, match="401"):
        prepare_pi_teacher(cfg)


def test_teacher_mode_without_reviewed_scenario_cannot_launch(teacher_config):
    cfg, _, _ = teacher_config
    del cfg.scenario
    with pytest.raises(ValueError, match="reviewed scenario.preset"):
        validate_pi_training_config(cfg)


def test_teacher_launch_allows_no_scheduled_student_evaluation(teacher_config):
    cfg, _, _ = teacher_config
    cfg.trainer.eval_interval = -1
    cfg.data.val_data = []
    validate_pi_training_config(cfg)
