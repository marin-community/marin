# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for model provenance, resume identity, and request routing."""

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
from marin.evaluation.harbor.driver_config import preflight_harbor_configs

from tests.evaluation.harbor_runtime import external_python, run_single_turn_aime_agent, write_runtime_overlay

pytestmark = pytest.mark.timeout(180)


@pytest.mark.parametrize(
    "model_alias, expected_model", [(None, "iceball-micro"), ("hosted_vllm/resumed-endpoint", "resumed-endpoint")]
)
def test_single_turn_aime_routes_requests_using_serving_alias(tmp_path, model_alias, expected_model):
    result = run_single_turn_aime_agent(tmp_path, content="The answer is 42", model_alias=model_alias)
    assert result["models"] == [expected_model]
    assert result["answer"] == "42\n"


def test_single_turn_aime_agent_grades_length_finished_response(tmp_path):
    content = "A long incomplete derivation mentions 12. The final result is \\boxed{137}."
    result = run_single_turn_aime_agent(tmp_path, content=content, finish_reason="length")

    assert result == {"answer": "137\n", "response": content, "attempts": 1, "models": ["iceball-micro"]}


@pytest.mark.parametrize("reasoning_content", [None, "An incomplete derivation suggests \\boxed{137}."])
def test_single_turn_aime_agent_writes_invalid_answer_when_length_budget_exhausts_before_final_text(
    tmp_path, reasoning_content
):
    result = run_single_turn_aime_agent(
        tmp_path, content=None, reasoning_content=reasoning_content, finish_reason="length"
    )

    assert result == {"answer": "-1\n", "response": "", "attempts": 1, "models": ["iceball-micro"]}


@pytest.mark.parametrize("finish_reason", [None, "stop"])
def test_single_turn_aime_agent_rejects_null_final_text_without_length_truncation(tmp_path, finish_reason):
    with pytest.raises(subprocess.CalledProcessError):
        run_single_turn_aime_agent(tmp_path, content=None, finish_reason=finish_reason)

    assert not (tmp_path / "answer.txt").exists()


def test_single_turn_aime_agent_retries_transient_proxy_failure(tmp_path):
    result = run_single_turn_aime_agent(
        tmp_path,
        content="Final answer: 42",
        transient_failures=1,
    )

    assert result == {
        "answer": "42\n",
        "response": "Final answer: 42",
        "attempts": 2,
        "models": ["iceball-micro", "iceball-micro"],
    }


@pytest.mark.parametrize("identity", ["org/original-model", "gs://models/export/step-10", "s3://models/export/step-10"])
def test_runtime_alias_change_preserves_identity_and_rejects_different_model(tmp_path, identity):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "agents": [{"name": "terminus-2", "model_name": "stale/identity", "model_alias": "stale/alias"}],
                "datasets": [{"name": "terminal-bench", "version": "2.0"}],
            }
        )
    )
    overlays = []
    for index, (model, alias) in enumerate(
        [(identity, "original-endpoint"), (identity, "resumed-endpoint"), (identity + "-other", "resumed-endpoint")]
    ):
        path = tmp_path / f"overlay-{index}.json"
        write_runtime_overlay(path, tmp_path, model_identity=model, served_model=alias)
        overlays.append(str(path))
    script = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path
        from harbor.agents.factory import AgentFactory
        from harbor.models.trial.config import TrialConfig, TaskConfig
        from harbor.models.job.reconciliation import trial_identity_key, classify_trial_plan_mismatches
        from marin.evaluation.harbor.trial_driver import effective_job_config

        policy, *overlays = map(Path, sys.argv[1:])
        configs = [effective_job_config(policy, overlay) for overlay in overlays]
        trials = [
            TrialConfig(task=TaskConfig(path=Path('/tasks/problem')), agent=config.agents[0]) for config in configs
        ]
        # Exercise the factory used by actual trials, including imported and registry agents.
        infos = []
        agents = [
            ('terminus-2', None), ('pi', None), ('opencode', None), ('mini-swe-agent', None),
            (None, 'harbor.agents.installed.sotopia:SotopiaAgent'), ('acp:pi-acp@0.0.33', None),
            (None, 'marin.evaluation.harbor.single_turn_aime_agent:SingleTurnAimeAgent'),
        ]
        for index, (name, import_path) in enumerate(agents):
            kwargs = {**configs[1].agents[0].kwargs, 'thinking_format': 'chat-template'}
            if name == 'mini-swe-agent':
                kwargs['version'] = '2.1.0'
            if import_path and 'SingleTurnAimeAgent' in import_path:
                kwargs.update(answer_path='/tmp/answer.txt', max_tokens=256)
            configured = configs[1].agents[0].model_copy(update={
                'name': name,
                'import_path': import_path,
                'kwargs': kwargs,
            })
            agent = AgentFactory.create_agent_from_config(configured, logs_dir=policy.parent / str(index))
            infos.append(agent.to_agent_info().model_info.model_dump())
        print(json.dumps({
            'same_resume_identity': trial_identity_key(trials[0]) == trial_identity_key(trials[1]),
            'different_model_fields': classify_trial_plan_mismatches([trials[0]], [trials[2]])[0].differing_fields,
            'infos': infos,
            'alias': configs[1].agents[0].model_alias,
            'cohort_model': configs[1].agents[0].model_name_without_provider,
        }))
    """
    )
    result = json.loads(external_python("-c", script, str(policy), *overlays).stdout)
    assert result["same_resume_identity"]
    assert result["different_model_fields"] == ["agent.model_name"]
    assert result["infos"] == [{"name": identity, "provider": "hosted_vllm"}] * 7
    assert result["alias"] == "hosted_vllm/resumed-endpoint"
    assert result["cohort_model"] == identity


def test_preflight_policy_identity_ignores_model_and_serving_names():
    with tempfile.TemporaryDirectory(prefix=".harbor-identity-", dir=Path(__file__).parents[2]) as directory:
        tmp_path = Path(directory)
        task = tmp_path / "dataset" / "task"
        task.mkdir(parents=True)
        (task / "task.toml").write_text('[task]\nname = "test/model-identity"\n')
        (task / "instruction.md").write_text("Complete the task.\n")
        (task / "environment").mkdir()
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("exit 0\n")
        paths = []
        for model_name, model_alias in [
            ("hosted_vllm/org/original", "hosted_vllm/old"),
            ("hosted_vllm/org/other", "hosted_vllm/new"),
        ]:
            path = tmp_path / f"policy-{len(paths)}.json"
            path.write_text(
                json.dumps(
                    {
                        "environment": {"type": "daytona"},
                        "agents": [{"name": "terminus-2", "model_name": model_name, "model_alias": model_alias}],
                        "datasets": [{"path": "dataset"}],
                    }
                )
            )
            paths.append(path)
        configs = preflight_harbor_configs([(path, {}) for path in paths], judge_model=None)
        assert configs[0].digest == configs[1].digest
        assert configs[0].benchmark.n_attempted == configs[1].benchmark.n_attempted == 1
