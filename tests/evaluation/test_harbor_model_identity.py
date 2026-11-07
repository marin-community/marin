# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight policy identity is independent of model and serving names."""

import json
import tempfile
import textwrap
from pathlib import Path

import pytest
from marin.evaluation.harbor.driver_config import preflight_harbor_configs

from tests.evaluation.harbor_runtime import external_python, write_runtime_overlay

pytestmark = pytest.mark.timeout(180)


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
        from harbor_config import JobConfig
        from marin.evaluation.harbor.trial_driver import RuntimeOverlay, effective_job_config

        policy, *overlays = map(Path, sys.argv[1:])
        stable_policy = JobConfig.model_validate_json(policy.read_text())
        configs = [
            effective_job_config(stable_policy, RuntimeOverlay.model_validate_json(overlay.read_text()))
            for overlay in overlays
        ]
        trials = [
            TrialConfig(task=TaskConfig(path=Path('/tasks/problem')), agent=config.agents[0]) for config in configs
        ]
        # Exercise the factory used by actual trials, including explicit import paths.
        infos = []
        agents = [
            ('terminus-2', None), ('pi', None), ('opencode', None), ('mini-swe-agent', None),
            ('codex', None), (None, 'harbor.agents.terminus_kira.terminus_kira:TerminusKira'),
        ]
        for index, (name, import_path) in enumerate(agents):
            kwargs = {**configs[1].agents[0].kwargs, 'thinking_format': 'chat-template'}
            if name == 'mini-swe-agent':
                kwargs['version'] = '2.1.0'
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
    assert result["infos"] == [{"name": identity, "provider": "hosted_vllm"}] * 6
    assert result["alias"] == "hosted_vllm/resumed-endpoint"
    assert result["cohort_model"] == identity


def test_preflight_policy_identity_ignores_model_and_serving_names():
    with tempfile.TemporaryDirectory(prefix=".harbor-identity-", dir=Path(__file__).parents[2]) as directory:
        tmp_path = Path(directory)
        task = tmp_path / "dataset/task"
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "task.toml").write_text(
            '[task]\nname = "test/model-identity"\n[environment]\n'
            f'docker_image = "example.test/task@sha256:{"a" * 64}"\n'
            'allow_internet = false\nworkdir = "/workspace"\n'
        )
        (task / "instruction.md").write_text("Complete the task.\n")
        (task / "tests/test.sh").write_text("exit 0\n")
        paths = []
        for model_name, model_alias in [
            ("hosted_vllm/org/original", "hosted_vllm/old"),
            ("hosted_vllm/org/other", "hosted_vllm/new"),
        ]:
            path = tmp_path / f"policy-{len(paths)}.json"
            path.write_text(
                json.dumps(
                    {
                        "environment": {"type": "local"},
                        "agents": [{"name": "terminus-2", "model_name": model_name, "model_alias": model_alias}],
                        "datasets": [{"path": "dataset"}],
                    }
                )
            )
            paths.append(path)
        configs = preflight_harbor_configs([(path, {}) for path in paths], judge_model=None)
        assert configs[0].digest == configs[1].digest
