import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

import harbor.models.job.lock as lock_models
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.bridge import BridgeConfig, BridgeKind
from harbor.models.job.lock import (
    JobLock,
    build_job_lock,
    sanitize_cli_invocation,
)
from harbor.skills import compute_skill_digest
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    UserAgentConfig,
    VerifierConfig,
)
from harbor.tasks.packager import Packager
from harbor.tasks.client import TaskDownloadResult


TASK_TOML = """\
[task]
name = "test-org/test-task"
description = "A test task"

[agent]
timeout_sec = 300
"""


def _make_task_dir(tmp_path: Path, name: str = "task") -> Path:
    task_dir = tmp_path / name
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(TASK_TOML)
    (task_dir / "instruction.md").write_text("Do the thing.")
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


def _make_skill(parent: Path, name: str, content: str = "# skill\n") -> Path:
    skill_dir = parent / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


def _sha(char: str) -> str:
    return f"sha256:{char * 64}"


def _trial(task: TaskConfig, trial_name: str = "trial-1", **kwargs) -> TrialConfig:
    return TrialConfig(task=task, trial_name=trial_name, **kwargs)


def test_local_task_uses_packager_content_hash(tmp_path: Path) -> None:
    task_dir = _make_task_dir(tmp_path)
    task = TaskConfig(path=task_dir)
    expected_hash, _ = Packager.compute_content_hash(task_dir)

    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task]),
        trial_configs=[_trial(task)],
        invocation=["harbor", "run"],
    )

    assert lock.trials[0].task.type == "local"
    assert lock.trials[0].task.digest == f"sha256:{expected_hash}"
    assert lock.trials[0].task.source is None
    trial_task_data = lock.trials[0].task.model_dump(mode="json")
    assert trial_task_data["type"] == "local"
    assert "kind" not in trial_task_data
    assert "local_path" not in trial_task_data
    assert "config" not in trial_task_data
    assert "tasks" not in lock.model_dump(mode="json")


def test_bridge_input_files_are_hashed(tmp_path: Path) -> None:
    task_dir = _make_task_dir(tmp_path)
    task = TaskConfig(path=task_dir)
    user_persona = tmp_path / "persona.md"
    user_prompt = tmp_path / "user.j2"
    bridge_prompt = tmp_path / "bridge.md"
    acpx_config = tmp_path / "acpx.json"
    user_persona.write_text("Grumpy")
    user_prompt.write_text("{{ persona }} {{ bridge_instructions }} {{ instruction }}")
    bridge_prompt.write_text("Use ACPX")
    acpx_config.write_text('{"timeout": 1800}')
    user_agent = UserAgentConfig(
        name="claude-code",
        user_persona_path=user_persona,
        user_prompt_template_path=user_prompt,
        bridge=BridgeConfig(
            kind=BridgeKind.ACP,
            prompt_path=bridge_prompt,
            kwargs={"acpx_config_path": str(acpx_config)},
        ),
    )

    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], user_agent=user_agent),
        trial_configs=[_trial(task, user_agent=user_agent)],
    ).trials[0]

    assert lock.user_persona is not None
    assert lock.user_prompt_template is not None
    assert lock.bridge_prompt is not None
    assert set(lock.bridge_inputs) == {"acpx_config_path"}
    assert lock.user_persona.digest == lock_models._file_sha256_digest(user_persona)
    assert lock.user_prompt_template.digest == lock_models._file_sha256_digest(
        user_prompt
    )
    assert lock.bridge_prompt.digest == lock_models._file_sha256_digest(bridge_prompt)
    assert lock.bridge_inputs["acpx_config_path"].digest == (
        lock_models._file_sha256_digest(acpx_config)
    )


def test_bridge_lock_equality_uses_file_contents_not_host_paths(
    tmp_path: Path,
) -> None:
    task_dir = _make_task_dir(tmp_path)
    task = TaskConfig(path=task_dir)

    def user_agent(parent: Path, content: str) -> UserAgentConfig:
        parent.mkdir()
        user_prompt = parent / "user.j2"
        bridge_prompt = parent / "bridge.md"
        acpx_config = parent / "acpx.json"
        user_prompt.write_text("{{ bridge_instructions }} {{ instruction }}")
        bridge_prompt.write_text("Use ACPX")
        acpx_config.write_text(content)
        return UserAgentConfig(
            name="claude-code",
            user_prompt_template_path=user_prompt,
            bridge=BridgeConfig(
                kind=BridgeKind.ACP,
                prompt_path=bridge_prompt,
                kwargs={"acpx_config_path": str(acpx_config)},
            ),
        )

    first_user = user_agent(tmp_path / "first", '{"timeout": 1800}')
    moved_user = user_agent(tmp_path / "moved", '{"timeout": 1800}')
    changed_user = user_agent(tmp_path / "changed", '{"timeout": 900}')

    first = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], user_agent=first_user),
        trial_configs=[_trial(task, user_agent=first_user)],
    ).trials[0]
    moved = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], user_agent=moved_user),
        trial_configs=[_trial(task, user_agent=moved_user)],
    ).trials[0]
    changed = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], user_agent=changed_user),
        trial_configs=[_trial(task, user_agent=changed_user)],
    ).trials[0]

    assert first == moved
    assert first != changed


def test_user_persona_lock_equality_uses_file_contents_not_host_paths(
    tmp_path: Path,
) -> None:
    task_dir = _make_task_dir(tmp_path)
    task = TaskConfig(path=task_dir)

    def user_agent(parent: Path, persona: str) -> UserAgentConfig:
        parent.mkdir()
        persona_path = parent / "persona.md"
        persona_path.write_text(persona)
        return UserAgentConfig(
            name="claude-code",
            user_persona_path=persona_path,
            bridge=BridgeConfig(kind=BridgeKind.ACP),
        )

    def lock_for(agent: UserAgentConfig) -> lock_models.TrialLock:
        return build_job_lock(
            config=JobConfig(job_name="job", tasks=[task], user_agent=agent),
            trial_configs=[_trial(task, user_agent=agent)],
        ).trials[0]

    first = lock_for(user_agent(tmp_path / "first", "Grumpy"))
    moved = lock_for(user_agent(tmp_path / "moved", "Grumpy"))
    changed = lock_for(user_agent(tmp_path / "changed", "Cheerful"))

    assert first == moved
    assert first != changed


def test_task_lock_equality_uses_digest_only() -> None:
    digest = _sha("a")
    assert lock_models.TaskLock(
        name="test-org/first",
        type="local",
        digest=digest,
        path=Path("first"),
    ) == lock_models.TaskLock(
        name="test-org/second",
        type="package",
        digest=digest,
        source="test-org/dataset",
    )


def test_package_task_uses_resolved_ref_digest() -> None:
    task_digest = _sha("a")
    task = TaskConfig(name="test-org/test-task", ref=task_digest, source="test-org/ds")

    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task]),
        trial_configs=[_trial(task)],
        invocation=["harbor", "run"],
    )

    assert lock.trials[0].task.type == "package"
    assert lock.trials[0].task.digest == task_digest


def test_extra_docker_compose_lock_changes_with_file_content(tmp_path: Path) -> None:
    extra = tmp_path / "compose.extra.yaml"
    extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
    task = TaskConfig(name="test-org/test-task", ref=_sha("b"))
    environment = EnvironmentConfig(extra_docker_compose=[extra])
    trial = _trial(task, environment=environment)

    first_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=environment),
        trial_configs=[trial],
        invocation=["harbor", "run"],
    )
    extra.write_text("services:\n  sidecar:\n    image: redis:8\n")
    second_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=environment),
        trial_configs=[trial],
        invocation=["harbor", "run"],
    )

    first_extra = first_lock.trials[0].extra_docker_compose
    second_extra = second_lock.trials[0].extra_docker_compose
    assert first_extra is not None
    assert second_extra is not None
    assert first_extra[0].path == extra
    assert first_extra[0].digest.startswith("sha256:")
    assert first_extra[0].digest != second_extra[0].digest
    assert first_lock != second_lock


def test_extra_docker_compose_lock_equality_uses_digest_only() -> None:
    digest = _sha("b")
    assert lock_models.ExtraDockerComposeLock(
        path=Path("compose.extra.yaml"), digest=digest
    ) == lock_models.ExtraDockerComposeLock(path=Path("other.yaml"), digest=digest)


def test_job_lock_equality_ignores_extra_docker_compose_path(tmp_path: Path) -> None:
    extra = tmp_path / "compose.extra.yaml"
    extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
    task = TaskConfig(name="test-org/test-task", ref=_sha("b"))
    environment = EnvironmentConfig(extra_docker_compose=[extra])
    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=environment),
        trial_configs=[_trial(task, environment=environment)],
        invocation=["harbor", "run"],
    )

    extra_lock = lock.trials[0].extra_docker_compose
    assert extra_lock is not None
    other_trial = lock.trials[0].model_copy(
        update={
            "extra_docker_compose": [
                extra_lock[0].model_copy(update={"path": Path("other.yaml")})
            ]
        }
    )
    other_lock = lock.model_copy(update={"trials": [other_trial]})

    assert lock == other_lock


def test_job_lock_equality_ignores_extra_docker_compose_input_path(
    tmp_path: Path,
) -> None:
    first_extra = tmp_path / "first.compose.yaml"
    second_extra = tmp_path / "second.compose.yaml"
    compose_content = "services:\n  sidecar:\n    image: redis:7\n"
    first_extra.write_text(compose_content)
    second_extra.write_text(compose_content)
    task = TaskConfig(name="test-org/test-task", ref=_sha("b"))
    first_environment = EnvironmentConfig(extra_docker_compose=[first_extra])
    second_environment = EnvironmentConfig(extra_docker_compose=[second_extra])

    first_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=first_environment),
        trial_configs=[_trial(task, environment=first_environment)],
        invocation=["harbor", "run"],
    )
    second_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=second_environment),
        trial_configs=[_trial(task, environment=second_environment)],
        invocation=["harbor", "run"],
    )

    assert first_lock == second_lock


def test_job_lock_equality_ignores_trial_order() -> None:
    first_task = TaskConfig(name="test-org/first", ref=_sha("1"))
    second_task = TaskConfig(name="test-org/second", ref=_sha("2"))
    config = JobConfig(job_name="job", tasks=[first_task, second_task])

    first_lock = build_job_lock(
        config=config,
        trial_configs=[
            _trial(first_task, trial_name="first-trial"),
            _trial(second_task, trial_name="second-trial"),
        ],
        invocation=["harbor", "run"],
    )
    second_lock = first_lock.model_copy(
        update={"trials": list(reversed(first_lock.trials))}
    )

    assert first_lock == second_lock


def test_job_lock_ignores_model_alias_but_keeps_model_name_in_identity() -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("1"))

    def lock_for(model_name: str, model_alias: str):
        agent = AgentConfig(
            name="terminus-2",
            model_name=model_name,
            model_alias=model_alias,
        )
        return build_job_lock(
            config=JobConfig(job_name="job", tasks=[task], agents=[agent]),
            trial_configs=[_trial(task, agent=agent)],
            invocation=["harbor", "run"],
        )

    original = lock_for("hosted_vllm/org/model", "hosted_vllm/original-endpoint")
    moved = lock_for("hosted_vllm/org/model", "hosted_vllm/resumed-endpoint")
    different_model = lock_for(
        "hosted_vllm/org/other-model", "hosted_vllm/resumed-endpoint"
    )

    assert original == moved
    assert original != different_model


def test_job_lock_equality_ignores_non_replay_identity_fields() -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("1"))
    lock = build_job_lock(
        config=JobConfig(job_name="original-job", tasks=[task]),
        trial_configs=[_trial(task, trial_name="original-trial")],
        invocation=["harbor", "run"],
    )
    other_lock = lock.model_copy(
        deep=True,
        update={
            "created_at": datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            "harbor": lock_models.HarborLockInfo(
                version="9.9.9",
                git_commit_hash="different",
                is_editable=False,
            ),
            "invocation": ["harbor", "job", "resume"],
        },
    )

    assert lock == other_lock
    data = lock.model_dump(mode="json")
    assert "job_id" not in data
    assert "job_name" not in data
    assert "trial_name" not in data["trials"][0]

    legacy_data = lock.model_dump(mode="json")
    legacy_data["job_id"] = "00000000-0000-0000-0000-000000000000"
    legacy_data["job_name"] = "legacy-job"
    legacy_data["trials"][0]["trial_name"] = "legacy-trial"
    legacy_lock = JobLock.model_validate(legacy_data)

    assert legacy_lock == lock
    rewritten_data = legacy_lock.model_dump(mode="json")
    assert "job_id" not in rewritten_data
    assert "job_name" not in rewritten_data
    assert "trial_name" not in rewritten_data["trials"][0]


def test_job_lock_equality_uses_serialized_sensitive_env_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-secret")
    task = TaskConfig(name="test-org/test-task", ref=_sha("1"))
    agent = AgentConfig(
        name="claude-code",
        env={"OPENAI_API_KEY": "sk-real-secret"},
    )
    environment = EnvironmentConfig(env={"OPENAI_API_KEY": "sk-real-secret"})
    verifier = VerifierConfig(env={"OPENAI_API_KEY": "sk-real-secret"})
    lock = build_job_lock(
        config=JobConfig(
            job_name="job",
            tasks=[task],
            agents=[agent],
            environment=environment,
            verifier=verifier,
        ),
        trial_configs=[
            _trial(
                task,
                agent=agent,
                environment=environment,
                verifier=verifier,
            )
        ],
        invocation=["harbor", "run"],
    )
    persisted_lock = JobLock.model_validate_json(
        lock.model_dump_json(exclude_none=True)
    )

    persisted_trial = persisted_lock.trials[0]
    assert persisted_trial.agent.env == {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
    assert persisted_trial.environment.env == {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
    assert persisted_trial.verifier.env == {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
    assert lock == persisted_lock


def test_package_task_uses_download_result_content_hash() -> None:
    content_hash = "b" * 64
    task = TaskConfig(name="test-org/test-task", ref="latest", source="test-org/ds")

    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task]),
        trial_configs=[_trial(task)],
        invocation=["harbor", "run"],
        task_download_results={
            task.get_task_id(): TaskDownloadResult(
                path=Path("/tmp/cache/test-task"),
                download_time_sec=0.0,
                cached=False,
                content_hash=content_hash,
            )
        },
    )

    assert lock.trials[0].task.type == "package"
    assert lock.trials[0].task.digest == f"sha256:{content_hash}"
    assert task.ref == "latest"


def test_git_task_uses_download_result_resolved_commit(monkeypatch) -> None:
    resolved_commit = "f" * 40
    content_hash = "a" * 64
    task = TaskConfig(
        path=Path("tasks/hello-world"),
        git_url="https://example.com/repo.git",
        source="test-dataset",
    )

    monkeypatch.setattr(
        lock_models.Packager,
        "compute_content_hash",
        lambda _path: (content_hash, []),
    )

    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task]),
        trial_configs=[_trial(task)],
        invocation=["harbor", "run"],
        task_download_results={
            task.get_task_id(): TaskDownloadResult(
                path=Path("/tmp/cache/hello-world"),
                download_time_sec=0.0,
                cached=False,
                resolved_git_commit_id=resolved_commit,
            )
        },
    )

    assert lock.trials[0].task.type == "git"
    assert lock.trials[0].task.digest == f"sha256:{content_hash}"
    assert lock.trials[0].task.git_commit_id == resolved_commit
    assert task.git_commit_id is None


def test_dataset_config_is_not_written_but_trial_task_source_remains(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    task = TaskConfig(name="test-org/test-task", ref=_sha("c"), source=dataset_dir.name)

    lock = build_job_lock(
        config=JobConfig(job_name="job", datasets=[DatasetConfig(path=dataset_dir)]),
        trial_configs=[_trial(task)],
        invocation=["harbor", "run"],
    )

    data = lock.model_dump(mode="json")
    assert "datasets" not in data
    assert lock.trials[0].task.source == dataset_dir.name
    assert [trial.task.digest for trial in lock.trials] == [_sha("c")]


def test_seed_values_are_not_indexed_separately() -> None:
    digest = _sha("d")
    task = TaskConfig(name="test-org/test-task", ref=digest)
    agent = AgentConfig(name="claude-code", kwargs={"seed": 123})

    lock = build_job_lock(
        config=JobConfig(
            job_name="job",
            tasks=[task],
            agents=[agent],
        ),
        trial_configs=[_trial(task, agent=agent)],
        invocation=["harbor", "run"],
    )

    data = lock.model_dump(mode="json")
    assert "observed_trials" not in data
    assert "seed_values" not in data
    assert data["trials"][0]["agent"]["kwargs"]["seed"] == 123


def test_lock_records_extra_instruction_digests(tmp_path: Path, monkeypatch) -> None:
    task_dir = _make_task_dir(tmp_path)
    task = TaskConfig(path=task_dir)
    extra_hint = tmp_path / "extra-no-multimodal-hint.md"
    extra_hint.write_text("extra hint\n")
    monkeypatch.chdir(tmp_path)
    extra_instruction_paths = [Path("extra-no-multimodal-hint.md")]
    trial = _trial(
        task,
        extra_instruction_paths=extra_instruction_paths,
    )

    lock = build_job_lock(
        config=JobConfig(
            job_name="job",
            tasks=[task],
            extra_instruction_paths=extra_instruction_paths,
        ),
        trial_configs=[trial],
        invocation=["harbor", "run"],
    )

    trial_lock = lock.model_dump(mode="json")["trials"][0]
    assert trial_lock["extra_instructions"] == [
        {
            "path": "extra-no-multimodal-hint.md",
            "digest": f"sha256:{hashlib.sha256(extra_hint.read_bytes()).hexdigest()}",
        }
    ]


def test_extra_instruction_lock_equality_uses_digest_only() -> None:
    digest = _sha("d")
    assert lock_models.ExtraInstructionLock(
        path=Path("extra-instruction.md"), digest=digest
    ) == lock_models.ExtraInstructionLock(path=Path("other.md"), digest=digest)


def test_job_lock_equality_ignores_extra_instruction_path(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = _make_task_dir(tmp_path)
    task = TaskConfig(path=task_dir)
    extra_hint = tmp_path / "extra-no-multimodal-hint.md"
    extra_hint.write_text("extra hint\n")
    monkeypatch.chdir(tmp_path)
    trial = _trial(
        task,
        extra_instruction_paths=[Path("extra-no-multimodal-hint.md")],
    )
    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task]),
        trial_configs=[trial],
        invocation=["harbor", "run"],
    )

    instruction_lock = lock.trials[0].extra_instructions
    assert instruction_lock is not None
    other_trial = lock.trials[0].model_copy(
        update={
            "extra_instructions": [
                instruction_lock[0].model_copy(update={"path": Path("other.md")})
            ]
        }
    )
    other_lock = lock.model_copy(update={"trials": [other_trial]})

    assert lock == other_lock


def test_lock_errors_on_missing_extra_instruction_path(tmp_path: Path) -> None:
    task_dir = _make_task_dir(tmp_path)
    task = TaskConfig(path=task_dir)
    extra_instruction_paths = [Path("extra-no-multimodal-hint.md")]
    trial = _trial(
        task,
        extra_instruction_paths=extra_instruction_paths,
    )

    with pytest.raises(FileNotFoundError, match="Extra instruction file not found"):
        build_job_lock(
            config=JobConfig(
                job_name="job",
                tasks=[task],
                extra_instruction_paths=extra_instruction_paths,
            ),
            trial_configs=[trial],
            invocation=["harbor", "run"],
        )


def test_agent_skill_lock_equality_ignores_source_path() -> None:
    digest = _sha("e")
    assert lock_models.AgentSkillLock(
        name="skill", source=Path("/tmp/skill"), digest=digest
    ) == lock_models.AgentSkillLock(
        name="skill", source=Path("/other/skill"), digest=digest
    )


def test_job_lock_equality_ignores_agent_skill_source_path(tmp_path: Path) -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("e"))
    root = tmp_path / "skills"
    _make_skill(root, "alpha", "# alpha\n")
    agent = AgentConfig(name="claude-code", skills=[root])
    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[agent]),
        trial_configs=[_trial(task, agent=agent)],
        invocation=["harbor", "run"],
    )

    skill_lock = lock.trials[0].skills[0]
    other_trial = lock.trials[0].model_copy(
        update={
            "skills": [skill_lock.model_copy(update={"source": Path("/other/alpha")})]
        }
    )
    other_lock = lock.model_copy(update={"trials": [other_trial]})

    assert lock == other_lock


def test_job_lock_equality_ignores_agent_skill_input_path(tmp_path: Path) -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("e"))
    first_skill = _make_skill(tmp_path / "first-skills", "alpha", "# alpha\n")
    second_skill = _make_skill(tmp_path / "second-skills", "alpha", "# alpha\n")
    first_agent = AgentConfig(name="claude-code", skills=[first_skill])
    second_agent = AgentConfig(name="claude-code", skills=[second_skill])

    first_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[first_agent]),
        trial_configs=[_trial(task, agent=first_agent)],
        invocation=["harbor", "run"],
    )
    second_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[second_agent]),
        trial_configs=[_trial(task, agent=second_agent)],
        invocation=["harbor", "run"],
    )

    assert first_lock == second_lock


def test_agent_skill_locks_include_sorted_sources_and_digests(tmp_path: Path) -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("e"))
    root = tmp_path / "skills"
    beta = _make_skill(root, "beta", "# beta\n")
    alpha = _make_skill(root, "alpha", "# alpha\nextra\n")
    agent = AgentConfig(name="claude-code", skills=[root])

    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[agent]),
        trial_configs=[_trial(task, agent=agent)],
        invocation=["harbor", "run"],
    )

    skill_locks = lock.trials[0].skills
    assert [skill.name for skill in skill_locks] == ["alpha", "beta"]
    assert skill_locks[0].source == alpha.resolve()
    assert skill_locks[0].digest == compute_skill_digest(alpha)
    assert skill_locks[1].source == beta.resolve()
    assert skill_locks[1].digest == compute_skill_digest(beta)


def test_lock_uses_pruned_trial_locks_without_job_level_duplicates() -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("e"))
    agent = AgentConfig(name="claude-code", model_name="claude-opus-4-1")
    environment = EnvironmentConfig(
        type=None,
        import_path="custom.env:Environment",
        env={"ENV_SECRET": "secret-value-123"},
    )
    verifier = VerifierConfig(
        override_timeout_sec=7.0,
        max_timeout_sec=8.0,
        env={"VERIFIER_MODE": "strict"},
        disable=True,
    )
    config = JobConfig(
        job_name="job",
        tasks=[task],
        agents=[agent],
        timeout_multiplier=2.0,
        agent_timeout_multiplier=3.0,
        verifier_timeout_multiplier=4.0,
        agent_setup_timeout_multiplier=5.0,
        environment_build_timeout_multiplier=6.0,
        environment=environment,
        verifier=verifier,
    )
    trial = _trial(
        task,
        agent=agent,
        timeout_multiplier=config.timeout_multiplier,
        agent_timeout_multiplier=config.agent_timeout_multiplier,
        verifier_timeout_multiplier=config.verifier_timeout_multiplier,
        agent_setup_timeout_multiplier=config.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=config.environment_build_timeout_multiplier,
        environment=environment,
        verifier=verifier,
    )

    lock = build_job_lock(
        config=config,
        trial_configs=[trial],
        invocation=["harbor", "run"],
    )

    data = lock.model_dump(mode="json")
    assert "requested_config" not in data
    assert "config_path" not in data
    assert "config_hash" not in data
    assert "updated_at" not in data
    assert "cli_invocation" not in data
    assert "n_attempts" not in data
    assert "agents" not in data
    assert "environment" not in data
    assert "verifier" not in data
    assert "timeout_multiplier" not in data
    assert "datasets" not in data
    assert "created_at" in data
    assert data["invocation"] == ["harbor", "run"]
    assert data["trials"][0]["task"]["type"] == "package"
    assert "kind" not in data["trials"][0]["task"]
    assert data["trials"][0]["task"]["digest"] == _sha("e")
    trial_lock = data["trials"][0]
    assert "config" not in trial_lock
    assert "trials_dir" not in trial_lock
    assert "job_id" not in trial_lock
    assert "artifacts" not in trial_lock
    assert trial_lock["timeout_multiplier"] == 2.0
    assert trial_lock["agent_timeout_multiplier"] == 3.0
    assert trial_lock["verifier_timeout_multiplier"] == 4.0
    assert trial_lock["agent_setup_timeout_multiplier"] == 5.0
    assert trial_lock["environment_build_timeout_multiplier"] == 6.0
    assert trial_lock["agent"]["model_name"] == "claude-opus-4-1"
    assert trial_lock["environment"]["import_path"] == "custom.env:Environment"
    assert trial_lock["environment"]["env"]["ENV_SECRET"] == "secr****123"
    assert trial_lock["verifier"] == {
        "override_timeout_sec": 7.0,
        "max_timeout_sec": 8.0,
        "env": {"VERIFIER_MODE": "strict"},
        "disable": True,
    }


def test_harbor_metadata_uses_git_commit_hash_and_editable_install(
    monkeypatch,
) -> None:
    monkeypatch.setattr(lock_models, "_get_harbor_version", lambda: "1.2.3")
    monkeypatch.setattr(lock_models, "_get_harbor_git_commit_hash", lambda: "abc123")
    monkeypatch.setattr(lock_models, "_get_harbor_is_editable_install", lambda: True)
    task = TaskConfig(name="test-org/test-task", ref=_sha("f"))

    lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task]),
        trial_configs=[_trial(task)],
        invocation=["harbor", "run"],
    )

    harbor_data = lock.model_dump(mode="json")["harbor"]
    assert harbor_data == {
        "version": "1.2.3",
        "git_commit_hash": "abc123",
        "is_editable": True,
    }
    assert "git_sha" not in harbor_data
    assert "is_editable_install" not in harbor_data


def test_harbor_git_commit_hash_uses_direct_url_vcs_commit(monkeypatch) -> None:
    commit_id = "a" * 40
    monkeypatch.setattr(
        lock_models,
        "_get_harbor_direct_url_data",
        lambda: {
            "url": "https://github.com/harbor-framework/harbor.git",
            "vcs_info": {"vcs": "git", "commit_id": commit_id},
        },
    )

    def fail_git_lookup(_repo_path: Path) -> str:
        raise AssertionError("git should not be called for direct_url vcs_info")

    monkeypatch.setattr(lock_models, "_get_git_commit_hash", fail_git_lookup)

    assert lock_models._get_harbor_git_commit_hash() == commit_id


def test_harbor_git_commit_hash_uses_editable_direct_url_path(
    monkeypatch, tmp_path: Path
) -> None:
    repo_path = tmp_path / "harbor repo"
    repo_path.mkdir()
    commit_id = "b" * 40
    git_lookup_paths: list[Path] = []

    monkeypatch.setattr(
        lock_models,
        "_get_harbor_direct_url_data",
        lambda: {
            "url": repo_path.as_uri(),
            "dir_info": {"editable": True},
        },
    )

    def fake_git_lookup(path: Path) -> str:
        git_lookup_paths.append(path)
        return commit_id

    monkeypatch.setattr(lock_models, "_get_git_commit_hash", fake_git_lookup)

    assert lock_models._get_harbor_git_commit_hash() == commit_id
    assert git_lookup_paths == [repo_path]


def test_harbor_git_commit_hash_ignores_noneditable_file_direct_url(
    monkeypatch, tmp_path: Path
) -> None:
    repo_path = tmp_path / "harbor"
    repo_path.mkdir()
    monkeypatch.setattr(
        lock_models,
        "_get_harbor_direct_url_data",
        lambda: {
            "url": repo_path.as_uri(),
            "dir_info": {"editable": False},
        },
    )

    def fail_git_lookup(_repo_path: Path) -> str:
        raise AssertionError("git should not be called for noneditable file installs")

    monkeypatch.setattr(lock_models, "_get_git_commit_hash", fail_git_lookup)

    assert lock_models._get_harbor_git_commit_hash() is None


def test_cli_invocation_redacts_sensitive_env_values(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_TOKEN", "tok-abc123xyz")

    argv = sanitize_cli_invocation(
        [
            "harbor",
            "run",
            "--ae",
            "AUTH_TOKEN=tok-abc123xyz",
            "--verifier-env=OPENAI_API_KEY=sk-not-in-host-env",
            "--agent-env",
            "AWS_REGION=us-east-1",
        ]
    )

    assert argv == [
        "harbor",
        "run",
        "--ae",
        "AUTH_TOKEN=${AUTH_TOKEN}",
        "--verifier-env=OPENAI_API_KEY=sk-n****env",
        "--agent-env",
        "AWS_REGION=us-east-1",
    ]


def test_cli_invocation_preserves_capability_url() -> None:
    endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"

    argv = sanitize_cli_invocation(
        [
            "harbor",
            "run",
            "--ae",
            f"HARBOR_PI_API_BASE={endpoint}",
            "--ak",
            f"api_base={endpoint}",
        ]
    )

    assert argv[3] == f"HARBOR_PI_API_BASE={endpoint}"
    assert argv[-1] == f"api_base={endpoint}"


@pytest.mark.parametrize(
    "endpoint_env",
    ["HARBOR_PI_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"],
)
def test_job_lock_equality_ignores_agent_endpoint_env_drift(
    endpoint_env: str,
) -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("a"))
    old_agent = AgentConfig(
        name="pi",
        model_name="hosted_vllm/test-model",
        kwargs={"api_base": "https://old.example/v1"},
        env={endpoint_env: "https://old.example/v1"},
    )
    new_agent = old_agent.model_copy(
        update={
            "kwargs": {"api_base": "https://new.example/v1"},
            "env": {endpoint_env: "https://new.example/v1"},
        }
    )

    old_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[old_agent]),
        trial_configs=[_trial(task, agent=old_agent)],
    )
    new_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[new_agent]),
        trial_configs=[_trial(task, agent=new_agent)],
    )

    assert old_lock == new_lock


def test_job_lock_equality_still_catches_agent_semantic_env_drift() -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("a"))
    old_agent = AgentConfig(
        name="pi",
        model_name="hosted_vllm/test-model",
        env={"PI_CONFIG": "old"},
    )
    new_agent = old_agent.model_copy(update={"env": {"PI_CONFIG": "new"}})

    old_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[old_agent]),
        trial_configs=[_trial(task, agent=old_agent)],
    )
    new_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], agents=[new_agent]),
        trial_configs=[_trial(task, agent=new_agent)],
    )

    assert old_lock != new_lock


@pytest.mark.parametrize(
    ("overrides", "matches"),
    [
        (
            {
                "OPENAI_BASE_URL": "https://iris.example:8443/proxy/t/new-token/serve.new/v1?token=new#api"
            },
            True,
        ),
        ({"OPENAI_BASE_URL": "https://another.example/v1"}, False),
        ({"OPENAI_BASE_URL": "http://iris.example/v1"}, False),
        ({"MODEL_NAME": "different-judge"}, False),
        ({"REASONING_EFFORT": "high"}, False),
    ],
)
def test_judge_endpoint_rotation_preserves_resume_identity(overrides, matches) -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("a"))
    old_env = {
        "OPENAI_BASE_URL": "https://iris.example/proxy/t/old-token/serve.old/v1",
        "MODEL_NAME": "judge-model",
        "REASONING_EFFORT": "low",
    }
    old_trial = _trial(task, verifier=VerifierConfig(env=old_env))
    saved_trial = TrialConfig.model_validate_json(old_trial.model_dump_json())
    new_trial = _trial(task, verifier=VerifierConfig(env=old_env | overrides))

    assert saved_trial.matches_identity(new_trial) is matches
    config = JobConfig(job_name="job", tasks=[task])
    saved_lock = JobLock.model_validate_json(
        build_job_lock(config=config, trial_configs=[old_trial]).model_dump_json()
    )
    new_lock = build_job_lock(config=config, trial_configs=[new_trial])
    assert (saved_lock == new_lock) is matches
    assert new_trial.verifier.env == old_env | overrides
    assert new_lock.trials[0].verifier.env == old_env | overrides


def test_judge_endpoint_rotation_without_model_identity_is_rejected() -> None:
    task = TaskConfig(name="test-org/test-task", ref=_sha("a"))
    trials = [
        _trial(task, verifier=VerifierConfig(env={"OPENAI_BASE_URL": url}))
        for url in ("https://iris.example/old/v1", "https://iris.example/new/v1")
    ]
    assert not trials[0].matches_identity(trials[1])
    config = JobConfig(job_name="job", tasks=[task])
    locks = [build_job_lock(config=config, trial_configs=[trial]) for trial in trials]
    assert locks[0] != locks[1]


@pytest.mark.parametrize(
    "provisioning_kwarg",
    [
        {"auto_snapshot": True},
        {"snapshot_template_name": "harbor__{name}__snapshot"},
        {"assume_global_snapshot": True},
    ],
)
def test_job_lock_equality_ignores_environment_cache_acquisition_kwargs(
    provisioning_kwarg: dict,
) -> None:
    """Daytona cache-acquisition kwargs (auto_snapshot and the sibling snapshot
    knobs) must not factor into lock identity. A resume that turns
    auto_snapshot on (or off) re-resolves the same lock, so
    ``_write_job_lock`` must not false-reject with FileExistsError."""
    task = TaskConfig(name="test-org/test-task", ref=_sha("a"))
    without = EnvironmentConfig(kwargs={})
    with_flag = EnvironmentConfig(kwargs=provisioning_kwarg)

    without_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=without),
        trial_configs=[_trial(task, environment=without)],
        invocation=["harbor", "run"],
    )
    with_flag_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=with_flag),
        trial_configs=[_trial(task, environment=with_flag)],
        invocation=["harbor", "run"],
    )

    assert without_lock == with_flag_lock


def test_job_lock_equality_still_catches_environment_semantic_kwarg_drift() -> None:
    """A non-cache-acquisition environment kwarg still changes lock identity —
    the cache-acquisition exclusion must not over-broaden."""
    task = TaskConfig(name="test-org/test-task", ref=_sha("a"))
    off = EnvironmentConfig(kwargs={"network_block_all": False})
    on = EnvironmentConfig(kwargs={"network_block_all": True})

    off_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=off),
        trial_configs=[_trial(task, environment=off)],
        invocation=["harbor", "run"],
    )
    on_lock = build_job_lock(
        config=JobConfig(job_name="job", tasks=[task], environment=on),
        trial_configs=[_trial(task, environment=on)],
        invocation=["harbor", "run"],
    )

    assert off_lock != on_lock
