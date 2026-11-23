# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.capability_driven_envs import launch

REQUIRED = {"CW_KEY_ID": "cw-id-0123456789", "CW_KEY_SECRET": "cw-secret-0123456789", "GLM_API_KEY": "glm-0123456789"}


def base_argv() -> list[str]:
    argv = ["--proposals", "s3://b/sample.json", "--output", "s3://b/out", "--policy", "s3://b/policy.json"]
    return [*argv, "--job-prefix", "tf", "--concurrency", "4", "--cpu", "8"]


def commands(*flags: str, **environ: str) -> list[list[str]]:
    return launch.commands(launch.parser().parse_args([*base_argv(), *flags]), {**REQUIRED, **environ})


def values(command: list[str], flag: str) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == flag]


def test_launcher_builds_iris_argv_with_secrets_by_reference_only() -> None:
    secrets = {**REQUIRED, "PARALLEL_API_KEY": "parallel-0123456789"}
    (command,) = commands("--drivers", "1", "--glm-relay-job", "/me/relay", PARALLEL_API_KEY=secrets["PARALLEL_API_KEY"])
    iris, driver = command[: command.index("--")], command[command.index("--") + 1 :]
    assert driver[:4] == ["python", "-m", "taskforge.driver", "run"]
    options = {"--target-cluster", "--cpu", "--memory", "--disk", "--max-retries", "--job-name", "--secret", "-e"}
    assert options <= set(iris) and not options & set(driver)
    assert values(iris, "--max-retries") == ["3"] and values(iris, "--job-name") == ["tf-0"]
    assert not [arg for arg in command for value in secrets.values() if value in arg]
    granted = {iris[i + 1]: iris[i + 2] for i, arg in enumerate(iris) if arg == "--secret"}
    assert granted == {key: f"env:{key}" for key in secrets}
    environment = {iris[i + 1]: iris[i + 2] for i, arg in enumerate(iris) if arg == "-e"}
    assert environment == {key: "8" for key in launch.THREAD_PINS}
    assert values(driver, "--glm-relay-job") == ["/me/relay"] and "--glm-base-url" not in driver
    assert values(iris, "--cpu") == ["8"] and "--resume-held" not in command
    assert values(driver, "--buckets") == ["4096"] and values(driver, "--bucket-range") == ["0:4096"]


@pytest.mark.parametrize("endpoints", [(), ("--glm-relay-job", "/me/relay", "--glm-base-url", "http://glm/v1")])
def test_launcher_requires_exactly_one_endpoint(endpoints: tuple[str, ...], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        commands("--drivers", "2", *endpoints)
    assert "--glm-relay-job" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value",
    [("--glm-relay-job", ""), ("--glm-relay-job", "  "), ("--glm-base-url", ""), ("--glm-base-url", "http://glm")],
)
def test_launcher_refuses_a_blank_or_malformed_gateway_before_building_a_job(
    flag: str, value: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[list[str]] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **_: built.append(command))
    monkeypatch.setattr("sys.argv", ["launch", "--drivers", "3", flag, value, *base_argv()])
    for key, secret in REQUIRED.items():
        monkeypatch.setenv(key, secret)
    with pytest.raises(SystemExit) as stopped:
        launch.main()
    assert stopped.value.code == 2 and built == []
    assert f"argument {flag}" in capsys.readouterr().err


def test_launcher_forwards_a_direct_endpoint_and_resume_held_only_when_asked() -> None:
    for command in commands("--drivers", "3", "--resume-held", "--glm-base-url", "http://glm/v1"):
        driver = command[command.index("--") + 1 :]
        assert values(driver, "--glm-base-url") == ["http://glm/v1"] and "--glm-relay-job" not in command
        assert command[-1] == "--resume-held" and command.index("--resume-held") > command.index("--")


@pytest.mark.parametrize("drivers,span", [(1, None), (7, None), (32, None), (4096, None), (32, "24:32"), (64, "48:64")])
def test_launcher_bucket_ranges_partition_the_buckets(drivers: int, span: str | None) -> None:
    flags = ("--drivers", str(drivers), *(("--driver-range", span) if span else ()))
    built = commands(*flags, "--glm-relay-job", "/me/relay")
    ranges = [tuple(map(int, values(command, "--bucket-range")[0].split(":"))) for command in built]
    covered = [bucket for start, end in ranges for bucket in range(start, end)]
    assert covered == list(range(3072 if span else 0, 4096))
    assert len(built) == (int(span.split(":")[1]) - int(span.split(":")[0]) if span else drivers)
    assert max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1
