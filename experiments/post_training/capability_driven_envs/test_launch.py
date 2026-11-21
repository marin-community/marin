# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.capability_driven_envs import launch

REQUIRED = {"CW_KEY_ID": "cw-id-0123456789", "CW_KEY_SECRET": "cw-secret-0123456789", "GLM_API_KEY": "glm-0123456789"}


def commands(*flags: str, **environ: str) -> list[list[str]]:
    argv = ["--proposals", "s3://b/sample.json", "--output", "s3://b/out", "--policy", "s3://b/policy.json"]
    argv += ["--job-prefix", "tf", "--concurrency", "4", "--cpu", "8", *flags]
    return launch.commands(launch.parser().parse_args(argv), {**REQUIRED, **environ})


def values(command: list[str], flag: str) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == flag]


def test_launcher_builds_iris_argv_with_secrets_by_reference_only() -> None:
    secrets = {**REQUIRED, "PARALLEL_API_KEY": "parallel-0123456789"}
    (command,) = commands("--drivers", "1", GLM_RELAY_JOB="/me/relay", PARALLEL_API_KEY=secrets["PARALLEL_API_KEY"])
    iris, driver = command[: command.index("--")], command[command.index("--") + 1 :]
    assert driver[:4] == ["python", "-m", "taskforge.driver", "run"]
    options = {"--target-cluster", "--cpu", "--memory", "--disk", "--max-retries", "--job-name", "--secret", "-e"}
    assert options <= set(iris) and not options & set(driver)
    assert values(iris, "--max-retries") == ["3"] and values(iris, "--job-name") == ["tf-0"]
    assert not [arg for arg in command for value in secrets.values() if value in arg]
    granted = {iris[i + 1]: iris[i + 2] for i, arg in enumerate(iris) if arg == "--secret"}
    assert granted == {key: f"env:{key}" for key in secrets}
    environment = {iris[i + 1]: iris[i + 2] for i, arg in enumerate(iris) if arg == "-e"}
    assert environment == {"GLM_RELAY_JOB": "/me/relay", **{key: "8" for key in launch.THREAD_PINS}}
    assert values(iris, "--cpu") == ["8"] and "--resume-held" not in command
    assert values(driver, "--buckets") == ["4096"] and values(driver, "--bucket-range") == ["0:4096"]


@pytest.mark.parametrize("endpoints", [{}, {"GLM_RELAY_JOB": "/me/relay", "GLM_BASE_URL": "http://glm/v1"}])
def test_launcher_requires_exactly_one_endpoint(endpoints: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        commands("--drivers", "2", **endpoints)


def test_launcher_forwards_a_direct_endpoint_and_resume_held_only_when_asked() -> None:
    for command in commands("--drivers", "3", "--resume-held", GLM_BASE_URL="http://glm/v1"):
        assert values(command, "-e")[0] == "GLM_BASE_URL" and "GLM_RELAY_JOB" not in command
        assert command[-1] == "--resume-held" and command.index("--resume-held") > command.index("--")


@pytest.mark.parametrize("drivers,span", [(1, None), (7, None), (32, None), (4096, None), (32, "24:32"), (64, "48:64")])
def test_launcher_bucket_ranges_partition_the_buckets(drivers: int, span: str | None) -> None:
    flags = ("--drivers", str(drivers), *(("--driver-range", span) if span else ()))
    built = commands(*flags, GLM_RELAY_JOB="/me/relay")
    ranges = [tuple(map(int, values(command, "--bucket-range")[0].split(":"))) for command in built]
    covered = [bucket for start, end in ranges for bucket in range(start, end)]
    assert covered == list(range(3072 if span else 0, 4096))
    assert len(built) == (int(span.split(":")[1]) - int(span.split(":")[0]) if span else drivers)
    assert max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1
