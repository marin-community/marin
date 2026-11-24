"""Execute task-declared public fixture initialization before agent access."""

from harbor.environments.base import BaseEnvironment
from harbor.trial.errors import EnvironmentSetupError
from harbor_config.models.task.config import EnvironmentSetupConfig


async def run_environment_setup(environment: BaseEnvironment, setup: EnvironmentSetupConfig) -> None:
    """Initialize public fixtures; leave private tests and solutions host-side."""
    result = await environment.exec(setup.command, timeout_sec=setup.timeout_sec, user="root")
    if result.return_code != 0:
        raise EnvironmentSetupError(
            f"Public fixture initialization exited {result.return_code}: {result.stderr or result.stdout}"
        )
