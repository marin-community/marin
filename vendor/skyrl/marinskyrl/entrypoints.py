"""Runtime modules for authored RL execution modes."""

from types import MappingProxyType

from .recipe_schema.rules import RLEntrypoint

RL_ENTRYPOINTS = MappingProxyType(
    {
        RLEntrypoint.GENERATE: "skyrl_train.entrypoints.main_generate",
        RLEntrypoint.STANDARD: "skyrl_train.entrypoints.main_base",
        RLEntrypoint.TASKCOMPENDIUM: "skyrl_train.entrypoints.taskcompendium",
        RLEntrypoint.TERMINAL_BENCH: "skyrl_train.entrypoints.terminal_bench",
        RLEntrypoint.TERMINAL_BENCH_GENERATE: "skyrl_train.entrypoints.terminal_bench_generate",
    }
)
