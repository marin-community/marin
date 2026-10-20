"""Episode logging is DEFAULT-OFF (throughput fix).

`debug.json` (the full raw litellm request+response, written synchronously by
litellm's logger_fn) plus the per-turn prompt/response writes stall the shared
rollout event loop when logs_dir is remote (s3://). `enable_episode_logging`
defaults False and gates `_setup_episode_logging` so those paths are None ->
litellm's logger_fn is never attached and no per-turn object-store write happens.

Ref: agent_logs/2026-07-03_r5_engine_starvation_rootcause.md
"""

from harbor.agents.terminus_2.terminus_2 import EpisodeLoggingPaths, Terminus2


def _agent(enabled: bool) -> Terminus2:
    # Construct without running __init__ (which builds an LLM); set only the
    # attribute the method under test reads. Mirrors the trajectory-persist test.
    agent = object.__new__(Terminus2)
    agent._enable_episode_logging = enabled
    return agent


def test_episode_logging_off_returns_none_and_writes_nothing(tmp_path):
    agent = _agent(False)
    assert agent._setup_episode_logging(tmp_path, 0) == EpisodeLoggingPaths(
        debug=None, prompt=None, response=None
    )
    # No episode dir is even created -> no debug.json/prompt/response writes.
    assert not (tmp_path / "episode-0").exists()


def test_episode_logging_on_returns_paths(tmp_path):
    agent = _agent(True)
    paths = agent._setup_episode_logging(tmp_path, 2)
    assert paths.debug == tmp_path / "episode-2" / "debug.json"
    assert paths.prompt == tmp_path / "episode-2" / "prompt.txt"
    assert paths.response == tmp_path / "episode-2" / "response.txt"
    # No eager mkdir: the artifact writer creates local parents on write, and
    # object stores need no directories (a remote mkdir would be a blocking
    # round-trip on the event loop).
    assert not (tmp_path / "episode-2").exists()


def test_episode_logging_on_but_no_dir_still_none(tmp_path):
    agent = _agent(True)
    assert agent._setup_episode_logging(None, 0) == EpisodeLoggingPaths(
        debug=None, prompt=None, response=None
    )
