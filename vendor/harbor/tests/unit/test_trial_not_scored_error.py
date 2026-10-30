"""Unit tests for the reworked verifier-result invariant + its score-invariance.

``Trial._enforce_verifier_result_invariant`` no longer flattens an informative
upstream error (e.g. ``EnvironmentStartTimeoutError``) into a
``VerificationNotCompletedError`` (VNC). New semantics:

  - verifier-disabled / multi-step / ``verifier_result is not None`` -> early return (untouched).
  - prior ``exception_info`` present -> LEAVE IT UNTOUCHED (preserve the true cause).
  - prior empty, verifier reached -> stamp ``VerificationNotCompletedError`` (a ``TrialNotScoredError``).
  - prior empty, verifier NOT reached -> stamp the catch-all ``TrialNotScoredError`` base.

The load-bearing parity invariant is SCORE-INVARIANCE: the scored denominator
``n_trials`` and ``reward_stats`` in ``JobStats.increment`` key off
``verifier_result`` (not the exception type), so relabelling an unscored trial
changes ONLY ``exception_stats`` bucketing — never the score.
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from harbor.models.job.result import JobStats
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.trial.errors import (
    EnvironmentStartTimeoutError,
    TrialNotScoredError,
    VerificationNotCompletedError,
)
from harbor.trial.trial import Trial


# --------------------------------------------------------------------------- #
# Gate 1: the four-case truth table for _enforce_verifier_result_invariant
# --------------------------------------------------------------------------- #


def _fake_trial(
    *,
    verifier_result,
    exception_info,
    reached_verifier: bool,
    tmp_path: Path,
    disable: bool = False,
    has_steps: bool = False,
) -> SimpleNamespace:
    """A minimal stand-in carrying only what the invariant method reads/writes."""
    return SimpleNamespace(
        config=SimpleNamespace(verifier=SimpleNamespace(disable=disable)),
        task=SimpleNamespace(has_steps=has_steps),
        result=SimpleNamespace(
            verifier_result=verifier_result,
            exception_info=exception_info,
        ),
        _reached_verifier=reached_verifier,
        paths=SimpleNamespace(
            exception_message_path=(tmp_path / "exception_message.txt")
        ),
    )


def _run_invariant(fake: SimpleNamespace) -> None:
    # Call the real method against the fake (it only touches the attrs above).
    # Round-3: the invariant is now a coroutine (it offloads its exception.txt
    # write off the coordinator event loop), so drive it to completion here.
    import asyncio

    asyncio.run(Trial._enforce_verifier_result_invariant(fake))


def _exc_info_from(exc: BaseException) -> ExceptionInfo:
    return ExceptionInfo.from_exception(exc)


def test_prior_env_start_timeout_is_preserved_not_wrapped(tmp_path: Path):
    """prior=EnvironmentStartTimeoutError, verifier_result=None,
    reached_verifier=False -> exception_info UNCHANGED (still
    EnvironmentStartTimeoutError, NOT wrapped)."""
    prior = _exc_info_from(EnvironmentStartTimeoutError("env start timed out"))
    fake = _fake_trial(
        verifier_result=None,
        exception_info=prior,
        reached_verifier=False,
        tmp_path=tmp_path,
    )
    _run_invariant(fake)
    assert fake.result.exception_info is prior
    assert fake.result.exception_info.exception_type == "EnvironmentStartTimeoutError"


def test_reached_verifier_no_result_is_vnc(tmp_path: Path):
    """prior=None, verifier_result=None, reached_verifier=True ->
    VerificationNotCompletedError, and isinstance(err, TrialNotScoredError)."""
    fake = _fake_trial(
        verifier_result=None,
        exception_info=None,
        reached_verifier=True,
        tmp_path=tmp_path,
    )
    _run_invariant(fake)
    assert fake.result.exception_info is not None
    assert fake.result.exception_info.exception_type == "VerificationNotCompletedError"
    # VNC is a TrialNotScoredError subtype.
    assert issubclass(VerificationNotCompletedError, TrialNotScoredError)


def test_not_reached_verifier_no_result_is_base(tmp_path: Path):
    """prior=None, verifier_result=None, reached_verifier=False ->
    TrialNotScoredError (NOT the VNC subclass)."""
    fake = _fake_trial(
        verifier_result=None,
        exception_info=None,
        reached_verifier=False,
        tmp_path=tmp_path,
    )
    _run_invariant(fake)
    assert fake.result.exception_info is not None
    assert fake.result.exception_info.exception_type == "TrialNotScoredError"


def test_verifier_result_present_is_untouched(tmp_path: Path):
    """verifier_result present -> early return (untouched), regardless of prior."""
    vr = VerifierResult(rewards={"reward": 1})
    # Even with a prior set, an early-return must leave it alone.
    prior = _exc_info_from(EnvironmentStartTimeoutError("env start timed out"))
    fake = _fake_trial(
        verifier_result=vr,
        exception_info=prior,
        reached_verifier=True,
        tmp_path=tmp_path,
    )
    _run_invariant(fake)
    assert fake.result.verifier_result is vr
    assert fake.result.exception_info is prior

    # And with no prior, it must NOT stamp anything.
    fake2 = _fake_trial(
        verifier_result=vr,
        exception_info=None,
        reached_verifier=True,
        tmp_path=tmp_path,
    )
    _run_invariant(fake2)
    assert fake2.result.exception_info is None


def test_disabled_and_multistep_are_untouched(tmp_path: Path):
    """verifier-disabled / multi-step -> early return even when unscored."""
    for disable, has_steps in [(True, False), (False, True)]:
        fake = _fake_trial(
            verifier_result=None,
            exception_info=None,
            reached_verifier=False,
            tmp_path=tmp_path,
            disable=disable,
            has_steps=has_steps,
        )
        _run_invariant(fake)
        assert fake.result.exception_info is None


# --------------------------------------------------------------------------- #
# Gate 2: score-invariance — relabelling an unscored trial changes ONLY
# exception_stats bucketing, never n_trials / reward_stats.
# --------------------------------------------------------------------------- #


def _unscored_trial_result(
    *, exception_type_exc: BaseException, name: str
) -> TrialResult:
    """An UNSCORED trial (verifier_result is None) labelled by the given error."""
    tc = TaskConfig(path="/tmp/x")
    cfg = TrialConfig(task=tc, trials_dir="/tmp/trials")
    return TrialResult(
        task_name="t",
        trial_name=name,
        trial_uri="file:///tmp/x",
        task_id=cfg.task.get_task_id(),
        task_checksum="abc",
        config=cfg,
        agent_info=AgentInfo(name="agent", version="1"),
        source="dataset",
        verifier_result=None,
        exception_info=ExceptionInfo(
            exception_type=type(exception_type_exc).__name__,
            exception_message=str(exception_type_exc),
            exception_traceback="",
            occurred_at=datetime.now(timezone.utc),
        ),
    )


def _scored_trial_result(*, reward, name: str) -> TrialResult:
    """A normally-scored trial that DOES contribute to n_trials / reward_stats."""
    tc = TaskConfig(path="/tmp/x")
    cfg = TrialConfig(task=tc, trials_dir="/tmp/trials")
    return TrialResult(
        task_name="t",
        trial_name=name,
        trial_uri="file:///tmp/x",
        task_id=cfg.task.get_task_id(),
        task_checksum="abc",
        config=cfg,
        agent_info=AgentInfo(name="agent", version="1"),
        source="dataset",
        verifier_result=VerifierResult(rewards={"reward": reward}),
        exception_info=None,
    )


def test_score_is_invariant_to_unscored_label(tmp_path: Path):
    """An unscored trial labelled EnvironmentStartTimeoutError vs
    VerificationNotCompletedError vs TrialNotScoredError yields IDENTICAL
    n_trials and reward_stats; only exception_stats bucketing differs."""
    labels = [
        EnvironmentStartTimeoutError("env start timed out"),
        VerificationNotCompletedError("verifier reached, no result"),
        TrialNotScoredError("never reached verifier"),
    ]

    aggregates = []
    for exc in labels:
        results = [
            _scored_trial_result(reward=1, name="scored-pass"),
            _scored_trial_result(reward=0, name="scored-fail"),
            _unscored_trial_result(exception_type_exc=exc, name="unscored"),
        ]
        stats = JobStats.from_trial_results(results)
        eval_stats = stats.evals["agent__dataset"]
        # Normalise reward_stats (defaultdicts -> plain dict of dicts) for compare.
        reward_norm = {
            k: {rv: sorted(ids) for rv, ids in v.items()}
            for k, v in eval_stats.reward_stats.items()
        }
        exc_norm = {k: sorted(v) for k, v in eval_stats.exception_stats.items()}
        aggregates.append(
            {
                "n_trials": eval_stats.n_trials,
                "n_errors": eval_stats.n_errors,
                "reward_stats": reward_norm,
                "exception_stats": exc_norm,
                "label": type(exc).__name__,
            }
        )

    base = aggregates[0]
    for agg in aggregates[1:]:
        # The score (denominator + reward distribution) is INVARIANT.
        assert agg["n_trials"] == base["n_trials"], (
            agg["label"],
            agg["n_trials"],
            base["n_trials"],
        )
        assert agg["reward_stats"] == base["reward_stats"], agg["label"]
        # n_errors is also invariant (still exactly one errored trial).
        assert agg["n_errors"] == base["n_errors"], agg["label"]

    # Sanity: the two scored trials are the only ones counted; the unscored
    # trial never contributes to n_trials.
    assert base["n_trials"] == 2
    assert base["n_errors"] == 1

    # The ONLY thing that differs across labels is the exception bucket name.
    bucket_names = {tuple(agg["exception_stats"].keys()) for agg in aggregates}
    assert bucket_names == {
        ("EnvironmentStartTimeoutError",),
        ("VerificationNotCompletedError",),
        ("TrialNotScoredError",),
    }
