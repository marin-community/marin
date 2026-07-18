# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the step-naming + ckpt-naming convention.

`Checkpointer.on_step` receives the count of completed steps (0 before any step
has run). After the
trainer calls the hook with ``step=info.next_step`` (= ``state.step``), the
modulo trigger and on-disk destination should line up so that:

- A save every ``every=k`` fires on ``state.step = k, 2k, 3k, ...`` (never on
  ``k-1, 2k-1, ...``).
- The on-disk name is ``step-N`` where N is the number of completed steps.
- A forced save (e.g. end-of-training) at a non-boundary state.step lands at
  ``step-{state.step}``.
- A forced save at a state.step that just had a periodic save is a no-op
  (no dup write).

The historical bug: the trainer passed ``info.step`` (= ``state.step - 1``)
to the checkpointer, so a 50000-step run ended with ``step-49000`` (periodic)
and ``step-49999`` (force-save) — gap 999, off-by-one.
"""

import pathlib
import tempfile

import jax
import pytest

from levanter.callbacks import StepInfo
from levanter.checkpoint import Checkpointer, CheckpointInterval, _load_metadata
from levanter.trainer_state import TrainerState


def _step_info(state_step: int) -> StepInfo:
    """Build a minimal StepInfo where ``state.step == state_step``."""
    return StepInfo(
        state=TrainerState(
            step=state_step,
            model=None,
            optimizer=None,  # type: ignore[arg-type]
            opt_state=None,
            training_key=jax.random.PRNGKey(0),
            is_trainable=True,
            mp=None,
            model_averaging=None,
        ),
        loss=0.0,
        step_duration=0.0,
    )


def _on_step_new_semantics(checkpointer: Checkpointer, state_step: int, *, force: bool = False):
    """Invoke the checkpointer the way the post-fix trainer does (step=next_step)."""
    info = _step_info(state_step)
    checkpointer.on_step(tree=info.state.saveable_state, step=info.next_step, force=force)


def _checkpoint_names(checkpoint_dir) -> list[str]:
    """List the on-disk basenames of any saved checkpoints, sorted by numeric step."""
    paths = list(pathlib.Path(checkpoint_dir).iterdir())
    return [p.name for p in sorted(paths, key=lambda p: _load_metadata(p)["step"])]


def _checkpoint_steps(checkpoint_dir) -> list[int]:
    """List the metadata step values of any saved checkpoints, sorted."""
    paths = list(pathlib.Path(checkpoint_dir).iterdir())
    return sorted(_load_metadata(p)["step"] for p in paths)


def test_step_info_numbering():
    """``step`` is 0-indexed (state.step - 1); ``next_step`` is the 1-based completed-step count."""
    info = _step_info(state_step=42)
    assert info.step == 41
    assert info.next_step == 42


def test_periodic_saves_fire_at_boundary():
    """Modulo trigger every 10 should fire at state.step = 10, 20, ..., 50, naming step-N accordingly."""
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, None, [CheckpointInterval(every=10)])
        for state_step in range(0, 51):
            _on_step_new_semantics(checkpointer, state_step)
        checkpointer.wait_until_finished()

        # We should have saves at state.step = 10, 20, 30, 40, 50 — NOT 9, 19, 29, 39, 49.
        assert _checkpoint_steps(tmpdir) == [10, 20, 30, 40, 50]
        # On-disk destinations match the same numbers.
        assert _checkpoint_names(tmpdir) == ["step-10", "step-20", "step-30", "step-40", "step-50"]


def test_changing_policy_uses_new_semantics():
    """Re-run the existing changing-policy regression under the new semantics."""
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(
            tmpdir,
            None,
            [
                CheckpointInterval(every=2, until=10),
                CheckpointInterval(every=5, until=20),
                CheckpointInterval(every=10, until=None),
            ],
        )
        for state_step in range(1, 50):
            _on_step_new_semantics(checkpointer, state_step)
        checkpointer.wait_until_finished()

        assert _checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10, 15, 20, 30, 40]
        assert _checkpoint_names(tmpdir) == [
            "step-2",
            "step-4",
            "step-6",
            "step-8",
            "step-10",
            "step-15",
            "step-20",
            "step-30",
            "step-40",
        ]


def test_force_save_at_nonboundary_step():
    """End-of-training force-save at e.g. state.step = 53 lands at step-53 (catches non-boundary final state)."""
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, None, [CheckpointInterval(every=10)])
        for state_step in range(0, 54):
            _on_step_new_semantics(checkpointer, state_step)
        # Mimic Trainer.train()'s final `self.run_hooks(info, force=True)`.
        _on_step_new_semantics(checkpointer, 53, force=True)
        checkpointer.wait_until_finished()

        # Periodic saves at 10/20/30/40/50, plus a forced save at 53.
        assert _checkpoint_steps(tmpdir) == [10, 20, 30, 40, 50, 53]
        assert _checkpoint_names(tmpdir) == [
            "step-10",
            "step-20",
            "step-30",
            "step-40",
            "step-50",
            "step-53",
        ]


def test_force_save_dedup_with_periodic():
    """Force-save at a state.step that just got a periodic save is a no-op (no dup write).

    Scenario: a 50-step run with ``every=10`` ends at state.step = 50. The periodic
    save fires at state.step = 50; then the trainer's end-of-training
    ``run_hooks(info, force=True)`` calls back into the checkpointer with step=50
    and force=True. The guard should suppress the second write.
    """
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, None, [CheckpointInterval(every=10)])
        for state_step in range(1, 51):  # 50 training steps → periodic save at 50
            _on_step_new_semantics(checkpointer, state_step)
        # Sanity: periodic just landed at step 50.
        assert checkpointer._last_save_step == 50

        # Mimic Trainer.train()'s final run_hooks(info, force=True).
        _on_step_new_semantics(checkpointer, 50, force=True)
        checkpointer.wait_until_finished()

        # Periodic saves at 10/20/30/40/50, and NO dup at 50.
        assert _checkpoint_steps(tmpdir) == [10, 20, 30, 40, 50]
        assert _checkpoint_names(tmpdir) == [
            "step-10",
            "step-20",
            "step-30",
            "step-40",
            "step-50",
        ]
        assert checkpointer._last_save_step == 50


def test_state_step_zero_no_save_without_force():
    """``state.step == 0`` (0 completed steps) should not save under the modulo trigger."""
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, None, [CheckpointInterval(every=10)])
        _on_step_new_semantics(checkpointer, 0)
        checkpointer.wait_until_finished()
        assert _checkpoint_steps(tmpdir) == []


def test_state_step_zero_saves_when_forced():
    """A forced save at ``state.step == 0`` must still write.

    Regression: ``_last_save_step`` is None until the first save, so the dedup
    guard must not treat an as-yet-unsaved step 0 as "already saved" (which a
    0-initialized sentinel did, swallowing e.g. a save-initial-checkpoint hook).
    """
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, None, [CheckpointInterval(every=10)])
        _on_step_new_semantics(checkpointer, 0, force=True)
        checkpointer.wait_until_finished()
        assert _checkpoint_steps(tmpdir) == [0]
        assert _checkpoint_names(tmpdir) == ["step-0"]


def test_final_step_naming_matches_count_of_completed_steps():
    """End-of-training: a 50-step run with every=10 should end at step-50 (not step-49)."""
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, None, [CheckpointInterval(every=10)])
        for state_step in range(1, 51):  # 50 training steps → state.step ends at 50
            _on_step_new_semantics(checkpointer, state_step)
        checkpointer.wait_until_finished()

        assert _checkpoint_steps(tmpdir)[-1] == 50
        assert _checkpoint_names(tmpdir)[-1] == "step-50"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
