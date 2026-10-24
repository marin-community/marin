"""PBS shaping vector: EDIT-token placement, telescoping to the final potential, and the total bound."""

from skyrl_train.utils.span_tagger import SPAN_OTHER, SPAN_THINK, SPAN_ACTION, SPAN_EDIT
from skyrl_train.utils.pbs_shaping import compute_pbs_token_shaping
from skyrl_train.utils.test_delta_parser import TestRunResult


def _run(idx, passed, failed):
    return TestRunResult(
        message_index=idx, passed=passed, failed=failed, total_runnable=passed + failed, framework="pytest"
    )


# ---------------------------------------------------------------------------
# PBS vector: shape, location, telescoping, bound
# ---------------------------------------------------------------------------


def test_pbs_credits_edit_tokens_only():
    # Layout: [think, edit, edit, OTHER(obs), think, edit] — 2 edit turns.
    tags = [SPAN_THINK, SPAN_EDIT, SPAN_EDIT, SPAN_OTHER, SPAN_THINK, SPAN_EDIT]
    runs = [_run(99, 0, 2), _run(99, 1, 1), _run(99, 2, 0)]
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=10.0, test_runs=runs)
    assert len(vec) == len(tags)
    for j, t in enumerate(tags):
        if t != SPAN_EDIT:
            assert vec[j] == 0.0, f"non-edit token {j} got shaping"
    assert any(vec[j] != 0.0 for j in (1, 2, 5))


def test_pbs_telescopes_to_potential_difference():
    # With gamma=1, sum of shaping == Φ_final − Φ_0.
    tags = [SPAN_EDIT, SPAN_OTHER, SPAN_EDIT]
    runs = [_run(99, 1, 3), _run(99, 3, 1)]  # 0.25 -> 0.75
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=10.0, test_runs=runs)
    total = sum(vec)
    # Φ(s0)=0, Φ(final)=0.75 ; telescopes to 0.75.
    assert abs(total - 0.75) < 1e-9


def test_pbs_bounded():
    tags = [SPAN_EDIT, SPAN_EDIT]
    runs = [_run(99, 10, 0)]  # frac 1.0 -> potential 1.0 > bound 0.3
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=0.3, test_runs=runs)
    assert abs(sum(vec)) <= 0.3 + 1e-9
    assert abs(sum(vec) - 0.3) < 1e-9  # scaled to the ceiling


def test_pbs_no_test_signal_is_zeros():
    tags = [SPAN_EDIT, SPAN_THINK, SPAN_EDIT]
    vec = compute_pbs_token_shaping(None, tags, test_runs=[])
    assert vec == [0.0, 0.0, 0.0]


def test_pbs_no_edit_turn_is_zeros():
    # Tests moved but the turn is ACTION (not EDIT) -> nothing to credit.
    tags = [SPAN_THINK, SPAN_ACTION, SPAN_OTHER]
    runs = [_run(99, 1, 0)]
    vec = compute_pbs_token_shaping(None, tags, test_runs=runs)
    assert all(v == 0.0 for v in vec)


def test_pbs_scatter_is_uniform_within_turn():
    tags = [SPAN_EDIT, SPAN_EDIT, SPAN_EDIT]  # one 3-token edit turn
    runs = [_run(99, 2, 2)]  # 0 -> 0.5
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=10.0, test_runs=runs)
    assert abs(sum(vec) - 0.5) < 1e-9
    assert abs(vec[0] - vec[1]) < 1e-12 and abs(vec[1] - vec[2]) < 1e-12
