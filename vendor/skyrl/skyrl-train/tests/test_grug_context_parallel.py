import pytest

from skyrl_train.models.grug_context_parallel import grug_cp_communication, validate_cp_attention


def test_snowball_cp4_preserves_sliding_windows_without_splitting_five_kv_heads():
    shape = dict(query_heads=20, kv_heads=5, tensor_parallel=1, context_parallel=4)
    with pytest.raises(ValueError, match="divisible by CP"):
        validate_cp_attention(**shape, communication="a2a", sliding_window=True)
    for communication in ("p2p", "a2a+p2p"):
        with pytest.raises(ValueError, match="sliding-window"):
            validate_cp_attention(**shape, communication=communication, sliding_window=True)
    modes = grug_cp_communication((False, False, True))
    for is_long, communication in zip((False, False, True), modes, strict=True):
        validate_cp_attention(**shape, communication=communication, sliding_window=not is_long)
    assert modes == ["all_gather", "all_gather", "p2p"]
