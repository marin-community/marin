import pytest

from capability_pipeline.synthesis import (
    SynthesisError,
    _validate_environment_fidelity,
)


@pytest.mark.parametrize(
    "admitted,lowered",
    [("reasoning", "none"), ("shellsim", "shellsim"), ("container", "docker")],
)
def test_final_binding_matches_admitted_solver_environment(admitted, lowered):
    _validate_environment_fidelity(
        {"proposal": {"environment": admitted}},
        {
            "environment": {"kind": lowered},
            "tools": []
            if admitted == "reasoning"
            else [{"kind": "harness", "backend": lowered}],
        },
    )


def test_private_code_verifier_does_not_allow_solver_shell_escalation():
    with pytest.raises(SynthesisError, match="requires hash-bound readmission"):
        _validate_environment_fidelity(
            {"proposal": {"environment": "reasoning", "verification": "code"}},
            {
                "environment": {"kind": "shellsim"},
                "tools": [{"kind": "shell", "backend": "shellsim"}],
            },
        )


def test_container_binding_cannot_silently_remove_public_tools():
    with pytest.raises(SynthesisError, match="public tool bindings"):
        _validate_environment_fidelity(
            {"proposal": {"environment": "container"}},
            {"environment": {"kind": "docker"}, "tools": []},
        )
