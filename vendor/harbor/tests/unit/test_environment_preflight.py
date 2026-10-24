"""The environment factory admits only maintained Shellbox adapters."""

import pytest
from pydantic import ValidationError

from harbor_config import load_config_document


@pytest.mark.parametrize(
    "environment", [{"type": "docker"}, {"import_path": "untrusted.module:Environment"}]
)
def test_config_rejects_execution_outside_shellbox_before_loading_a_backend(
    environment,
):
    with pytest.raises(ValidationError):
        load_config_document({"environment": environment})
