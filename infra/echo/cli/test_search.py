# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from infra.echo.cli import search as echo_search


def test_validate_db_user_rejects_personal_adc() -> None:
    with pytest.raises(SystemExit):
        echo_search.validate_db_user("alice@gmail.com")


@pytest.mark.parametrize("user", ["alice@openathena.ai", "echo-api@hai-gcp-models.iam"])
def test_validate_db_user_accepts_authorized_identity(user: str) -> None:
    assert echo_search.validate_db_user(user) == user
