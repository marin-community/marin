# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes exec-gated operator login. Run only inside the controller container."""

import argparse
import secrets
from pathlib import Path

from rigging.timing import Timestamp
from sqlalchemy import create_engine, insert

from iris.cluster.config import load_config, resolve_config_secrets
from iris.cluster.controller.auth import SESSION_TOKEN_TTL_SECONDS, create_controller_auth
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.main import LOCAL_STATE_DIR_DEFAULT
from iris.cluster.controller.schema import task_credential_audit_table


def mint_operator_token(config_path: str, principal: str) -> str:
    """Exercise existing Pod-exec authority, recording the mint before releasing a JWT."""
    config = resolve_config_secrets(load_config(config_path))
    if config.auth is None or not isinstance(config.auth.signing_key, str) or not config.auth.signing_key:
        raise ValueError("Operator login requires a configured controller signing key")
    auth = create_controller_auth(config.auth, cluster_name=config.name, signing_key_pem=config.auth.signing_key)
    assert auth.jwt_manager is not None
    token = auth.jwt_manager.create_token(principal, "admin", secrets.token_urlsafe(16), SESSION_TOKEN_TTL_SECONDS)
    db_path = Path(config.storage.local_state_dir or LOCAL_STATE_DIR_DEFAULT) / ControllerDB.DB_FILENAME
    # Open the existing database without schema creation or migrations in this
    # short-lived exec process. Audit commit is required before releasing a JWT.
    engine = create_engine(f"sqlite:///file:{db_path}?mode=rw&uri=true")
    try:
        with engine.begin() as tx:
            tx.execute(
                insert(task_credential_audit_table).values(
                    at_ms=Timestamp.now(),
                    actor=principal,
                    action="MINT_OPERATOR_TOKEN",
                    scope="cluster",
                    target=config.name,
                )
            )
    finally:
        engine.dispose()
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--principal", required=True)
    args = parser.parse_args()
    print(mint_operator_token(args.config, args.principal))


if __name__ == "__main__":
    main()
