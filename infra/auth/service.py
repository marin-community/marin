# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the shared grant exchange with OpenBao and a fixed CoreWeave policy map."""

import argparse
import asyncio
import hmac
import logging
import ssl
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, Self

import requests
import uvicorn
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.grant_auth import GrantAuthService, create_app
from rigging.grant_database import GrantDatabase, PostgresGrantDatabase, SqliteGrantDatabase
from rigging.grants import GrantDenied, GrantStore, Permission
from rigging.openbao import (
    OpenBaoClient,
    OpenBaoIdentityVerifier,
    OpenBaoStorageIssuer,
    OpenBaoTransitSigner,
    OpenBaoTransitVerifier,
)
from rigging.server_auth import extract_bearer_token
from sqlalchemy.exc import SQLAlchemyError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from infra.auth.iris import IrisAttemptBroker, IrisBrokerConfig, mount_iris_attempts
from infra.auth.loom import LoomLauncher
from infra.auth.openfga import ACTIONS, AuthorizationUnavailable, OpenFGAPolicy, ProviderIdentity
from infra.auth.secret_bindings import OpenBaoSecretReader, SecretConfig

REFRESH_AUDIENCES = frozenset({"cw_s3"})
LOOM_RUNTIME_AUDIENCES = frozenset({"iris", "cw_s3", "openbao"})


class PermissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str
    actions: frozenset[str]
    resource: str

    def permission(self) -> Permission:
        return Permission(self.service, self.actions, self.resource)


class PrincipalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    principal: str
    owner: str
    max_ttl: int
    delegation_depth: int


class OpenFGAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_url: str
    store_id: str
    model_id: str
    token_file: Path
    principals: tuple[PrincipalConfig, ...]


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    role: str
    subject: str
    token_file: Path
    permissions: tuple[PermissionConfig, ...]


class LoomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str
    observer_token_file: Path
    callback_token_file: Path


class LoomLaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template: str
    title: str
    goal: str


class LoomCredentialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource: str
    run_id: str
    session_id: str


class LoomObserverTokenProvider:
    def __init__(self, path: Path):
        self.path = path

    def get_token(self) -> str:
        token = self.path.read_text().strip()
        if not token.startswith("loom_"):
            raise ValueError("The observer requires a native Loom credential")
        return token


class SqliteDatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["sqlite"]
    path: Path

    def connect(self) -> GrantDatabase:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return SqliteGrantDatabase(self.path)


class PostgresDatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["postgres"]
    url_file: Path
    ca_file: Path | None = None

    def connect(self) -> GrantDatabase:
        return PostgresGrantDatabase(
            self.url_file.read_text().strip(), ssl_context=ssl.create_default_context(cafile=self.ca_file)
        )


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bao_address: str
    ca_bundle: str | bool = True
    bao_issuer: str
    issuer: str
    broker_token_file: Path
    signing_key_version: int
    database: Annotated[SqliteDatabaseConfig | PostgresDatabaseConfig, Field(discriminator="backend")]
    subjects: dict[str, str]
    openfga: OpenFGAConfig
    storage: tuple[StorageConfig, ...]
    secrets: tuple[SecretConfig, ...] = ()
    max_token_ttl: int
    loom: LoomConfig | None
    iris: IrisBrokerConfig | None = None

    @model_validator(mode="after")
    def separate_workload_credentials(self) -> Self:
        paths = [
            self.broker_token_file,
            self.openfga.token_file,
            *(binding.token_file for binding in self.storage),
            *(binding.token_file for binding in self.secrets),
        ]
        if self.loom is not None:
            paths.extend((self.loom.observer_token_file, self.loom.callback_token_file))
        resolved = [path.resolve() for path in paths]
        if len(resolved) != len(set(resolved)):
            raise ValueError(
                "Broker, OpenFGA, storage, secret and Loom observer credentials require distinct token files"
            )
        if len({binding.name for binding in self.secrets}) != len(self.secrets):
            raise ValueError("Secret binding names must be unique")
        return self


def create_service(config: ServiceConfig) -> tuple[Starlette, GrantAuthService]:
    """Build one authority from deployment policy and private token-file references."""
    storage_permissions = {
        binding.name: tuple(permission.permission() for permission in binding.permissions) for binding in config.storage
    }
    policy = OpenFGAPolicy(
        api_url=config.openfga.api_url,
        store_id=config.openfga.store_id,
        model_id=config.openfga.model_id,
        token_file=config.openfga.token_file,
        principals=[p.model_dump() for p in config.openfga.principals],
        providers=[ProviderIdentity(binding.name, storage_permissions[binding.name]) for binding in config.storage],
        ca_bundle=config.ca_bundle,
        secrets=[binding.binding(config.bao_address) for binding in config.secrets],
    )
    store = GrantStore(config.database.connect(), policy.authorities, policy)
    broker = OpenBaoClient(config.bao_address, config.broker_token_file, ca_bundle=config.ca_bundle)
    signer = OpenBaoTransitSigner(broker, issuer=config.issuer, key_version=config.signing_key_version)
    audiences = frozenset(service for service, _action in ACTIONS)
    verifier = OpenBaoTransitVerifier(signer, audiences=audiences)
    authority = GrantAuthService(
        store,
        OpenBaoIdentityVerifier(
            issuer=config.bao_issuer,
            audience="marin-auth",
            subjects=config.subjects,
            ca_file=config.ca_bundle if isinstance(config.ca_bundle, str) else None,
        ),
        signer,
        verifier,
        max_token_ttl=config.max_token_ttl,
        refresh_audiences=REFRESH_AUDIENCES,
    )
    bindings = {
        binding.name: OpenBaoStorageIssuer(
            OpenBaoClient(config.bao_address, binding.token_file, ca_bundle=config.ca_bundle),
            role=binding.role,
            issuer=config.bao_issuer,
            subject=binding.subject,
            permissions=storage_permissions[binding.name],
        )
        for binding in config.storage
    }

    secret_readers = {
        binding.name: OpenBaoSecretReader(
            binding.binding(config.bao_address),
            OpenBaoClient(config.bao_address, binding.token_file, ca_bundle=config.ca_bundle),
        )
        for binding in config.secrets
    }

    async def authorization_unavailable(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"error": "Authorization is unavailable"}, status_code=503)

    app = create_app(authority)
    if config.iris is not None:
        mount_iris_attempts(app, IrisAttemptBroker(authority, config.iris, auth_url=config.issuer))
    app.add_exception_handler(AuthorizationUnavailable, authorization_unavailable)
    app.add_exception_handler(SQLAlchemyError, authorization_unavailable)
    serving = False
    reconciler: asyncio.Task[None] | None = None
    session = requests.Session()
    session.verify = config.ca_bundle
    launcher = None
    if config.loom is not None:
        launcher = LoomLauncher(
            store.database,
            base_url=config.loom.base_url,
            observer_tokens=LoomObserverTokenProvider(config.loom.observer_token_file),
            grants=authority,
            session=session,
        )

    def storage_token(request: Request) -> JSONResponse:
        token = extract_bearer_token(dict(request.headers))
        name = request.path_params["binding"]
        binding = bindings.get(name)
        if token is None or binding is None:
            raise GrantDenied("Unknown storage binding or missing grant credential")
        grant = None
        for permission in binding.permissions:
            for action in permission.actions:
                grant = authority.authorize(token, audience="cw_s3", action=action, resource=permission.resource)
        assert grant is not None
        if not policy.allows_provider(store.authority_principal(grant.id), name):
            raise GrantDenied("Standing policy does not allow this provider identity")
        result = binding.issue(grant)
        store.active(grant.id)
        return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    def read_secret(request: Request) -> JSONResponse:
        token = extract_bearer_token(dict(request.headers))
        reader = secret_readers.get(request.path_params["binding"])
        if token is None or reader is None:
            raise GrantDenied("Unknown secret binding or missing grant credential")
        authority.authorize(token, audience="openbao", action="secret.read", resource=reader.binding.resource)
        try:
            data = reader.read()
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Secret provider is unavailable"}, status_code=503)
        authority.authorize(token, audience="openbao", action="secret.read", resource=reader.binding.resource)
        return JSONResponse({"data": data}, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    def public_keys(_request: Request) -> JSONResponse:
        return JSONResponse(signer.public_jwks())

    async def health(_request: Request) -> JSONResponse:
        ready = serving and (launcher is None or (reconciler is not None and not reconciler.done()))
        return JSONResponse({"ready": ready, "policy_revision": policy.revision}, status_code=200 if ready else 503)

    async def launch_loom(request: Request) -> JSONResponse:
        token = extract_bearer_token(dict(request.headers))
        if token is None or launcher is None:
            raise GrantDenied("Loom launch is unavailable or unauthenticated")
        payload = LoomLaunchRequest.model_validate(await request.json())
        run = await run_in_threadpool(
            launcher.launch, token, template=payload.template, title=payload.title, goal=payload.goal
        )
        return JSONResponse(asdict(run), status_code=201)

    async def loom_credentials(request: Request) -> JSONResponse:
        token = extract_bearer_token(dict(request.headers))
        if token is None or launcher is None or config.loom is None:
            raise GrantDenied("Loom lifecycle integration and a grant access token are required")
        callback = request.headers.get("X-Marin-Loom-Callback", "")
        if not callback:
            raise GrantDenied("Trusted Loom callback authentication is required")
        try:
            expected_callback = config.loom.callback_token_file.read_text().strip()
        except OSError:
            return JSONResponse({"error": "Loom callback authentication is unavailable"}, status_code=503)
        if not expected_callback or not hmac.compare_digest(callback.encode(), expected_callback.encode()):
            raise GrantDenied("Trusted Loom callback authentication is required")
        payload = LoomCredentialRequest.model_validate(await request.json())
        grant = await run_in_threadpool(
            authority.authorize,
            token,
            audience="loom",
            action="session.create",
            resource=payload.resource,
        )
        try:
            await run_in_threadpool(
                launcher.observe, grant, payload.resource.strip("/"), payload.run_id, payload.session_id
            )
        except requests.RequestException:
            return JSONResponse({"error": "Loom launch verification is unavailable"}, status_code=503)
        runtime_permissions = tuple(item for item in grant.permissions if item.service in LOOM_RUNTIME_AUDIENCES)
        storage, iris_token = None, None
        if runtime_permissions:
            runtime = await run_in_threadpool(
                store.create,
                grant.actor,
                actor=f"loom:{grant.id}",
                permissions=runtime_permissions,
                expires_at=grant.expires_at,
                parent_id=grant.id,
                idempotency_key=f"loom-runtime:{grant.id}",
            )
            if any(item.service == "cw_s3" for item in runtime_permissions):
                storage = await run_in_threadpool(authority.storage_token, runtime)
            if any(item.service == "iris" for item in runtime_permissions):
                result = await run_in_threadpool(authority.mint_for, runtime, audience="iris")
                iris_token = result["access_token"]
        await run_in_threadpool(store.active, grant.id)
        return JSONResponse(
            {
                "grant_id": grant.id,
                "actor": grant.actor,
                "owner": grant.owner,
                "expires_at": grant.expires_at,
                "policy_revision": grant.policy_revision,
                "auth_url": config.issuer,
                "storage_token": storage,
                "iris_token": iris_token,
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    async def reconcile_loom() -> None:
        assert launcher is not None
        while True:
            try:
                await run_in_threadpool(launcher.reconcile)
            except requests.RequestException:
                logging.getLogger(__name__).exception("Loom lifecycle reconciliation failed; retrying in five seconds")
            await asyncio.sleep(5)

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        nonlocal serving, reconciler
        try:
            if launcher is None:
                serving = True
                yield
            else:
                async with asyncio.TaskGroup() as tasks:
                    reconciler = tasks.create_task(reconcile_loom())
                    serving = True
                    try:
                        yield
                    finally:
                        serving = False
                        reconciler.cancel()
        finally:
            serving = False
            session.close()
            store.close()

    app.router.routes.extend(
        [
            Route("/v1/storage/{binding}/token", storage_token, methods=["POST"]),
            Route("/v1/secrets/{binding}", read_secret, methods=["POST"]),
            Route("/.well-known/jwks.json", public_keys, methods=["GET"]),
            Route("/healthz", health, methods=["GET"]),
            Route("/v1/loom/launch", launch_loom, methods=["POST"]),
            Route("/v1/loom/credentials", loom_credentials, methods=["POST"]),
        ]
    )
    app.router.lifespan_context = lifespan
    return app, authority


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    args = parser.parse_args()
    config = ServiceConfig.model_validate(yaml.safe_load(args.config.read_text()))
    app, _authority = create_service(config)
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("both TLS certificate and key are required together")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=False,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
