"""Read-only PyPI/npm gateway. Run outside the agent's network trust boundary."""

import argparse
import base64
import hashlib
import json
import logging
import re
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version
from starlette.types import Receive, Scope, Send

from harbor.security.origin import public_origin

logger = logging.getLogger(__name__)
OSV_QUERY = "https://api.osv.dev/v1/query"
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
PYPI_MINIMUM = {"brotli": Version("1.2.1"), "brotlicffi": Version("1.2.1.0")}
NPM_NAME = re.compile(r"(?:@[a-z0-9._~-]+/)?[a-z0-9._~-]+\Z")


@dataclass(frozen=True)
class Artifact:
    url: str
    algorithm: str
    digest: bytes
    filename: str



class ArtifactResponse(FileResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            Path(self.path).unlink(missing_ok=True)


def local_block(ecosystem: str, name: str, version: str) -> str | None:
    if ecosystem == "PyPI":
        name = canonicalize_name(name)
        if name == "litellm":
            return "LiteLLM is blocked by Marin policy"
        if name in PYPI_MINIMUM and Version(version) < PYPI_MINIMUM[name]:
            return f"CVE-2026-111497: {name} requires >= {PYPI_MINIMUM[name]}"
    return None


def require_upstream(url: str, host: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != host or parsed.fragment:
        raise HTTPException(502, "Registry supplied an unapproved artifact origin")


async def read_json(client: httpx.AsyncClient, url: str, **kwargs) -> dict:
    async with client.stream(**{"method": "GET", "url": url, **kwargs}) as response:
        if response.status_code == 404:
            raise HTTPException(404, "Package or version not found")
        if response.status_code != 200:
            raise HTTPException(503, "Registry or advisory service unavailable")
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > MAX_METADATA_BYTES:
                raise HTTPException(502, "Upstream metadata exceeds size limit")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise HTTPException(502, "Invalid upstream metadata")
    return result


async def approve(client: httpx.AsyncClient, ecosystem: str, name: str, version: str) -> None:
    reason = local_block(ecosystem, name, version)
    if reason:
        raise HTTPException(403, reason)
    # Query on each artifact request; neither a cache hit nor a previous approval
    # exempts a version from newly published advisories.
    result = await read_json(
        client, OSV_QUERY, method="POST",
        json={"package": {"ecosystem": ecosystem, "name": name}, "version": version},
    )
    vulnerabilities = result.get("vulns", [])
    if not isinstance(vulnerabilities, list) or result.get("next_page_token"):
        raise HTTPException(503, "Incomplete advisory response")
    active = [item["id"] for item in vulnerabilities if not item.get("withdrawn")]
    if active:
        logger.warning("Package denied: %s %s %s %s", ecosystem, name, version, active)
        raise HTTPException(403, {"package": name, "version": version, "advisories": active})


async def serve_artifact(client: httpx.AsyncClient, artifact: Artifact) -> FileResponse:
    # Buffer to a bounded temporary file and verify before returning any bytes.
    # Direct upstream redirects are never returned to the sandbox.
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as output:
            path = Path(output.name)
            digest = hashlib.new(artifact.algorithm)
            size = 0
            async with client.stream("GET", artifact.url) as response:
                if response.status_code != 200:
                    raise HTTPException(503, "Artifact download failed")
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_ARTIFACT_BYTES:
                        raise HTTPException(502, "Artifact exceeds size limit")
                    digest.update(chunk)
                    output.write(chunk)
            if digest.digest() != artifact.digest:
                raise HTTPException(502, "Artifact digest does not match registry metadata")
        return ArtifactResponse(
            path, filename=artifact.filename,
            headers={"Cache-Control": "no-store"},
        )
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def python_file(filename: str) -> tuple[str, str]:
    if filename.endswith(".whl"):
        name, version, _, _ = parse_wheel_filename(filename)
    else:
        name, version = parse_sdist_filename(filename)
    return str(name), str(version)


def create_app(origin: str, client: httpx.AsyncClient) -> FastAPI:
    """Build a registry with an explicit public origin and caller-owned HTTP client."""
    origin = public_origin(origin)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(httpx.HTTPError)
    async def upstream_error(request, error):
        return JSONResponse({"detail": "Upstream request failed; download denied"}, status_code=503)

    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    @app.exception_handler(TypeError)
    async def metadata_error(request, error):
        return JSONResponse({"detail": "Invalid registry or advisory metadata; download denied"}, status_code=502)

    async def pypi_files(name: str) -> dict:
        name = canonicalize_name(name, validate=True)
        if name == "litellm":
            raise HTTPException(403, "LiteLLM is blocked by Marin policy")
        return await read_json(
            client, f"https://pypi.org/simple/{quote(name, safe='')}/",
            headers={"Accept": "application/vnd.pypi.simple.v1+json"},
        )

    @app.get("/pypi/simple/{name}/")
    async def pypi_index(name: str):
        name = canonicalize_name(name, validate=True)
        document = await pypi_files(name)
        files = []
        for item in document["files"]:
            filename = item["filename"]
            if not filename.endswith((".whl", ".tar.gz", ".zip")):
                continue
            file_name, version = python_file(filename)
            if file_name != name or local_block("PyPI", name, version):
                continue
            files.append({
                "filename": filename,
                "url": f"{origin}/artifacts/pypi/{quote(filename, safe='')}?{urlencode({'name': name})}",
                "hashes": {"sha256": item["hashes"]["sha256"]},
                "requires-python": item.get("requires-python"),
                "yanked": item.get("yanked", False),
            })
        return JSONResponse(
            {"meta": {"api-version": "1.0"}, "name": name, "files": files},
            media_type="application/vnd.pypi.simple.v1+json",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/artifacts/pypi/{filename}")
    async def pypi_artifact(filename: str, name: str):
        name = canonicalize_name(name, validate=True)
        file_name, version = python_file(filename)
        if file_name != name:
            raise HTTPException(400, "Package name does not match filename")
        await approve(client, "PyPI", name, version)
        document = await pypi_files(name)
        for item in document["files"]:
            if item["filename"] == filename:
                require_upstream(item["url"], "files.pythonhosted.org")
                artifact = Artifact(item["url"], "sha256", bytes.fromhex(item["hashes"]["sha256"]), filename)
                return await serve_artifact(client, artifact)
        raise HTTPException(404, "Artifact absent from registry metadata")

    def npm_name(name: str) -> str:
        if not NPM_NAME.fullmatch(name) or name.split("/")[-1].startswith((".", "_")):
            raise HTTPException(400, "Invalid npm package name")
        return name

    async def npm_document(name: str) -> dict:
        return await read_json(client, f"https://registry.npmjs.org/{quote(npm_name(name), safe='')}")

    @app.get("/artifacts/npm")
    async def npm_artifact(name: str, version: str):
        name = npm_name(name)
        await approve(client, "npm", name, version)
        document = await npm_document(name)
        package = document["versions"].get(version)
        if package is None:
            raise HTTPException(404, "Version absent from registry metadata")
        if package["name"] != name or package["version"] != version:
            raise HTTPException(502, "Registry package identity mismatch")
        dist = package["dist"]
        require_upstream(dist["tarball"], "registry.npmjs.org")
        integrity = dist["integrity"].split()
        sha512 = next((part[7:] for part in integrity if part.startswith("sha512-")), None)
        if sha512 is None:
            raise HTTPException(502, "npm artifact has no SHA-512 integrity value")
        artifact = Artifact(dist["tarball"], "sha512", base64.b64decode(sha512, validate=True), "package.tgz")
        return await serve_artifact(client, artifact)

    @app.get("/npm/{name:path}")
    async def npm_index(name: str):
        document = await npm_document(name)
        for version, package in document["versions"].items():
            package["dist"]["tarball"] = f"{origin}/artifacts/npm?{urlencode({'name': name, 'version': version})}"
        # Only the registry protocol fields are exposed; README and arbitrary
        # upstream fields do not create a general-purpose download relay.
        return JSONResponse(
            {"name": name, "versions": document["versions"], "dist-tags": document.get("dist-tags", {})},
            headers={"Cache-Control": "no-store"},
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-origin", required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    client = httpx.AsyncClient(timeout=60, follow_redirects=False, trust_env=False, headers={"Accept-Encoding": "identity"})
    app = create_app(args.public_origin, client)

    @asynccontextmanager
    async def lifespan(app):
        async with client:
            yield

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host="0.0.0.0", port=args.port, proxy_headers=False, limit_concurrency=8)


if __name__ == "__main__":
    main()
