"""Exercise package protocol responses without fetching or executing real packages."""

import base64
import hashlib
import json

import httpx
import pytest

from harbor.security.gateway import create_app

ORIGIN = "https://packages.example.test"
WHEEL = "demo_pkg-1.0-py3-none-any.whl"
PAYLOAD = b"inert artifact fixture"


@pytest.fixture
def registry():
    calls = []
    settings = {"advisories": {}, "artifact_status": 200, "artifact": PAYLOAD, "artifact_host": "files.pythonhosted.org"}

    def handle(request):
        calls.append((request.method, str(request.url)))
        if request.url.host == "api.osv.dev":
            payload = json.loads(request.content)
            assert payload["package"]["ecosystem"] in {"PyPI", "npm"}
            value = settings["advisories"]
            return httpx.Response(
                value if isinstance(value, int) else 200, json=None if isinstance(value, int) else value
            )
        if request.url.host == "pypi.org":
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "filename": WHEEL,
                            "url": f'https://{settings["artifact_host"]}/{WHEEL}',
                            "hashes": {"sha256": hashlib.sha256(PAYLOAD).hexdigest()},
                        }
                    ]
                },
            )
        if request.url.host == "registry.npmjs.org" and not request.url.path.endswith(".tgz"):
            return httpx.Response(
                200,
                json={
                    "name": "@scope/demo",
                    "dist-tags": {"latest": "1.0.0"},
                    "versions": {
                        "1.0.0": {
                            "name": "@scope/demo",
                            "version": "1.0.0",
                            "dependencies": {"other": "^1"},
                            "dist": {
                                "tarball": "https://registry.npmjs.org/@scope/demo/-/demo-1.0.0.tgz",
                                "integrity": "sha512-" + base64.b64encode(hashlib.sha512(PAYLOAD).digest()).decode(),
                            },
                        }
                    },
                },
            )
        return httpx.Response(
            settings["artifact_status"],
            content=settings["artifact"],
            headers={"Location": "https://evil.example/payload"},
        )

    return settings, calls, httpx.MockTransport(handle)


async def test_python_index_rewrites_download_and_checks_before_serving(registry):
    settings, calls, transport = registry
    async with httpx.AsyncClient(transport=transport) as upstream:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(ORIGIN, upstream)), base_url=ORIGIN
        ) as client:
            index = await client.get("/pypi/simple/demo-pkg/")
            url = index.json()["files"][0]["url"]
            assert url.startswith(ORIGIN + "/artifacts/")
            response = await client.get(url)
            assert response.status_code == 200
            assert response.content == PAYLOAD
            assert response.headers["cache-control"] == "no-store"
            assert next(i for i, (_, url) in enumerate(calls) if "api.osv.dev" in url) < next(
                i for i, (_, url) in enumerate(calls) if "files.pythonhosted.org" in url
            )


@pytest.mark.parametrize(
    "filename,name",
    [
        ("brotli-1.2.0.tar.gz", "brotli"),
        ("brotlicffi-1.2.0.2.tar.gz", "brotlicffi"),
        ("litellm-1.107.0.tar.gz", "litellm"),
    ],
)
async def test_known_block_cannot_be_bypassed_by_direct_artifact_url(registry, filename, name):
    _, calls, transport = registry
    async with httpx.AsyncClient(transport=transport) as upstream:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(ORIGIN, upstream)), base_url=ORIGIN
        ) as client:
            response = await client.get(f"/artifacts/pypi/{filename}", params={"name": name})
            assert response.status_code == 403
            assert not calls  # No request reaches the public registry or artifact server.


@pytest.mark.parametrize(
    "advisories", [{"vulns": [{"id": "OSV-fixture"}]}, 503, {"next_page_token": "more"}, {"vulns": None}]
)
async def test_advisory_failure_or_finding_prevents_artifact_fetch(registry, advisories):
    settings, calls, transport = registry
    settings["advisories"] = advisories
    async with httpx.AsyncClient(transport=transport) as upstream:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(ORIGIN, upstream)), base_url=ORIGIN
        ) as client:
            response = await client.get(f"/artifacts/pypi/{WHEEL}", params={"name": "demo-pkg"})
            assert response.status_code in {403, 503}
            assert all("files.pythonhosted.org" not in url for _, url in calls)


@pytest.mark.parametrize(
    "changes", [{"artifact": b"tampered"}, {"artifact_status": 302}, {"artifact_host": "127.0.0.1"}]
)
async def test_tampered_redirected_or_unapproved_artifacts_are_not_served(registry, changes):
    settings, calls, transport = registry
    settings.update(changes)
    async with httpx.AsyncClient(transport=transport) as upstream:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(ORIGIN, upstream)), base_url=ORIGIN
        ) as client:
            response = await client.get(f"/artifacts/pypi/{WHEEL}", params={"name": "demo-pkg"})
            assert response.status_code in {502, 503}
            assert response.content != settings["artifact"]
            assert all("127.0.0.1" not in url and "evil.example" not in url for _, url in calls)


async def test_scoped_npm_package_rechecks_new_advisories_on_repeat_download(registry):
    settings, calls, transport = registry
    async with httpx.AsyncClient(transport=transport) as upstream:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(ORIGIN, upstream)), base_url=ORIGIN
        ) as client:
            index = (await client.get("/npm/@scope%2Fdemo")).json()
            package = index["versions"]["1.0.0"]
            assert package["dependencies"] == {"other": "^1"}
            url = package["dist"]["tarball"]
            assert url.startswith(ORIGIN + "/artifacts/")
            assert (await client.get(url)).content == PAYLOAD
            settings["advisories"] = {"vulns": [{"id": "new-advisory"}]}
            response = await client.get(url)
            assert response.status_code == 403
            assert response.json()["detail"]["advisories"] == ["new-advisory"]
