"""Task #343 probes (not for merge): what main does today, and what a minimal change would do."""

import shutil
import uuid
from pathlib import Path

from marina.applets import package_applet
from marina.server import MarinaConfig, MarinaSurface, create_app, named_applet_hosts
from sqlalchemy import text
from starlette.testclient import TestClient

from test_applets import DEMO_APPLET, applet_client_and_store

PUBLIC = "https://public.applets.example"


def public_config(tmp_path, store, **extra):
    return MarinaConfig(
        data_root=str(tmp_path / "data"),
        iap_audience=None,
        database=store.database,
        public_applet_origin=PUBLIC,
        surface=MarinaSurface.PUBLIC_APPLETS,
        **extra,
    )


def named_package(tmp_path: Path, name: str) -> bytes:
    source = tmp_path / f"src-{name}-{uuid.uuid4().hex[:6]}"
    shutil.copytree(DEMO_APPLET, source)
    manifest = source / "applet.toml"
    manifest.write_text(f'name = "{name}"\n' + manifest.read_text())
    return package_applet(source)


def test_probe_1_named_public_is_rejected_today(tmp_path, database_url):
    publisher, _store = applet_client_and_store(tmp_path, database_url)
    name = f"probe{uuid.uuid4().hex[:8]}"
    applet_id = str(uuid.uuid4())
    rejected = publisher.post(
        f"/api/marina/applets/{applet_id}", params={"base_version": 0, "mode": "public"}, content=named_package(tmp_path, name)
    )
    print("PROBE1 named release --mode public:", rejected.status_code, rejected.json())
    private = publisher.post(f"/api/marina/applets/{applet_id}", params={"base_version": 0}, content=named_package(tmp_path, name))
    print("PROBE1 named release private:", private.status_code, private.json())
    mode = publisher.put(f"/api/marina/applets/{applet_id}/mode", json={"mode": "public", "base_version": 1})
    print("PROBE1 named mode public:", mode.status_code, mode.json())
    assert rejected.status_code == 400 and mode.status_code == 400


def test_probe_2_named_app_forced_public_in_db(tmp_path, database_url):
    """If only the two store guards were removed, what would the public service serve?"""
    publisher, store = applet_client_and_store(tmp_path, database_url)
    name = f"probe{uuid.uuid4().hex[:8]}"
    applet_id = str(uuid.uuid4())
    assert publisher.post(f"/api/marina/applets/{applet_id}", params={"base_version": 0}, content=named_package(tmp_path, name)).status_code == 201
    with store.engine.begin() as connection:
        connection.execute(text("UPDATE marina.applets SET mode = 'public' WHERE id = :id"), {"id": applet_id})
    public = TestClient(create_app(public_config(tmp_path, store)), base_url=PUBLIC, client=("10.0.0.7", 40000))
    results = {
        "GET /<name>/": public.get(f"/{name}/").status_code,
        "GET /<name>/api/problems": public.get(f"/{name}/api/problems").status_code,
        "GET /a/<uuid>/ (redirect)": public.get(f"/a/{applet_id}/", follow_redirects=False).status_code,
        "GET /a/<uuid>/v/1/": public.get(f"/a/{applet_id}/v/1/").status_code,
        "GET /a/<uuid>/v/1/api/problems": public.get(f"/a/{applet_id}/v/1/api/problems").status_code,
        "GET /a/<uuid>/v/1/api/identity": public.get(f"/a/{applet_id}/v/1/api/identity").json(),
        "POST /a/<uuid>/query": public.post(f"/a/{applet_id}/query", json={"sql": "SELECT 1"}).status_code,
        "POST /<name>/api/problems": public.post(f"/{name}/api/problems").status_code,
    }
    for key, value in results.items():
        print("PROBE2", key, "->", value)


def test_probe_3_public_surface_ignores_bound_hosts_today(tmp_path, database_url):
    publisher, store = applet_client_and_store(tmp_path, database_url)
    published = publisher.post("/api/marina/applets", params={"mode": "public"}, content=package_applet(DEMO_APPLET)).json()
    host = f"tree{uuid.uuid4().hex[:6]}.example"
    store.bind_host(host, uuid.UUID(published["id"]))
    app = create_app(public_config(tmp_path, store, applet_domains=frozenset({host})))
    vanity = TestClient(app, base_url=f"https://{host}", client=("10.0.0.7", 40000))
    print("PROBE3 today: GET / on bound host, public surface ->", vanity.get("/", follow_redirects=False).status_code)


def test_probe_4_prototype_public_vanity_host(tmp_path, database_url):
    """Prototype: wrap the public surface in the existing named-host router (the change create_app would make)."""
    publisher, store = applet_client_and_store(tmp_path, database_url)
    package = package_applet(DEMO_APPLET)
    public_app = publisher.post("/api/marina/applets", params={"mode": "public"}, content=package).json()
    private_app = publisher.post("/api/marina/applets", content=package).json()
    other_public = publisher.post("/api/marina/applets", params={"mode": "public"}, content=package).json()
    suffix = uuid.uuid4().hex[:6]
    public_host, private_host = f"tree{suffix}.example", f"secret{suffix}.example"
    store.bind_host(public_host, uuid.UUID(public_app["id"]))
    store.bind_host(private_host, uuid.UUID(private_app["id"]))
    domains = frozenset({public_host, private_host})
    app = named_applet_hosts(create_app(public_config(tmp_path, store)), store, domains)
    vanity = TestClient(app, base_url=f"https://{public_host}", client=("10.0.0.7", 40000))
    secret = TestClient(app, base_url=f"https://{private_host}", client=("10.0.0.7", 40000))

    root = vanity.get("/", follow_redirects=False)
    page = vanity.get("/")
    out = {
        "public host GET / ": (root.status_code, root.headers.get("location")),
        "public host page": (page.status_code, str(page.url), page.headers.get("cache-control")),
        "public host asset": vanity.get("/v/1/app.js").status_code,
        "public host HEAD asset": vanity.head("/v/1/app.js").status_code,
        "public host API read": vanity.get("/v/1/api/problems").status_code,
        "public host identity": vanity.get("/v/1/api/identity").json(),
        "public host POST api": vanity.post("/v/1/api/problems").status_code,
        "public host POST query": vanity.post("/v/1/query", json={"sql": "SELECT 1"}).status_code,
        "public host /api/marina/apps": vanity.get("/api/marina/apps").status_code,
        "public host /a/<other public uuid>/": vanity.get(f"/a/{other_public['id']}/v/1/").status_code,
        "public host /healthz": vanity.get("/healthz").status_code,
        "PRIVATE app host GET /": secret.get("/", follow_redirects=False).status_code,
        "PRIVATE app host asset": secret.get("/v/1/app.js").status_code,
        "PRIVATE app host api": secret.get("/v/1/api/problems").status_code,
    }
    for key, value in out.items():
        print("PROBE4", key, "->", value)

    # Revocation and release propagate without a deploy.
    assert publisher.post(f"/api/marina/applets/{public_app['id']}", params={"base_version": 1}, content=package).status_code == 201
    print("PROBE4 after release: GET / ->", vanity.get("/", follow_redirects=False).headers.get("location"))
    assert publisher.put(f"/api/marina/applets/{public_app['id']}/mode", json={"mode": "private", "base_version": 2}).status_code == 200
    print("PROBE4 after mode=private: GET / ->", vanity.get("/", follow_redirects=False).status_code, " asset ->", vanity.get("/v/2/app.js").status_code)
    store.unbind_host(public_host)
    print("PROBE4 after unbind: GET / ->", vanity.get("/", follow_redirects=False).status_code)


def test_probe_5_prototype_named_public_app(tmp_path, database_url):
    """With the prototype patch: a named app marked public is readable at /<name>/ on the public service."""
    publisher, store = applet_client_and_store(tmp_path, database_url)
    suffix = uuid.uuid4().hex[:8]
    pub_name, priv_name = f"pub{suffix}", f"priv{suffix}"
    pub_id, priv_id = str(uuid.uuid4()), str(uuid.uuid4())
    released = publisher.post(
        f"/api/marina/applets/{pub_id}", params={"base_version": 0, "mode": "public"}, content=named_package(tmp_path, pub_name)
    )
    print("PROBE5 named release --mode public:", released.status_code, released.json().get("mode"))
    assert publisher.post(f"/api/marina/applets/{priv_id}", params={"base_version": 0}, content=named_package(tmp_path, priv_name)).status_code == 201
    (tmp_path / "data" / pub_name).mkdir(parents=True)
    (tmp_path / "data" / pub_name / "x.json").write_text('{"k": 1}')

    public = TestClient(create_app(public_config(tmp_path, store)), base_url=PUBLIC, client=("10.0.0.7", 40000))
    private = TestClient(
        create_app(MarinaConfig(data_root=str(tmp_path / "data"), iap_audience="aud", database=store.database)),
        client=("10.0.0.7", 40000),
    )
    page = public.get(f"/{pub_name}/")
    out = {
        "public svc GET /<public name> (no slash)": public.get(f"/{pub_name}", follow_redirects=False).status_code,
        "public svc GET /<public name>/": (page.status_code, page.headers.get("cache-control")),
        "public svc GET /<public name>/app.js": public.get(f"/{pub_name}/app.js").status_code,
        "public svc GET /<public name>/some/client/route (html)": public.get(f"/{pub_name}/runs/abc", headers={"accept": "text/html"}).status_code,
        "public svc GET /<public name>/api/problems": public.get(f"/{pub_name}/api/problems").status_code,
        "public svc GET /<public name>/api/identity": public.get(f"/{pub_name}/api/identity").json(),
        "public svc GET /<public name>/data/x.json": public.get(f"/{pub_name}/data/x.json").status_code,
        "public svc POST /<public name>/api/problems": public.post(f"/{pub_name}/api/problems").status_code,
        "public svc DELETE /<public name>/api/problems": public.delete(f"/{pub_name}/api/problems").status_code,
        "public svc GET /<PRIVATE name>/": public.get(f"/{priv_name}/").status_code,
        "public svc GET /<PRIVATE name>/api/problems": public.get(f"/{priv_name}/api/problems").status_code,
        "public svc GET /<PRIVATE name>/data/x.json": public.get(f"/{priv_name}/data/x.json").status_code,
        "public svc GET /a/<private uuid>/v/1/": public.get(f"/a/{priv_id}/v/1/").status_code,
        "public svc GET /a/<public uuid>/nope (was 401)": public.get(f"/a/{pub_id}/nope").status_code,
        "public svc GET /": public.get("/").status_code,
        "public svc GET /api/marina/apps": public.get("/api/marina/apps").status_code,
        "public svc GET /api/marina/me": public.get("/api/marina/me").status_code,
        "public svc GET /api/marina/mcp/": public.get("/api/marina/mcp/").status_code,
        "public svc POST /api/marina/applets": public.post("/api/marina/applets").status_code,
        "IAP svc, no login, GET /<public name>/": private.get(f"/{pub_name}/").status_code,
        "IAP svc, no login, GET /<public name>/api/problems": private.get(f"/{pub_name}/api/problems").status_code,
        "loopback publisher GET /<public name>/api/identity": publisher.get(f"/{pub_name}/api/identity").json(),
    }
    for key, value in out.items():
        print("PROBE5", key, "->", value)
    assert publisher.put(f"/api/marina/applets/{pub_id}/mode", json={"mode": "private", "base_version": 1}).status_code == 200
    print("PROBE5 after mode=private: public svc GET /<name>/ ->", public.get(f"/{pub_name}/").status_code, " api ->", public.get(f"/{pub_name}/api/problems").status_code)


def test_probe_6_prototype_public_host_via_create_app(tmp_path, database_url):
    """With the prototype patch: create_app itself routes bound hosts on the public service."""
    publisher, store = applet_client_and_store(tmp_path, database_url)
    published = publisher.post("/api/marina/applets", params={"mode": "public"}, content=package_applet(DEMO_APPLET)).json()
    host = f"tree{uuid.uuid4().hex[:6]}.example"
    store.bind_host(host, uuid.UUID(published["id"]))
    app = create_app(public_config(tmp_path, store, applet_domains=frozenset({host})))
    vanity = TestClient(app, base_url=f"https://{host}", client=("10.0.0.7", 40000))
    shared = TestClient(app, base_url=PUBLIC, client=("10.0.0.7", 40000))
    print("PROBE6 bound host GET / ->", vanity.get("/", follow_redirects=False).status_code, vanity.get("/").status_code)
    print("PROBE6 bound host GET /healthz ->", vanity.get("/healthz").status_code)
    print("PROBE6 shared public host still serves /a/<uuid>/ ->", shared.get(f"/a/{published['id']}/").status_code)
