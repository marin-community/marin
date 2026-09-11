# Publishing an Analysis Site

One-off analysis pages (dashboards, side-by-side comparisons, dossiers) often end up on personal
sites that disappear when the author moves on. Marin gives them a durable, public home: a page
published here gets a stable `https://storage.googleapis.com/marin-public/...` URL, is recorded as
an Artifact so it can be fetched from code, and is listed in a public discovery index.

!!! warning "The `marin-public` bucket is world-readable"
    Everything you publish is readable by anyone on the internet, and any handle is unauthenticated
    (there is no per-user ownership). **Never publish anything sensitive** — no private data, no
    sampled corpus text, no credentials.

## Publish

`source` is either a single `.html` file or a directory whose `index.html` is the entrypoint (a
multi-file site — HTML plus JS/CSS/data). Versions are calendar versions, `YYYY.MM.DD[.N]`.

=== "CLI"

    ```bash
    uv run scripts/ops/publish_site.py report.html \
        --user held --slug datakit-sidebyside --version 2026.07.01 \
        --title "DataKit side-by-side"
    # -> https://storage.googleapis.com/marin-public/held/datakit-sidebyside/2026.07.01/index.html
    ```

=== "Python"

    ```python
    from pathlib import Path

    from marin.publish.sites import publish_site

    site = publish_site(
        Path("report.html"),
        user="held", slug="datakit-sidebyside", version="2026.07.01",
        title="DataKit side-by-side", summary="cluster examples, side by side",
    )
    print(site.url)
    ```

The page lands at `gs://marin-public/<user>/<slug>/<version>/`, alongside an `.artifact.json` record.
A new version publishes to a new path, so old links keep working.

## Fetch from code

The address is a pure function of `(user, slug, version)`, so no registry lookup is needed:

```python
from marin.publish.sites import site_uri
from marin.execution.artifact import Artifact

site = Artifact.raw_load(site_uri("held", "datakit-sidebyside", "2026.07.01"))
print(site.path)          # gs://marin-public/held/datakit-sidebyside/2026.07.01/
print(site.record.config) # {"user", "slug", "version", "url", "title", "summary"}
```

`site_url(...)` and `site_uri(...)` are pure string builders if you only need the link or the path.

## Discovery

Every publish upserts an entry into `gs://marin-public/index.json`, a list of
`{name, version, url, title, summary}`. Read it to enumerate published sites; the curated list also
appears under [Published analysis sites](../reports/index.md).
