"""Regression: the fsspec ``s3`` config injected via ``FSSPEC_S3`` (e.g. the CoreWeave/iris
object store) carries ``endpoint_url`` at the TOP level. Old s3fs forwards unrecognised
top-level kwargs to ``botocore.session.Session(**kwargs)``, which rejects ``endpoint_url``,
so the first ``UPath("s3://...").exists()`` dies with::

    TypeError: Session.__init__() got an unexpected keyword argument 'endpoint_url'

harbor relocates a top-level ``endpoint_url`` into ``client_kwargs`` at import so bare-UPath
s3 access works on any s3fs version.
"""

import fsspec.config
import pytest

from harbor import _route_s3_endpoint_url_to_client_kwargs


@pytest.fixture(autouse=True)
def _restore_s3_conf():
    """Snapshot/restore fsspec's global s3 config so tests don't leak into each other."""
    prev = fsspec.config.conf.get("s3")
    try:
        yield
    finally:
        if prev is None:
            fsspec.config.conf.pop("s3", None)
        else:
            fsspec.config.conf["s3"] = prev


def _route(conf: dict) -> dict:
    fsspec.config.conf["s3"] = conf
    _route_s3_endpoint_url_to_client_kwargs()
    return fsspec.config.conf["s3"]


@pytest.mark.unit
class TestS3EndpointUrlRouting:
    def test_top_level_endpoint_url_moved_into_client_kwargs(self):
        # The exact shape FSSPEC_S3 injects on the CoreWeave/iris store.
        out = _route(
            {
                "endpoint_url": "http://cwlota.com",
                "client_kwargs": {"region_name": "auto"},
            }
        )
        assert "endpoint_url" not in out, (
            "a top-level endpoint_url would reach botocore.session.Session"
        )
        assert out["client_kwargs"]["endpoint_url"] == "http://cwlota.com"
        assert (
            out["client_kwargs"]["region_name"] == "auto"
        )  # existing client_kwargs preserved

    def test_creates_client_kwargs_when_absent(self):
        out = _route({"endpoint_url": "http://cwlota.com"})
        assert "endpoint_url" not in out
        assert out["client_kwargs"]["endpoint_url"] == "http://cwlota.com"

    def test_explicit_client_kwargs_endpoint_url_wins(self):
        out = _route(
            {
                "endpoint_url": "http://top",
                "client_kwargs": {"endpoint_url": "http://explicit"},
            }
        )
        assert "endpoint_url" not in out  # stray top-level key dropped
        assert out["client_kwargs"]["endpoint_url"] == "http://explicit"

    def test_noop_without_top_level_endpoint_url(self):
        out = _route({"client_kwargs": {"region_name": "auto"}})
        assert out == {"client_kwargs": {"region_name": "auto"}}

    def test_leftover_top_level_kwargs_are_botocore_session_safe(self):
        """Reproduce the crash directly: mimic old s3fs handing the remaining top-level kwargs to
        botocore.session.Session. Before the fix this raises on endpoint_url; after, it's clean."""
        import botocore.session

        out = _route(
            {
                "endpoint_url": "http://cwlota.com",
                "client_kwargs": {"region_name": "auto"},
            }
        )
        session_kwargs = {
            k: v for k, v in out.items() if k not in ("client_kwargs", "config_kwargs")
        }
        botocore.session.Session(**session_kwargs)  # must NOT raise TypeError
