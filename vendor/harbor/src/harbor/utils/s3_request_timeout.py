"""Whole-request deadlines for archive-enabled jobs using S3."""

import fsspec.config
from rigging.filesystem.s3_compat import TotalDeadlineAIOHTTPSession


def configure_s3_request_timeout() -> None:
    """Set the process-wide S3 session default before creating job filesystems."""
    config = fsspec.config.conf.setdefault("s3", {})
    config.setdefault("config_kwargs", {}).setdefault(
        "http_session_cls", TotalDeadlineAIOHTTPSession
    )
