"""Validate the dedicated origin shared by the registry and sandbox routing."""

from urllib.parse import urlsplit


def public_origin(value: str) -> str:
    """Require a dedicated HTTPS origin without credentials or URL routing fields."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https" or not parsed.hostname or parsed.username
        or parsed.password or parsed.path not in ("", "/") or parsed.query
        or parsed.fragment or parsed.port not in (None, 443)
    ):
        raise ValueError("Package gateway must be a dedicated HTTPS origin")
    return f"https://{parsed.hostname}"

