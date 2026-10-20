"""Version matching shared by the Open-MOPD image and task preflights."""

from packaging.version import Version


def versions_match(expected: str, actual: str) -> bool:
    """Accept a matching local build when the manifest names a public release."""
    expected_version = Version(expected)
    actual_version = Version(actual)
    if expected_version.local is not None:
        return actual_version == expected_version
    return actual_version.public == expected_version.public
