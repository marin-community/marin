from datetime import timedelta

import pytimeparse


def parse_timedelta(td_str) -> timedelta:
    td = timedelta(seconds=pytimeparse.parse(td_str))
    if td.total_seconds() < 0:
        raise ValueError("Cannot encode negative timedelta")  # not worth the trouble

    return td


def encode_timedelta(td: timedelta) -> str:
    """Encodes a timedelta as a string that can be parsed by parse_timedelta/pytimeparse."""
    out = ""
    if td.total_seconds() < 0:
        raise ValueError("Cannot encode negative timedelta")  # not worth the trouble

    if td.days:
        out += f"{td.days}d"

    seconds: float = td.seconds

    if seconds > 3600:
        hours = seconds // 3600
        seconds -= hours * 3600
        out += f"{hours}h"
    if seconds > 60:
        minutes = seconds // 60
        seconds -= minutes * 60
        out += f"{minutes}m"

    if td.microseconds:
        seconds += td.microseconds / 1e6

    if seconds:
        out += f"{seconds}s"

    assert parse_timedelta(out) == td, f"Failed to encode {td} as {out}"
    return out
