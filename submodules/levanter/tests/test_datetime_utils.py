from levanter.utils.datetime_utils import encode_timedelta, parse_timedelta


def test_encode_timedelta():
    # various time strings from the unit tests for pytimeparse
    # https://github.com/wroberts/pytimeparse/blob/master/pytimeparse/tests/testtimeparse.py
    # i skipped negative ones because they're not supported, and fractional things because really

    def ensure_roundtrip(td_str, expected_seconds):
        # we don't enforce that the output is the same as the input,
        # but we do enforce that it can be parsed to the same timedelta
        td = parse_timedelta(td_str)
        assert td.total_seconds() == expected_seconds
        assert parse_timedelta(encode_timedelta(td)) == td, f"Failed to roundtrip {td_str}: {encode_timedelta(td)}"

    ensure_roundtrip("1d", 86400)
    ensure_roundtrip("+32 m 1 s", 1921)
    ensure_roundtrip("+ 32 m 1 s", 1921)
    ensure_roundtrip("32m", 1920)
    ensure_roundtrip("+32m", 1920)
    ensure_roundtrip("2h32m", 9120)
    ensure_roundtrip("+2h32m", 9120)
    ensure_roundtrip("3d2h32m", 268320)
    ensure_roundtrip("+3d2h32m", 268320)
    ensure_roundtrip("1w3d2h32m", 873120)
    ensure_roundtrip("1w 3d 2h 32m", 873120)
    ensure_roundtrip("1 w 3 d 2 h 32 m", 873120)
    ensure_roundtrip("4:13", 253)
    ensure_roundtrip(":13", 13)
    ensure_roundtrip("4:13:02", 15182)
    ensure_roundtrip("4:13:02.266", 15182.266)
    ensure_roundtrip("2:04:13:02.266", 187982.266)
    ensure_roundtrip("2 days,  4:13:02", 187982)
    ensure_roundtrip("5hr34m56s", 20096)
    ensure_roundtrip("5 hours, 34 minutes, 56 seconds", 20096)
    ensure_roundtrip("5 hrs, 34 mins, 56 secs", 20096)
    ensure_roundtrip("2 days, 5 hours, 34 minutes, 56 seconds", 192896)
    ensure_roundtrip("172 hr", 619200)
