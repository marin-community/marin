# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proto-specific tests for Duration and Timestamp proto roundtrips.

Non-proto timing tests have moved to lib/rigging/tests/test_timing.py.
"""

from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_from_proto, timestamp_to_proto
from rigging.timing import Duration, Timestamp


def test_timestamp_proto_roundtrip():
    """Timestamp proto serialization handles edge cases correctly."""
    original = Timestamp.now()
    proto = timestamp_to_proto(original)
    restored = timestamp_from_proto(proto)
    assert original == restored

    epoch_zero = Timestamp.from_ms(0)
    proto_zero = timestamp_to_proto(epoch_zero)
    restored_zero = timestamp_from_proto(proto_zero)
    assert epoch_zero == restored_zero
    assert restored_zero.epoch_ms() == 0

    negative = Timestamp.from_ms(-1000)
    proto_neg = timestamp_to_proto(negative)
    restored_neg = timestamp_from_proto(proto_neg)
    assert negative == restored_neg
    assert restored_neg.epoch_ms() == -1000

    large = Timestamp.from_ms(4102444800000)  # 2100-01-01
    proto_large = timestamp_to_proto(large)
    restored_large = timestamp_from_proto(proto_large)
    assert large == restored_large


def test_duration_proto_roundtrip():
    """Duration proto serialization handles edge cases correctly."""
    original = Duration.from_seconds(5.5)
    proto = duration_to_proto(original)
    restored = duration_from_proto(proto)
    assert original == restored

    zero = Duration.from_ms(0)
    proto_zero = duration_to_proto(zero)
    restored_zero = duration_from_proto(proto_zero)
    assert zero == restored_zero
    assert restored_zero.to_ms() == 0

    negative = Duration.from_ms(-5000)
    proto_neg = duration_to_proto(negative)
    restored_neg = duration_from_proto(proto_neg)
    assert negative == restored_neg
    assert restored_neg.to_ms() == -5000

    large = Duration.from_hours(24 * 365)
    proto_large = duration_to_proto(large)
    restored_large = duration_from_proto(proto_large)
    assert large == restored_large
