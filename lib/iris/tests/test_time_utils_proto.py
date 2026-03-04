# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.time_utils import Duration, Timestamp


def test_timestamp_proto_roundtrip():
    """Timestamp proto serialization handles edge cases correctly."""
    original = Timestamp.now()
    proto = original.to_proto()
    restored = Timestamp.from_proto(proto)
    assert original == restored

    epoch_zero = Timestamp.from_ms(0)
    proto_zero = epoch_zero.to_proto()
    restored_zero = Timestamp.from_proto(proto_zero)
    assert epoch_zero == restored_zero
    assert restored_zero.epoch_ms() == 0

    negative = Timestamp.from_ms(-1000)
    proto_neg = negative.to_proto()
    restored_neg = Timestamp.from_proto(proto_neg)
    assert negative == restored_neg
    assert restored_neg.epoch_ms() == -1000

    large = Timestamp.from_ms(4102444800000)  # 2100-01-01
    proto_large = large.to_proto()
    restored_large = Timestamp.from_proto(proto_large)
    assert large == restored_large


def test_duration_proto_roundtrip():
    """Duration proto serialization handles edge cases correctly."""
    original = Duration.from_seconds(5.5)
    proto = original.to_proto()
    restored = Duration.from_proto(proto)
    assert original == restored

    zero = Duration.from_ms(0)
    proto_zero = zero.to_proto()
    restored_zero = Duration.from_proto(proto_zero)
    assert zero == restored_zero
    assert restored_zero.to_ms() == 0

    negative = Duration.from_ms(-5000)
    proto_neg = negative.to_proto()
    restored_neg = Duration.from_proto(proto_neg)
    assert negative == restored_neg
    assert restored_neg.to_ms() == -5000

    large = Duration.from_hours(24 * 365)  # 1 year
    proto_large = large.to_proto()
    restored_large = Duration.from_proto(proto_large)
    assert large == restored_large
