import hashlib
import struct


def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.

    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]
