# MIT License
#
# Copyright (c) 2023-2025 Kenan Hanke
# Copyright (c) 2023-2025 Zachary Dremann
# Copyright (c) 2023 Rory McNamara
# Copyright (c) 2024 Dan Lenski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from typing import Any, BinaryIO, Iterable, Union, final

@final
class Bloom:

    # expected_items:  max number of items to be added to the filter
    # false_positive_rate:  max false positive rate of the filter
    # Note: This bloom filter expects pre-hashed integers (i128) for add() and __contains__()
    def __init__(self, expected_items: int, false_positive_rate: float) -> None: ...

    # number of buckets in the filter
    @property
    def size_in_bits(self) -> int: ...

    # estimated number of items in the filter
    @property
    def approx_items(self) -> float: ...

    # load from file path or file-like object, see section "Persistence"
    @classmethod
    def load(cls, source: Union[str, bytes, os.PathLike, BinaryIO]) -> Bloom: ...

    # load from bytes(), see section "Persistence"
    @classmethod
    def load_bytes(cls, data: bytes) -> Bloom: ...

    # save to file path or file-like object, see section "Persistence"
    def save(self, dest: Union[str, bytes, os.PathLike, BinaryIO]) -> None: ...

    # save to a bytes(), see section "Persistence"
    def save_bytes(self) -> bytes: ...

    #####################################################################
    #                    ALL SUBSEQUENT METHODS ARE                     #
    #              EQUIVALENT TO THE CORRESPONDING METHODS              #
    #                     OF THE BUILT-IN SET TYPE                      #
    #      EXCEPT THEY ACCEPT PRE-HASHED INTEGERS (i128) INSTEAD        #
    #####################################################################

    def add(self, hashed: int, /) -> None: ...
    def __contains__(self, hashed: int) -> bool: ...
    def __bool__(self) -> bool: ...  # False if empty
    def __repr__(self) -> str: ...  # basic info
    def __or__(self, other: Bloom) -> Bloom: ...  # self | other
    def __ior__(self, other: Bloom) -> None: ...  # self |= other
    def __and__(self, other: Bloom) -> Bloom: ...  # self & other
    def __iand__(self, other: Bloom) -> None: ...  # self &= other

    # extension of __or__
    def union(self, *others: Union[Iterable[int], Bloom]) -> Bloom: ...

    # extension of __ior__
    def update(self, *others: Union[Iterable[int], Bloom]) -> None: ...

    # extension of __and__
    def intersection(self, *others: Union[Iterable[int], Bloom]) -> Bloom: ...

    # extension of __iand__
    def intersection_update(self, *others: Union[Iterable[int], Bloom]) -> None: ...

    # these implement <, >, <=, >=, ==, !=
    def __lt__(self, other: Bloom) -> bool: ...
    def __gt__(self, other: Bloom) -> bool: ...
    def __le__(self, other: Bloom) -> bool: ...
    def __ge__(self, other: Bloom) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def issubset(self, other: Bloom, /) -> bool: ...  # self <= other
    def issuperset(self, other: Bloom, /) -> bool: ...  # self >= other
    def clear(self) -> None: ...  # remove all items
    def copy(self) -> Bloom: ...  # duplicate self
