# rBloom

[![PyPI](https://img.shields.io/pypi/v/rbloom)](https://pypi.org/project/rbloom/)
[![license](https://img.shields.io/github/license/KenanHanke/rbloom)](https://github.com/KenanHanke/rbloom/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17084925.svg)](https://doi.org/10.5281/zenodo.17084925)
[![build](https://img.shields.io/github/actions/workflow/status/KenanHanke/rbloom/CI.yml)](https://github.com/KenanHanke/rbloom/actions)

## Fork Differences

This differs from the upstream [KenanHanke/rbloom](https://github.com/KenanHanke/rbloom) with the following changes:

- **Pre-hashing API**: Accepts pre-hashed `i128` integers instead of arbitrary objects. The `hash_func` parameter has been removed entirely.
- **File object support**: `save()` and `load()` now accept file-like objects (e.g., `io.BytesIO`) in addition to file paths.
- **Build targets**: Simplified to x86_64 only, Python 3.11+, with wheels hosted via GitHub releases.

A fast, simple and lightweight
[Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter) library for
Python, implemented in Rust. It's designed to be as pythonic as possible,
mimicking the built-in `set` type where it can. This fork requires pre-hashed
integer values, giving you full control over the hashing function. While it's
a new library (this project was started in 2023), it's currently the fastest
option for Python by a long shot (see the section [Benchmarks](#benchmarks)).

## Quickstart

This library defines only one class, which can be used as follows:

```python
>>> from rbloom import Bloom
>>> from hashlib import sha256
>>>
>>> # Helper function to hash objects to i128 integers
>>> def hash_obj(obj):
...     h = sha256(str(obj).encode()).digest()
...     return int.from_bytes(h[:16], "big", signed=True)
>>>
>>> bf = Bloom(200, 0.01)  # 200 items max, false positive rate of 1%
>>> bf.add(hash_obj("hello"))
>>> hash_obj("hello") in bf
True
>>> hash_obj("world") in bf
False
>>> bf.update([hash_obj("hello"), hash_obj("world")])  # both now in bf
>>> other_bf = Bloom(200, 0.01)

### add some items to other_bf

>>> third_bf = bf | other_bf    # third_bf now contains all items in
                                # bf and other_bf
>>> third_bf = bf.copy()
... third_bf.update(other_bf)   # same as above
>>> bf.issubset(third_bf)    # bf <= third_bf also works
True
```

For the full API, see the section [Documentation](#documentation).

## Installation

On almost all platforms, simply run:

```sh
pip install rbloom
```

If you're on an uncommon platform, this may cause pip to build the library
from source, which requires the Rust
[toolchain](https://www.rust-lang.org/tools/install). You can also build
`rbloom` by cloning this repository and running
[maturin](https://github.com/PyO3/maturin):

```sh
maturin build --release
```

This will create a wheel in the `target/wheels/` directory, which can
subsequently also be passed to pip.

## Why rBloom?

Why should you use this library instead of one of the other
Bloom filter libraries on PyPI?

- **Simple:** Almost all important methods work exactly like their
  counterparts in the built-in `set`
  [type](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset).
- **Fast:** `rbloom` is implemented in Rust, which makes it
  blazingly fast. See section [Benchmarks](#benchmarks) for more
  information.
- **Lightweight:** `rbloom` has no dependencies of its own.
- **Maintainable:** This library is very concise, and it's written
  in idiomatic Rust. Even if I were to stop maintaining `rbloom` (which I
  don't intend to), it would be trivially easy for you to fork it and keep
  it working for you.

I started `rbloom` because I was looking for a simple Bloom filter
dependency for a project, and the only sufficiently fast option
(`pybloomfiltermmap3`) was segfaulting on recent Python versions. `rbloom`
ended up being twice as fast and has grown to encompass a more complete
API (e.g. with set comparisons like `issubset`). Do note that it doesn't
use mmapped files, however. This shouldn't be an issue in most cases, as
the random access heavy nature of a Bloom filter negates the benefits of
mmap after very few operations, but it is something to keep in mind for
edge cases.

## Benchmarks

The following simple benchmark was implemented in the respective API of
each library (see the [comparison benchmarks](benchmarks/compare.py)):

```python
from hashlib import sha256

def hash_obj(obj):
    h = sha256(str(obj).encode()).digest()
    return int.from_bytes(h[:16], "big", signed=True)

bf = Bloom(10_000_000, 0.01)

for i in range(10_000_000):
    bf.add(hash_obj(i + 0.5))

for i in range(10_000_000):
    assert hash_obj(i + 0.5) in bf
```

This resulted in the following average runtimes on an M1 Pro (confirmed to be proportional to runtimes on an Intel machine):

| Library                                                            | Time    | Notes                                 |
| ------------------------------------------------------------------ | ------- | ------------------------------------- |
| [rBloom](https://pypi.org/project/rbloom/)                         | 2.52s   | works out-of-the-box                  |
| [pybloomfiltermmap3](https://pypi.org/project/pybloomfiltermmap3/) | 4.78s   | unreliable [1]                        |
| [pybloom3](https://pypi.org/project/pybloom3/)                     | 46.76s  | works out-of-the-box                  |
| [Flor](https://pypi.org/project/Flor/)                             | 76.94s  | doesn't work on arbitrary objects [2] |
| [bloom-filter2](https://pypi.org/project/bloom-filter2/)           | 165.54s | doesn't work on arbitrary objects [2] |

[1] The official package failed to install on Python 3.11 and kept segfaulting on 3.10 (Linux, January 2023). It seems to be fine for now (October 2024).
[2] I was forced to convert to a byte representation, which is bad default behavior as it presents
the problems mentioned below in the section "Cryptographic security".

Also note that `rbloom` is compiled against a stable ABI for
portability, and that you can get a small but measurable speedup by
removing the `"abi3-py311"` flag from `Cargo.toml` and building
it yourself.

## Documentation

This library defines only one class, the signature of which should be
thought of as follows. Note that this fork requires pre-hashed i128
integers rather than arbitrary objects:

```python
class Bloom:

    # expected_items:  max number of items to be added to the filter
    # false_positive_rate:  max false positive rate of the filter
    def __init__(self, expected_items: int, false_positive_rate: float)

    @property
    def size_in_bits(self) -> int      # number of buckets in the filter

    @property
    def approx_items(self) -> float    # estimated number of items in
                                       # the filter

    # see section "Persistence" for more information on these methods
    # filepath can be a string path or a file-like object with write()/read()
    @classmethod
    def load(cls, filepath: Union[str, IO]) -> Bloom
    def save(self, filepath: Union[str, IO])
    @classmethod
    def load_bytes(cls, data: bytes) -> Bloom
    def save_bytes(self) -> bytes

    #####################################################################
    #                    ALL SUBSEQUENT METHODS ARE                     #
    #              EQUIVALENT TO THE CORRESPONDING METHODS              #
    #              OF THE BUILT-IN SET TYPE, EXCEPT THEY                #
    #                 EXPECT PRE-HASHED i128 INTEGERS                   #
    #####################################################################

    def add(self, hashed: int)                    # add hashed value to self
    def __contains__(self, hashed: int) -> bool   # check if hashed in self
    def __bool__(self) -> bool                    # False if empty
    def __repr__(self) -> str                     # basic info

    def __or__(self, other: Bloom) -> Bloom       # self | other
    def __ior__(self, other: Bloom)               # self |= other
    def __and__(self, other: Bloom) -> Bloom      # self & other
    def __iand__(self, other: Bloom)              # self &= other

    # these extend the functionality of __or__, __ior__, __and__, __iand__
    # iterables should contain pre-hashed i128 integers
    def union(self, *others: Union[Iterable[int], Bloom]) -> Bloom        # __or__
    def update(self, *others: Union[Iterable[int], Bloom])                # __ior__
    def intersection(self, *others: Union[Iterable[int], Bloom]) -> Bloom # __and__
    def intersection_update(self, *others: Union[Iterable[int], Bloom])   # __iand__

    # these implement <, >, <=, >=, ==, !=
    def __lt__, __gt__, __le__, __ge__, __eq__, __ne__(self,
                                                       other: Bloom) -> bool
    def issubset(self, other: Bloom) -> bool      # self <= other
    def issuperset(self, other: Bloom) -> bool    # self >= other

    def clear(self)                               # remove all items
    def copy(self) -> Bloom                       # duplicate self
```

To prevent death and destruction, the bitwise set operations only work on
filters where all parameters are equal. Because this is a Bloom filter, the
`__contains__` and `approx_items` methods are probabilistic, as are all the
methods that compare two filters (such as `__le__` and `issubset`).

## Pre-hashing

This fork requires you to pre-hash your objects to i128 integers before
adding them to the filter. Your hash function must return an integer between
-2^127 and 2^127 - 1. This gives you full control over the hashing strategy.

For most use cases, SHA256-based hashing is recommended:

```python
from rbloom import Bloom
from hashlib import sha256

def hash_obj(obj):
    h = sha256(str(obj).encode()).digest()
    # use sys.byteorder instead of "big" for a small speedup when
    # reproducibility across machines isn't a concern
    return int.from_bytes(h[:16], "big", signed=True)

bf = Bloom(100_000_000, 0.01)
bf.add(hash_obj("my_item"))
assert hash_obj("my_item") in bf
```

If you need Python-compatible hashing (where `1`, `1.0`, and `True` are
considered equal), you can use `hash()`:

```python
def hash_obj(obj):
    h = hash(obj)
    # Clamp to i128 range
    return max(-2**127, min(h, 2**127 - 1))
```

Note that `hash()` uses a random salt that changes between Python
invocations, so filters using `hash()` cannot be persisted to disk.

## Persistence

The `save` and `load` methods, along with their byte-oriented counterparts
`save_bytes` and `load_bytes`, allow you to save and load filters to and
from disk/Python `bytes` objects. The `save` and `load` methods accept either
file paths (as strings) or file-like objects (anything with `write()`/`read()` methods).

```python
from hashlib import sha256
import io

def hash_obj(obj):
    h = sha256(str(obj).encode()).digest()
    return int.from_bytes(h[:16], "big", signed=True)

bf = Bloom(10_000, 0.01)
bf.add(hash_obj("hello"))
bf.add(hash_obj("world"))

# saving to a file path
bf.save("bf.bloom")

# loading from a file path
loaded_bf = Bloom.load("bf.bloom")
assert loaded_bf == bf

# saving to a file-like object
buffer = io.BytesIO()
bf.save(buffer)

# loading from a file-like object
buffer.seek(0)
loaded_bf = Bloom.load(buffer)
assert loaded_bf == bf

# saving to bytes
bf_bytes = bf.save_bytes()

# loading from bytes
loaded_bf_from_bytes = Bloom.load_bytes(bf_bytes)
assert loaded_bf_from_bytes == bf
```

The size of the saved filter is `bf.size_in_bits / 8 + 8` bytes.

---

**Statement of attribution:** Bloom filters were originally proposed in
[(Bloom, 1970)](https://doi.org/10.1145/362686.362692). Furthermore, this
implementation makes use of a constant recommended by
[(L'Ecuyer, 1999)](https://doi.org/10.1090/S0025-5718-99-00996-5) for
redistributing the entropy of a single hash over multiple integers using a
linear congruential generator.
