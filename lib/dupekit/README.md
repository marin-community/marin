dupekit
---
Raison d'être: Home for the Rust code used for text deduplication.

## Install

- **Locally:** This code is auto-magically built by `uv` via Cargo and [Maturin](https://github.com/PyO3/maturin). You
might need to install them (e.g., `brew install maturin rust` on macOS).
- **Cluster:** This code is compiled as part of the Docker build (`uv pip install -e ...` step): Maturin builds the Rust
code and places it in the system `site-packages` (e.g., `/home/ray/anaconda3/lib/python3.11/site-packages/dupekit/dupekit.abi3.so`).

>[!NOTE]
>What about making `dupekit` a hybrid Python/Rust Maturin workspace? We tried and experienced issues getting
the Docker build to work while keeping it simple—a simple Rust workspace helps keep the setup clean.

>[!WARNING]
>`dupekit` is configured as an optional dependency, thus must be explicitly requested e.g. via an `extra` group.
>If you get `ModuleNotFoundError: No module named 'dupekit'`, make sure you have requested/installed it.

## Benchmarking

The goal of these benchmarks is to test different ways of marshaling large text content
between Python and Rust "foreign function interface" ([wiki:FFI](https://en.wikipedia.org/wiki/Foreign_function_interface)). These tests are designed to isolate the overhead of marshaling
from the actual Rust computation (by doing minimal processing in Rust).

Dataset: 1 shard of [`HuggingFaceFW/fineweb-edu/sample/10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT) (2.15 GB Parquet file, benchmarked on **250k** out of 726k documents)

Install:
```shell
uv sync --all-packages --extra=benchmark --group dev
```

Benchmark (Takes a few minutes):
```shell
uv run pytest lib/dupekit/tests/bench/test_dedupe.py --run-benchmark --benchmark-min-rounds=20
uv run pytest lib/dupekit/tests/bench/test_marshaling.py --run-benchmark
uv run pytest lib/dupekit/tests/bench/test_batch_tuning.py --run-benchmark
uv run pytest lib/dupekit/tests/bench/test_io.py --run-benchmark
uv run pytest lib/dupekit/tests/bench/test_hashing.py --run-benchmark
uv run pytest lib/dupekit/tests/bench/test_minhash.py --run-benchmark
```
Note: Run separated by type of benchmark (otherwise results are mixed within one table)

Footprint (Note: sampling the stack might taint the mem measurements, so we disable benchmarking):
```shell
uv run pytest lib/dupekit/tests/bench/test_marshaling.py \
  --run-benchmark \
  --benchmark-disable \
  --memray \
  --native \
  --most-allocations=0
```
### Results
#### Dedup: Rust vs. Python
```
---------------------------------------------------------------------------- benchmark 'Documents: Exact Deduplication': 2 tests ----------------------------------------------------------------------------
Name (time in ms)                             Min                 Max                Mean            StdDev              Median               IQR            Outliers       OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_deduplication[rust-documents]         3.9872 (1.0)        5.2516 (1.0)        4.2341 (1.0)      0.1949 (1.0)        4.2247 (1.0)      0.2845 (1.0)          52;2  236.1805 (1.0)         188           1
test_deduplication[python-documents]     133.8747 (33.58)    157.3844 (29.97)    139.6233 (32.98)    7.7842 (39.94)    135.3300 (32.03)    9.6717 (34.00)         5;0    7.1621 (0.03)         20           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------- benchmark 'Documents: Hash Generation': 2 tests ---------------------------------------------------------------------------
Name (time in ms)                       Min                 Max                Mean            StdDev              Median               IQR            Outliers       OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_hashing[rust-documents]         2.2755 (1.0)        2.4842 (1.0)        2.3041 (1.0)      0.0301 (1.0)        2.2938 (1.0)      0.0381 (1.0)          50;9  434.0169 (1.0)         375           1
test_hashing[python-documents]     130.0445 (57.15)    132.3783 (53.29)    130.7795 (56.76)    0.6663 (22.13)    130.5910 (56.93)    0.6259 (16.44)         5;3    7.6465 (0.02)         20           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------- benchmark 'Paragraphs: Exact Deduplication': 2 tests ----------------------------------------------------------------------------
Name (time in ms)                              Min                 Max                Mean             StdDev              Median                IQR            Outliers      OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_deduplication[rust-paragraphs]        85.4666 (1.0)      109.3652 (1.0)       90.2916 (1.0)       6.8294 (1.0)       87.3405 (1.0)       2.0275 (1.0)           4;4  11.0752 (1.0)          20           1
test_deduplication[python-paragraphs]     303.0885 (3.55)     342.9836 (3.14)     321.3022 (3.56)     13.8377 (2.03)     329.4886 (3.77)     25.1111 (12.39)         9;0   3.1123 (0.28)         20           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------- benchmark 'Paragraphs: Hash Generation': 2 tests ---------------------------------------------------------------------------
Name (time in ms)                        Min                 Max                Mean             StdDev              Median               IQR            Outliers      OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_hashing[rust-paragraphs]        23.9739 (1.0)       26.9860 (1.0)       25.3823 (1.0)       0.4419 (1.0)       25.3160 (1.0)      0.2099 (1.0)           5;5  39.3975 (1.0)          38           1
test_hashing[python-paragraphs]     247.5415 (10.33)    321.4654 (11.91)    255.3421 (10.06)    19.0653 (43.15)    249.1948 (9.84)     1.7899 (8.53)          2;2   3.9163 (0.10)         20           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Marshaling
```
-------------------------------------------------------------------------------------------- benchmark: 7 tests -------------------------------------------------------------------------------------------
Name (time in ms)                    Min                   Max                  Mean             StdDev                Median                IQR            Outliers      OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_arrow_giant                 86.4414 (1.0)         96.0537 (1.01)        90.0259 (1.0)       2.8582 (31.54)       90.4363 (1.0)       4.1787 (27.96)         3;0  11.1079 (1.0)          11           1
test_arrow_small                 94.4010 (1.09)        94.6679 (1.0)         94.5616 (1.05)      0.0906 (1.0)         94.5570 (1.05)      0.1494 (1.0)           5;0  10.5751 (0.95)         11           1
test_dicts_batched_stream     3,975.1581 (45.99)    3,979.7102 (42.04)    3,977.7639 (44.18)     1.8357 (20.26)    3,978.3399 (43.99)     2.8370 (18.98)         2;0   0.2514 (0.02)          5           1
test_dicts_batch              4,398.7191 (50.89)    4,421.9632 (46.71)    4,410.0489 (48.99)     8.7694 (96.78)    4,411.2232 (48.78)    12.0295 (80.50)         2;0   0.2268 (0.02)          5           1
test_dicts_loop               4,411.8727 (51.04)    4,457.0985 (47.08)    4,431.9081 (49.23)    19.8323 (218.86)   4,430.5465 (48.99)    35.6846 (238.78)        2;0   0.2256 (0.02)          5           1
test_rust_structs             4,449.5728 (51.47)    4,479.8173 (47.32)    4,465.2999 (49.60)    14.1041 (155.65)   4,472.5336 (49.46)    24.8971 (166.60)        3;0   0.2239 (0.02)          5           1
test_arrow_tiny               7,023.5789 (81.25)    7,064.2094 (74.62)    7,044.9691 (78.25)    19.4414 (214.55)   7,047.1538 (77.92)    37.8036 (252.96)        1;0   0.1419 (0.01)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### PyArrow Batch Size
```
--------------------------------------------------------------------------------------------- benchmark: 11 tests ----------------------------------------------------------------------------------------------
Name (time in ms)                         Min                   Max                  Mean             StdDev                Median                IQR            Outliers      OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_arrow_batch_sizes[8192]          28.6030 (1.0)         32.0178 (1.07)        29.2802 (1.0)       0.8846 (4.75)        28.9333 (1.0)       0.7970 (3.52)          5;3  34.1528 (1.0)          34           1
test_arrow_batch_sizes[16384]         28.7303 (1.00)        30.8987 (1.03)        29.3111 (1.00)      0.5447 (2.92)        29.1404 (1.01)      0.5907 (2.61)          9;2  34.1168 (1.00)         33           1
test_arrow_batch_sizes[4096]          28.8488 (1.01)        30.1474 (1.01)        29.2876 (1.00)      0.3776 (2.03)        29.2212 (1.01)      0.6339 (2.80)         12;0  34.1441 (1.00)         34           1
test_arrow_batch_sizes[2048]          29.1493 (1.02)        30.4442 (1.02)        29.5710 (1.01)      0.3013 (1.62)        29.5505 (1.02)      0.3483 (1.54)         10;1  33.8169 (0.99)         32           1
test_arrow_batch_sizes[32768]         29.2200 (1.02)        29.9410 (1.0)         29.5896 (1.01)      0.1863 (1.0)         29.5706 (1.02)      0.2423 (1.07)         11;0  33.7956 (0.99)         34           1
test_arrow_batch_sizes[65536]         30.3973 (1.06)        31.3805 (1.05)        30.9409 (1.06)      0.2453 (1.32)        30.9829 (1.07)      0.2263 (1.0)           9;3  32.3197 (0.95)         33           1
test_arrow_batch_sizes[131072]        30.7074 (1.07)        33.1845 (1.11)        31.4322 (1.07)      0.6799 (3.65)        31.1102 (1.08)      0.8739 (3.86)          6;1  31.8145 (0.93)         32           1
test_arrow_batch_sizes[1024]          30.7724 (1.08)        32.6049 (1.09)        31.6173 (1.08)      0.5506 (2.96)        31.6311 (1.09)      0.9233 (4.08)         13;0  31.6283 (0.93)         30           1
test_arrow_batch_sizes[512]           33.8866 (1.18)        36.2981 (1.21)        34.5224 (1.18)      0.6189 (3.32)        34.2960 (1.19)      0.5087 (2.25)          6;3  28.9667 (0.85)         29           1
test_arrow_batch_sizes[128]           51.0530 (1.78)        56.3190 (1.88)        53.5492 (1.83)      1.6124 (8.65)        53.7474 (1.86)      2.3557 (10.41)         7;0  18.6744 (0.55)         18           1
test_arrow_batch_sizes[1]          2,781.2088 (97.23)    2,812.2547 (93.93)    2,797.8572 (95.55)    11.6892 (62.74)    2,801.0024 (96.81)    15.3956 (68.03)         2;0   0.3574 (0.01)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### I/O
```
------------------------------------------------------------------------------- benchmark: 4 tests ------------------------------------------------------------------------------
Name (time in s)          Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_rust_native       1.6757 (1.0)      1.6848 (1.0)      1.6794 (1.0)      0.0035 (1.26)     1.6783 (1.0)      0.0047 (1.73)          2;0  0.5955 (1.0)           5           1
test_arrow_giant       2.9501 (1.76)     2.9570 (1.76)     2.9521 (1.76)     0.0028 (1.0)      2.9511 (1.76)     0.0027 (1.0)           1;0  0.3387 (0.57)          5           1
test_arrow_small       3.3476 (2.00)     3.6588 (2.17)     3.5583 (2.12)     0.1241 (44.48)    3.5726 (2.13)     0.1289 (47.18)         1;0  0.2810 (0.47)          5           1
test_dicts_loop_io     7.3664 (4.40)     7.3913 (4.39)     7.3837 (4.40)     0.0101 (3.63)     7.3871 (4.40)     0.0113 (4.14)          1;0  0.1354 (0.23)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Hashing
```
--------------------------------------------------------------------------------------- benchmark: 6 tests ---------------------------------------------------------------------------------------
Name (time in ms)                     Min                Max               Mean            StdDev             Median               IQR            Outliers       OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_hash_rust_xxh3_64_batch       4.4886 (1.0)       4.9466 (1.0)       4.5860 (1.0)      0.0616 (1.57)      4.5939 (1.0)      0.0957 (2.52)         74;1  218.0558 (1.0)         210           1
test_hash_rust_xxh3_64_scalar      5.0276 (1.12)      5.3367 (1.08)      5.1276 (1.12)     0.0393 (1.0)       5.1307 (1.12)     0.0379 (1.0)         41;12  195.0244 (0.89)        190           1
test_hash_rust_xxh3_128            6.1686 (1.37)      6.5772 (1.33)      6.2901 (1.37)     0.1098 (2.79)      6.2334 (1.36)     0.1731 (4.56)         37;0  158.9811 (0.73)        160           1
test_hash_rust_blake3             28.7743 (6.41)     29.0392 (5.87)     28.8919 (6.30)     0.0593 (1.51)     28.8799 (6.29)     0.0709 (1.87)         10;1   34.6118 (0.16)         35           1
test_hash_rust_blake2             54.1043 (12.05)    55.0271 (11.12)    54.4180 (11.87)    0.3711 (9.43)     54.1916 (11.80)    0.7337 (19.34)         5;0   18.3763 (0.08)         19           1
test_hash_python_blake2b          84.0109 (18.72)    84.1698 (17.02)    84.0611 (18.33)    0.0465 (1.18)     84.0469 (18.30)    0.0595 (1.57)          3;0   11.8961 (0.05)         12           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

Mem Footprin (sorted from high to low):
```shell
Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_rust_structs at the high watermark

	 📦 Total memory allocated: 4.3GiB
	 📏 Total allocations: 21
	 📊 Histogram of allocation sizes: | ▃█▁▃|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_dicts_batch at the high watermark

	 📦 Total memory allocated: 3.3GiB
	 📏 Total allocations: 20
	 📊 Histogram of allocation sizes: |  ▁█▂|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_dicts_loop at the high watermark

	 📦 Total memory allocated: 3.3GiB
	 📏 Total allocations: 19
	 📊 Histogram of allocation sizes: |  ▁█▂|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_arrow_giant at the high watermark

	 📦 Total memory allocated: 64.9MiB
	 📏 Total allocations: 36
	 📊 Histogram of allocation sizes: |▅█   |

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_dicts_batched_stream at the high watermark

	 📦 Total memory allocated: 28.1MiB
	 📏 Total allocations: 7
	 📊 Histogram of allocation sizes: |█▄▄▄▄|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_arrow_tiny at the high watermark

	 📦 Total memory allocated: 22.0MiB
	 📏 Total allocations: 37
	 📊 Histogram of allocation sizes: |█▇   |

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_arrow_small at the high watermark

	 📦 Total memory allocated: 551.7KiB
	 📏 Total allocations: 42
	 📊 Histogram of allocation sizes: |▂█▁  |
```
___
**Statement of attribution:**
- This code was seeded from [nelson-liu/rbloom-gcs](https://github.com/nelson-liu/rbloom-gcs).
- Bloom filters were originally proposed in [(Bloom, 1970)](https://doi.org/10.1145/362686.362692). Furthermore, this  implementation makes use of a constant recommended by [(L'Ecuyer, 1999)](https://doi.org/10.1090/S0025-5718-99-00996-5) for  redistributing the entropy of a single hash over multiple integers using a  linear congruential generator.
