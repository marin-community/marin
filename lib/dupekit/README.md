dupekit
---
🚧 WIP 🚧

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
uv run pytest lib/dupekit/tests/bench/test_marshaling.py --run-benchmark
uv run pytest lib/dupekit/tests/bench/test_batch_tuning.py --run-benchmark
uv run pytest lib/dupekit/tests/bench/test_io.py --run-benchmark
uv run pytest lib/dupekit/tests/bench/test_hashing.py --run-benchmark
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

Marshaling:
```
-------------------------------------------------------------------------------------------- benchmark: 7 tests -------------------------------------------------------------------------------------------
Name (time in ms)                    Min                   Max                  Mean             StdDev                Median                IQR            Outliers      OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_arrow_giant                 55.9836 (1.0)         58.2390 (1.0)         56.5522 (1.0)       0.7886 (1.0)         56.1923 (1.0)       0.7495 (1.0)           2;1  17.6828 (1.0)          10           1
test_arrow_small                 56.0370 (1.00)        58.7577 (1.01)        57.3037 (1.01)      0.9010 (1.14)        57.2863 (1.02)      1.7235 (2.30)          8;0  17.4509 (0.99)         18           1
test_dicts_batched_stream     1,778.3461 (31.77)    1,835.7805 (31.52)    1,798.0700 (31.79)    25.1567 (31.90)    1,781.9829 (31.71)    37.1003 (49.50)         1;0   0.5562 (0.03)          5           1
test_dicts_batch              2,080.9016 (37.17)    2,265.2497 (38.90)    2,174.9781 (38.46)    69.6372 (88.30)    2,182.9771 (38.85)    97.2326 (129.73)        2;0   0.4598 (0.03)          5           1
test_dicts_loop               2,114.4937 (37.77)    2,174.1042 (37.33)    2,146.2951 (37.95)    25.9456 (32.90)    2,141.2760 (38.11)    45.6841 (60.95)         2;0   0.4659 (0.03)          5           1
test_rust_structs             2,357.4601 (42.11)    2,514.4476 (43.17)    2,422.5251 (42.84)    62.9378 (79.81)    2,413.3255 (42.95)    96.5711 (128.85)        2;0   0.4128 (0.02)          5           1
test_arrow_tiny               3,395.4023 (60.65)    3,526.9167 (60.56)    3,440.3690 (60.84)    51.3052 (65.06)    3,429.9975 (61.04)    54.6624 (72.93)         1;0   0.2907 (0.02)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
```

PyArrow Batch Size:
```
-------------------------------------------------------------------------------------------- benchmark: 11 tests ---------------------------------------------------------------------------------------------
Name (time in ms)                         Min                   Max                  Mean            StdDev                Median               IQR            Outliers      OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_arrow_batch_sizes[2048]          20.7560 (1.0)         22.9952 (1.02)        21.2667 (1.0)      0.5215 (1.45)        21.0788 (1.0)      0.3261 (2.06)          7;7  47.0218 (1.0)          45           1
test_arrow_batch_sizes[4096]          20.8044 (1.00)        23.8119 (1.06)        21.3230 (1.00)     0.5752 (1.60)        21.0865 (1.00)     0.3438 (2.17)          7;7  46.8977 (1.00)         46           1
test_arrow_batch_sizes[131072]        20.8506 (1.00)        22.5053 (1.0)         21.2994 (1.00)     0.3647 (1.02)        21.1946 (1.01)     0.2305 (1.46)          8;6  46.9498 (1.00)         44           1
test_arrow_batch_sizes[8192]          21.2869 (1.03)        24.0307 (1.07)        21.9948 (1.03)     0.7101 (1.98)        21.6701 (1.03)     0.9976 (6.30)          7;1  45.4654 (0.97)         44           1
test_arrow_batch_sizes[1024]          21.4135 (1.03)        23.7615 (1.06)        21.9310 (1.03)     0.4709 (1.31)        21.7954 (1.03)     0.3408 (2.15)          6;5  45.5975 (0.97)         44           1
test_arrow_batch_sizes[16384]         21.8802 (1.05)        23.3602 (1.04)        22.2929 (1.05)     0.3585 (1.0)         22.1901 (1.05)     0.1583 (1.0)           9;9  44.8574 (0.95)         43           1
test_arrow_batch_sizes[65536]         22.9272 (1.10)        25.8951 (1.15)        23.4272 (1.10)     0.4639 (1.29)        23.3029 (1.11)     0.1784 (1.13)          4;6  42.6854 (0.91)         42           1
test_arrow_batch_sizes[32768]         23.1957 (1.12)        25.7514 (1.14)        23.9128 (1.12)     0.4346 (1.21)        23.7943 (1.13)     0.2274 (1.44)          6;6  41.8186 (0.89)         41           1
test_arrow_batch_sizes[512]           23.2635 (1.12)        25.5281 (1.13)        23.7065 (1.11)     0.5275 (1.47)        23.4970 (1.11)     0.4111 (2.60)          3;3  42.1825 (0.90)         41           1
test_arrow_batch_sizes[128]           31.4752 (1.52)        34.1684 (1.52)        32.0354 (1.51)     0.6418 (1.79)        31.8064 (1.51)     0.5851 (3.70)          5;2  31.2154 (0.66)         31           1
test_arrow_batch_sizes[1]          1,344.6158 (64.78)    1,363.3658 (60.58)    1,357.5016 (63.83)    7.3751 (20.57)    1,360.0707 (64.52)    5.5255 (34.91)         1;1   0.7366 (0.02)          5           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

I/O:
```
----------------------------------------------------------------------------------------- benchmark: 4 tests ----------------------------------------------------------------------------------------
Name (time in ms)             Min                   Max                  Mean              StdDev                Median                 IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_arrow_small         674.4445 (1.0)        704.9876 (1.0)        688.8099 (1.0)       11.3550 (1.16)       686.7134 (1.0)       14.6090 (1.36)          2;0  1.4518 (1.0)           5           1
test_arrow_giant       1,070.4218 (1.59)     1,097.6739 (1.56)     1,084.1012 (1.57)       9.8224 (1.0)      1,083.3669 (1.58)      10.7357 (1.0)           2;0  0.9224 (0.64)          5           1
test_rust_native       1,295.6483 (1.92)     1,381.2643 (1.96)     1,326.4371 (1.93)      32.2405 (3.28)     1,318.3940 (1.92)      24.9539 (2.32)          1;1  0.7539 (0.52)          5           1
test_dicts_loop_io     3,110.1088 (4.61)     4,634.4713 (6.57)     3,594.2389 (5.22)     618.8147 (63.00)    3,330.8318 (4.85)     721.5654 (67.21)         1;0  0.2782 (0.19)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

Hashing:
```
--------------------------------------------------------------------------------------- benchmark: 6 tests ---------------------------------------------------------------------------------------
Name (time in ms)                     Min                Max               Mean            StdDev             Median               IQR            Outliers       OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_hash_rust_xxh3_64_batch       1.8703 (1.0)       2.5233 (1.0)       1.9286 (1.0)      0.0660 (1.0)       1.9113 (1.0)      0.0431 (1.0)         26;19  518.5155 (1.0)         335           1
test_hash_rust_xxh3_64_scalar      2.0558 (1.10)      2.9430 (1.17)      2.1173 (1.10)     0.0777 (1.18)      2.0944 (1.10)     0.0507 (1.18)        31;30  472.2909 (0.91)        468           1
test_hash_rust_xxh3_128            2.5731 (1.38)      3.1518 (1.25)      2.6416 (1.37)     0.0842 (1.28)      2.6147 (1.37)     0.0561 (1.30)        26;22  378.5631 (0.73)        306           1
test_hash_rust_blake3             38.1745 (20.41)    38.8501 (15.40)    38.3502 (19.89)    0.1699 (2.57)     38.3138 (20.05)    0.1325 (3.07)          4;3   26.0755 (0.05)         26           1
test_hash_rust_blake2             43.9950 (23.52)    44.8368 (17.77)    44.1726 (22.90)    0.2063 (3.13)     44.1266 (23.09)    0.1125 (2.61)          2;2   22.6385 (0.04)         23           1
test_hash_python_blake2b          66.8906 (35.76)    67.6681 (26.82)    67.1190 (34.80)    0.2485 (3.76)     66.9855 (35.05)    0.2902 (6.73)          3;0   14.8989 (0.03)         15           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

Mem Footprin (sorted from high to low):
```shell
Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_rust_structs at the high watermark

	 📦 Total memory allocated: 4.3GiB
	 📏 Total allocations: 22
	 📊 Histogram of allocation sizes: |▁▃█▁▃|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_dicts_batch at the high watermark

	 📦 Total memory allocated: 3.3GiB
	 📏 Total allocations: 19
	 📊 Histogram of allocation sizes: |   █▃|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_dicts_loop at the high watermark

	 📦 Total memory allocated: 3.3GiB
	 📏 Total allocations: 18
	 📊 Histogram of allocation sizes: |   █▄|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_arrow_tiny at the high watermark

	 📦 Total memory allocated: 742.0MiB
	 📏 Total allocations: 74
	 📊 Histogram of allocation sizes: |█▄  ▁|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_arrow_giant at the high watermark

	 📦 Total memory allocated: 64.9MiB
	 📏 Total allocations: 54
	 📊 Histogram of allocation sizes: |▃█   |

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_dicts_batched_stream at the high watermark

	 📦 Total memory allocated: 28.6MiB
	 📏 Total allocations: 17
	 📊 Histogram of allocation sizes: |█▆▆▃▁|

Allocation results for lib/dupekit/tests/bench/test_marshaling.py::test_arrow_small at the high watermark

	 📦 Total memory allocated: 1.2MiB
	 📏 Total allocations: 77
	 📊 Histogram of allocation sizes: |▁█▁▁▁|
```
___
**Statement of attribution:**
- This code was seeded from [nelson-liu/rbloom-gcs](https://github.com/nelson-liu/rbloom-gcs).
- Bloom filters were originally proposed in [(Bloom, 1970)](https://doi.org/10.1145/362686.362692). Furthermore, this  implementation makes use of a constant recommended by [(L'Ecuyer, 1999)](https://doi.org/10.1090/S0025-5718-99-00996-5) for  redistributing the entropy of a single hash over multiple integers using a  linear congruential generator.
