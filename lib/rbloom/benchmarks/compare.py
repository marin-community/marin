import struct
import timeit

import pybloomfilter
import bloom_filter2
import flor
import pybloom
import rbloom


NUM_ITEMS = 10_000_000


def run(ty):
    bf = ty(NUM_ITEMS, 0.01)

    for i in range(NUM_ITEMS):
        bf.add(i + 0.5)  # floats because ints are hashed as themselves

    for i in range(NUM_ITEMS):
        if i + 0.5 not in bf:
            raise ValueError("Should be no false negatives")


def run_bytes(ty):
    bf = ty(NUM_ITEMS, 0.01)

    for i in range(NUM_ITEMS):
        bf.add(struct.pack("d", i + 0.5))

    for i in range(NUM_ITEMS):
        if struct.pack("d", i + 0.5) not in bf:
            raise ValueError("Should be no false negatives")


types = {
    "rbloom": rbloom.Bloom,
    "pybloomfiltermmap3": pybloomfilter.BloomFilter,
    "pybloom3": pybloom.BloomFilter,
    "flor": flor.BloomFilter,
    "bloomfilter2": bloom_filter2.BloomFilter,
}


def main():
    for name, ty in types.items():
        print(f"Running {name}")
        try:
            results = timeit.repeat(lambda: run(ty), number=1, repeat=5)
            extras = ""
        except Exception as e:
            results = timeit.repeat(lambda: run_bytes(ty), number=1, repeat=5)
            extras = f" (via bytes because {e})"
        avg = sum(results) / len(results)
        print(f"  {avg:6.2f}s{extras}")


if __name__ == "__main__":
    main()
