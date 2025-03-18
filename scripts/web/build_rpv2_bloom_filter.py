import hashlib
import os

import fsspec
import rbloom
import tqdm
import tempfile

from marin.web.rpv2 import NUM_SHARDS, RPV2_CRAWLS, iterate_rpv2_file

# Create a bloom filter from a shard of RPV2


# Python's built-in hash function is not deterministic across invocations
def hash_func(s):
    # can't overflow C long
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16)
    return h & 0x7FFFFFFFFFFFFFFF


SLICE_SIZE = 50


def chunk_range(n, chunk_size):
    for i in range(0, n, chunk_size):
        yield range(i, min(i + chunk_size, n))


ALL_SLICES = list(chunk_range(NUM_SHARDS, SLICE_SIZE))


def mk_bloom_for_shard_slice(out_path, crawl, part, lang, shard_slice=range(NUM_SHARDS)):
    # NB: this must be 0 so we can make the bloom filter sizes the same so we can union them
    num_files_per_shard_mid = sum(1 for _ in iterate_rpv2_file(crawl, 0, "en", "middle"))
    num_files_per_shard_head = sum(1 for _ in iterate_rpv2_file(crawl, 0, "en", "head"))
    approx_total_items = (num_files_per_shard_head + num_files_per_shard_mid) * NUM_SHARDS

    bloom_filter = rbloom.Bloom(approx_total_items, 0.01, hash_func)
    total_mb = bloom_filter.size_in_bits / 8.0 / 1024 / 1024
    print(f"Creating bloom filter for {crawl} {lang} {part} {shard_slice} with {total_mb:.2f} MB")
    count = 0
    # don't use pbar if we're running inside a cloud function
    if "FUNCTION_TARGET" in os.environ:
        pbar = shard_slice
    else:
        pbar = tqdm.tqdm(shard_slice, desc=f"Processing {crawl} {part} {lang}", position=0)
    for n in pbar:
        for _, qs in iterate_rpv2_file(crawl, n, lang, part):
            url = qs["metadata"]["url"]
            bloom_filter.add(url)
            count += 1
        approx = bloom_filter.approx_items
        error = abs(count - approx) / count
        if "FUNCTION_TARGET" not in os.environ:
            pbar.set_postfix(count=count, error=error)

    print(f"Finished {crawl} {lang} {part} {shard_slice} with {count} items, approx {approx} error {error:.2f}")

    # in GCF this doubles memory usage, so instead we write to a temp
    # bytes = bloom_filter.save_bytes()
    # with fsspec.open(out_path, "wb") as f:
    #     f.write(bytes)
    save_bloom_filter_fsspec(bloom_filter, out_path)
    print(f"Finished {crawl} {lang} {part} {shard_slice}")


def save_bloom_filter_fsspec(bloom_filter, out_path):
    with tempfile.NamedTemporaryFile() as f:
        f.close()
        bloom_filter.save(f.name)
        del bloom_filter
        fs = fsspec.filesystem("gs")
        fs.put(f.name, out_path)
        print(f"Saved bloom filter to {out_path}")


def load_bloom(path):
    with fsspec.open(path, "rb") as f:
        bytes = f.read()
    bloom_filter = rbloom.Bloom.load_bytes(bytes, hash_func)
    return bloom_filter


def union_blooms(out_path, paths):
    # use reduce to avoid load
    if len(paths) == 0:
        raise ValueError("No paths provided")

    bloom_filter = load_bloom(paths[0])
    approx_total_items = bloom_filter.approx_items
    for path in paths[1:]:
        other = load_bloom(path)
        bloom_filter.update(other)
        approx_total_items += other.approx_items
        del other
        # run gc to avoid memory issues
        import gc

        gc.collect()

    compare = bloom_filter.approx_items
    error = abs(approx_total_items - compare) / approx_total_items
    print(f"Unioned {len(paths)} blooms with approx {compare} items, approx sum {approx_total_items} error {error:.2f}")
    save_bloom_filter_fsspec(bloom_filter, out_path)


def all_paths_for_crawl(base_path, crawl, lang):
    for slice in ALL_SLICES:
        for part in ["head", "middle"]:
            out_path = os.path.join(base_path, f"tmp/{lang}_{crawl}_{part}_{slice.start:04d}-{slice.stop:04d}.bloom")
            yield (part, slice, out_path)


def _fsspec_exists(path):
    fs = fsspec.filesystem("gs")
    return fs.exists(path)


if __name__ == "__main__":
    base_path = "gs://levanter-data/marin/v0/url_blooms/"
    lang = "en"
    for crawl in RPV2_CRAWLS:
        paths = []
        for part, slice, out_path in all_paths_for_crawl(base_path, crawl, lang):
            paths.append(out_path)
            mk_bloom_for_shard_slice(out_path, crawl, part, lang, slice)

        out_path = os.path.join(base_path, f"{lang}_{crawl}.bloom")
        union_blooms(out_path, paths)
