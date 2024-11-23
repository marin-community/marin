import os

import fsspec
import gcsfs
import ray

from marin.utils import fsspec_glob


@ray.remote
def test():
    url = "gs://marin-us-central2/documents/stackexchange-qa-vote-geq-5-rm-duplicate-71a5cd"
    double_wildcard_pattern = "**/*.jsonl.gz"
    single_wildcard_pattern = "*.jsonl.gz"

    print(f"fsspec version: {fsspec.__version__}")
    print(f"gcsfs version: {gcsfs.__version__}")

    double_wildcard_files = fsspec_glob(os.path.join(url, double_wildcard_pattern))
    single_wildcard_files = fsspec_glob(os.path.join(url, single_wildcard_pattern))

    assert len(single_wildcard_files) > 0, "No files found!"

    # ERRORS HERE!
    assert len(double_wildcard_files) > 0, "No files found!"


if __name__ == "__main__":
    ray.get(test.remote())
