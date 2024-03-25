import json
import logging

import fsspec
from pyarrow import parquet as pq

logger = logging.getLogger(__name__)

RPV2_CRAWLS = [
    "2014-15",
    "2014-23",
    "2014-35",
    "2014-41",
    "2014-42",
    "2014-49",
    "2014-52",
    "2015-14",
    "2015-22",
    "2015-27",
    "2015-32",
    "2015-35",
    "2015-40",
    "2015-48",
    "2016-07",
    "2016-18",
    "2016-22",
    "2016-26",
    "2016-30",
    "2016-36",
    "2016-40",
    "2016-44",
    "2016-50",
    "2017-04",
    "2017-09",
    "2017-17",
    "2017-22",
    "2017-26",
    "2017-30",
    "2017-34",
    "2017-39",
    "2017-43",
    "2017-47",
    "2017-51",
    "2018-05",
    "2018-09",
    "2018-13",
    "2018-17",
    "2018-22",
    "2018-26",
    "2018-30",
    "2018-34",
    "2018-39",
    "2018-43",
    "2018-47",
    "2018-51",
    "2019-04",
    "2019-09",
    "2019-13",
    "2019-18",
    "2019-22",
    "2019-26",
    "2019-30",
    "2019-35",
    "2019-39",
    "2019-43",
    "2019-47",
    "2019-51",
    "2020-05",
    "2020-10",
    "2020-16",
    "2020-24",
    "2020-29",
    "2020-34",
    "2020-40",
    "2020-45",
    "2020-50",
    "2021-04",
    "2021-10",
    "2021-17",
    "2021-21",
    "2021-25",
    "2021-31",
    "2021-39",
    "2021-43",
    "2021-49",
    "2022-05",
    "2022-21",
    "2022-27",
    "2022-33",
    "2022-40",
    "2022-49",
    "2023-06",
    "2023-14",
]
_URL_BASE = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
NUM_SHARDS = 5000
_LANGUAGES = ("en", "de", "fr", "es", "it")


def iterate_rpv2_file(snapshot, n, lang, part):
    base_tag = f"{snapshot}/{n:04d}/{lang}_{part}"
    qs_file = f"{_URL_BASE}/quality_signals/{base_tag}.signals.json.gz"
    dupe_file = f"{_URL_BASE}/duplicates/{base_tag}.duplicates.parquet"

    # Load duplicates
    try:
        with fsspec.open(dupe_file, "rb", compression="infer") as df:
            duplicates = set(pq.read_table(df, columns=["doc_id"], use_pandas_metadata=False)["doc_id"].to_pylist())
    except Exception as e:
        logger.exception(f"No duplicate ids found for {base_tag}: {e}")
        duplicates = set()

    try:
        with fsspec.open(qs_file, "r", compression="infer", encoding="utf-8") as qf:
            for row, line in enumerate(qf):
                doc_id = f"{base_tag}.json.gz/{row}"
                try:
                    qs = json.loads(line)
                    is_duplicate = doc_id in duplicates
                    yield doc_id, {**qs, "is_duplicate": is_duplicate}
                except Exception as e:
                    logger.exception(f"Error processing {doc_id}: {e}")
    except Exception as e:
        logger.exception(f"Error processing {qs_file}: {e}")


def all_urls():
    """Returns all the quality and dupe urls from together.xyz"""
    for crawl in RPV2_CRAWLS:
        for lang in _LANGUAGES[:1]:  # just english for now
            for part in ("head", "middle"):
                for n in range(NUM_SHARDS):
                    base_tag = f"{crawl}/{n:04d}/{lang}_{part}"
                    qs_file = f"{_URL_BASE}/quality_signals/{base_tag}.signals.json.gz"
                    dupe_file = f"{_URL_BASE}/duplicates/{base_tag}.duplicates.parquet"
                    yield qs_file
                    yield dupe_file


if __name__ == "__main__":
    import csv
    out = csv.writer(open("all_urls.csv", "w"))
    out.writerow(["url"])
    for url in all_urls():
        out.writerow([url])





