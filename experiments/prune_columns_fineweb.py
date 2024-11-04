"""Keep only specified columns from FineWeb parquet files."""

from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.schemas.web.convert import ResiliparseConfig
from operations.download.huggingface.stream_remove_columns import DatasetConfig, prune_hf_dataset
from scripts.fineweb.process_parquet_fw import ParquetFWConfig, process_fw_dump


def filter_fineweb_parquet():
    # FineWeb subsets to keep, complete "main" subset list present at: https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data
    # sample subsets present at: https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample, we exclude these
    # and "default" subset
    subsets = [
        "data/CC-MAIN-2013-20",
        "data/CC-MAIN-2013-48",
        "data/CC-MAIN-2014-10",
        "data/CC-MAIN-2014-15",
        "data/CC-MAIN-2014-23",
        "data/CC-MAIN-2014-35",
        "data/CC-MAIN-2014-41",
        "data/CC-MAIN-2014-42",
        "data/CC-MAIN-2014-49",
        "data/CC-MAIN-2014-52",
        "data/CC-MAIN-2015-06",
        "data/CC-MAIN-2015-11",
        "data/CC-MAIN-2015-14",
        "data/CC-MAIN-2015-18",
        "data/CC-MAIN-2015-22",
        "data/CC-MAIN-2015-27",
        "data/CC-MAIN-2015-32",
        "data/CC-MAIN-2015-35",
        "data/CC-MAIN-2015-40",
        "data/CC-MAIN-2015-48",
        "data/CC-MAIN-2016-07",
        "data/CC-MAIN-2016-18",
        "data/CC-MAIN-2016-22",
        "data/CC-MAIN-2016-26",
        "data/CC-MAIN-2016-30",
        "data/CC-MAIN-2016-36",
        "data/CC-MAIN-2016-40",
        "data/CC-MAIN-2016-44",
        "data/CC-MAIN-2016-50",
        "data/CC-MAIN-2017-04",
        "data/CC-MAIN-2017-09",
        "data/CC-MAIN-2017-13",
        "data/CC-MAIN-2017-17",
        "data/CC-MAIN-2017-22",
        "data/CC-MAIN-2017-26",
        "data/CC-MAIN-2017-30",
        "data/CC-MAIN-2017-34",
        "data/CC-MAIN-2017-39",
        "data/CC-MAIN-2017-43",
        "data/CC-MAIN-2017-47",
        "data/CC-MAIN-2017-51",
        "data/CC-MAIN-2018-05",
        "data/CC-MAIN-2018-09",
        "data/CC-MAIN-2018-13",
        "data/CC-MAIN-2018-17",
        "data/CC-MAIN-2018-22",
        "data/CC-MAIN-2018-26",
        "data/CC-MAIN-2018-30",
        "data/CC-MAIN-2018-34",
        "data/CC-MAIN-2018-39",
        "data/CC-MAIN-2018-43",
        "data/CC-MAIN-2018-47",
        "data/CC-MAIN-2018-51",
        "data/CC-MAIN-2019-04",
        "data/CC-MAIN-2019-09",
        "data/CC-MAIN-2019-13",
        "data/CC-MAIN-2019-18",
        "data/CC-MAIN-2019-22",
        "data/CC-MAIN-2019-26",
        "data/CC-MAIN-2019-30",
        "data/CC-MAIN-2019-35",
        "data/CC-MAIN-2019-39",
        "data/CC-MAIN-2019-43",
        "data/CC-MAIN-2019-47",
        "data/CC-MAIN-2019-51",
        "data/CC-MAIN-2020-05",
        "data/CC-MAIN-2020-10",
        "data/CC-MAIN-2020-16",
        "data/CC-MAIN-2020-24",
        "data/CC-MAIN-2020-29",
        "data/CC-MAIN-2020-34",
        "data/CC-MAIN-2020-40",
        "data/CC-MAIN-2020-45",
        "data/CC-MAIN-2020-50",
        "data/CC-MAIN-2021-04",
        "data/CC-MAIN-2021-10",
        "data/CC-MAIN-2021-17",
        "data/CC-MAIN-2021-21",
        "data/CC-MAIN-2021-25",
        "data/CC-MAIN-2021-31",
        "data/CC-MAIN-2021-39",
        "data/CC-MAIN-2021-43",
        "data/CC-MAIN-2021-49",
        "data/CC-MAIN-2022-05",
        "data/CC-MAIN-2022-21",
        "data/CC-MAIN-2022-27",
        "data/CC-MAIN-2022-33",
        "data/CC-MAIN-2022-40",
        "data/CC-MAIN-2022-49",
        "data/CC-MAIN-2023-06",
        "data/CC-MAIN-2023-14",
        "data/CC-MAIN-2023-23",
        "data/CC-MAIN-2023-40",
        "data/CC-MAIN-2023-50",
        "data/CC-MAIN-2024-10",
        "data/CC-MAIN-2024-18",
    ]

    filtered_fineweb = ExecutorStep(
        name="raw/fineweb-urls",
        fn=prune_hf_dataset,
        config=DatasetConfig(
            hf_repo_id="HuggingFaceFW/fineweb",
            hf_revision=versioned("main"),
            hf_paths=subsets,
            output_path=this_output_path(),
            keep_columns=[
                "id",
                "url",
                "file_path",
                "language_score",
                "token_count",
            ],
        ),
    )

    transform_resiliparse_preserve_formatting = ExecutorStep(
        name="documents/fineweb-urls-small-resiliparse-preserve-formatting",
        fn=process_fw_dump,
        config=ParquetFWConfig(
            input_path=output_path_of(filtered_fineweb),
            cc_dumps=versioned(["CC-MAIN-2024-18"]),
            md_output_path=this_output_path("md"),
            text_output_path=this_output_path("text"),
            html_output_path=this_output_path("html"),
            extract_method=versioned("resiliparse"),
            config=ResiliparseConfig(
                preserve_formatting=versioned(True),
                main_content=versioned(True),
                links=versioned(False),
            ),
        ),
    )

    return [
        filtered_fineweb,
        transform_resiliparse_preserve_formatting,
    ]


if __name__ == "__main__":
    executor_main(steps=filter_fineweb_parquet())
