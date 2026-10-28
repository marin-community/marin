# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NUPA's sampled full protocol and separately pinned NUPA5K convenience panel."""

from functools import partial

from taskcompendium.evaluation import EvaluationProtocol, Metric, MetricKind, Sampling, TrialPolicy
from taskcompendium.importers.evaluation_nupa import convert_nupa, nupa_item, read_nupa_panel, read_nupa_sample
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse
from verifyit.adapters.evalchemy_math import CHAT_REQUEST_STOPS

from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

NUPA_REVISION = "01e3831ec00dfd618a77d9f6fe7fc0d327ad16d7"
PANEL_SHA256 = "27f122972d76dc2a1170a3d105a9856b55beedabe037f697c1a1aa2baeb67036"
PANEL_SOURCE_COMMIT = "623134de90723e7f189036da005067f36817b6e7"


def sources() -> list[RlDataSource[EvalDataSet]]:
    result = []
    for name, rows, reader in (("NUPA-Loose", 238926, read_nupa_sample), ("NUPA5K-Loose", 5000, read_nupa_panel)):
        inputs = (
            {
                "panel": UrlSource(
                    f"https://raw.githubusercontent.com/marin-community/marin/{PANEL_SOURCE_COMMIT}/"
                    "vendor/evalchemy/eval/chat_benchmarks/NUPA5K-Loose/data/nupa5k_manifest.jsonl",
                    PANEL_SHA256,
                    "nupa5k_manifest.jsonl",
                    SourceFormat.JSONL,
                )
            }
            if name == "NUPA5K-Loose"
            else {}
        )
        result.append(
            RlDataSource(
                info=SourceInfo(
                    id="Evalchemy:" + name,
                    title=name,
                    origin="Evalchemy",
                    family="numeric-representation",
                    tags=("eval", "source:evalchemy", "single-turn", "benchmark", "family:numeric-representation"),
                    count=rows,
                ),
                config=EvalDataSet(
                    recipe=CurationRecipe(
                        name="evalchemy-" + name.lower(),
                        source=HfSource(
                            "HaotongYang/NUPA_text", NUPA_REVISION, ("test.json",), SourceFormat.JSON, read=reader
                        ),
                        inputs=inputs,
                        convert=partial(convert_nupa, benchmark=name),
                        version="1",
                        intended_use=IntendedUse.EVAL,
                    ),
                    protocol=EvaluationProtocol(
                        benchmark=name,
                        version="maintained-nupa-loose-v2",
                        operation="chat_generation",
                        sampling=Sampling(
                            temperature=0,
                            top_p=1,
                            top_k=None,
                            max_tokens=256,
                            stop=CHAT_REQUEST_STOPS,
                            chat_template_kwargs={},
                        ),
                        trials=TrialPolicy(count=1, seed=None),
                        metrics=tuple(
                            Metric(
                                name=metric,
                                kind=(
                                    MetricKind.BINARY
                                    if metric in ("exact_match", "format_valid_rate", "no_answer_rate")
                                    else MetricKind.CONTINUOUS
                                ),
                                higher_is_better=metric not in ("dlength", "no_answer_rate"),
                            )
                            for metric in (
                                "exact_match",
                                "digit_match",
                                "dlength",
                                "format_valid_rate",
                                "no_answer_rate",
                            )
                        ),
                        primary_metric="exact_match",
                        reducer=(
                            "nupa_fixed_panel_mean_v1" if name == "NUPA5K-Loose" else "nupa_stratified_sample_mean_v1"
                        ),
                        expected_population=rows,
                        context_window=73664 if name == "NUPA5K-Loose" else None,
                    ),
                    item_builder=nupa_item,
                ),
                pipeline=curate,
            )
        )
    return result
