# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Downloads HF datasets for evaluation tasks and converts them to prompt/response JSONL format.

Each dataset follows a two-step pattern: a raw HuggingFace download, then a conversion to
prompt/response JSONL for log-prob evaluation or to dolma text format for decontamination.
"""

import dataclasses

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.transform.huggingface.dataset_to_eval import DatasetConversionConfig, OutputFormatOptions, hf_dataset_to_jsonl


@dataclasses.dataclass(frozen=True)
class EvalDataset:
    """A dataset for log prob evaluation. Steps point to data in prompt/response JSONL format."""

    org: str
    name: str
    steps: list[ArtifactStep[Artifact]]
    tags: list[str] = dataclasses.field(default_factory=list)


def eval_datasets() -> list[EvalDataset]:
    """Build and return all evaluation datasets as lazy artifact handles.

    Raw download handles are built once and shared across the derived steps that consume them.
    """
    # --- raw downloads ---
    mmlu_raw = hf_download(
        "raw/cais/mmlu",
        hf_id="cais/mmlu",
        revision="c30699e",
        urls_glob=["**/*.parquet", "*.md"],
        pin="raw/cais/mmluhf",
        version="2026.06.28",
    )
    boolq_raw = hf_download(
        "raw/google/boolq",
        hf_id="google/boolq",
        revision="35b264d",
        urls_glob=["**/*.parquet"],
        pin="raw/google/boolqhf",
        version="2026.06.28",
    )
    hellaswag_raw = hf_download(
        "raw/Rowan/hellaswag",
        hf_id="Rowan/hellaswag",
        revision="50441ce",
        urls_glob=["**/*.parquet"],
        pin="raw/Rowan/hellaswaghf",
        version="2026.06.28",
    )
    piqa_raw = hf_download(
        "raw/ybisk/piqa",
        hf_id="ybisk/piqa",
        revision="142c512",
        urls_glob=["**/*.parquet"],
        pin="raw/ybisk/piqahf",
        version="2026.06.28",
    )
    winogrande_raw = hf_download(
        "raw/allenai/winogrande",
        hf_id="allenai/winogrande",
        revision="ebf71e3",
        urls_glob=["winogrande_xl/**/*.parquet"],
        pin="raw/allenai/winograndehf",
        version="2026.06.28",
    )
    arc_raw = hf_download(
        "raw/allenai/ai2_arc",
        hf_id="allenai/ai2_arc",
        revision="210d026",
        urls_glob=["**/*.parquet", "*.md"],
        pin="raw/allenai/ai2_archf",
        version="2026.06.28",
    )
    openbookqa_raw = hf_download(
        "raw/allenai/openbookqa",
        hf_id="allenai/openbookqa",
        revision="388097e",
        urls_glob=["**/*.parquet", "*.md"],
        pin="raw/allenai/openbookqahf",
        version="2026.06.28",
    )
    humaneval_raw = hf_download(
        "raw/openai/openai_humaneval",
        hf_id="openai/openai_humaneval",
        revision="7dce605",
        urls_glob=["**/*.parquet", "*.md"],
        pin="gs://marin-us-central2/raw/openai/openai_humanevalhf",
        version="2026.06.28",
    )
    mbpp_raw = hf_download(
        "raw/google-research-datasets/mbpp",
        hf_id="google-research-datasets/mbpp",
        revision="4bb6404",
        urls_glob=["**/*.parquet", "*.md"],
        pin="raw/google-research-datasets/mbpphf",
        version="2026.06.28",
    )

    # --- derived evaluation steps ---
    mmlu_aux_eval = ArtifactStep(
        name="evaluation/mmlu-eval-aux",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["all"],
            splits=["auxiliary_train"],
            input_path=ctx.artifact_path(mmlu_raw),
            hf_path="cais/mmlu",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        ),
        deps=(mmlu_raw,),
    )

    mmlu_subject_eval = ArtifactStep(
        name="evaluation/mmlu-eval-subject",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["*"],
            splits=["dev", "validation"],
            input_path=ctx.artifact_path(mmlu_raw),
            hf_path="cais/mmlu",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
            exclude_subsets=["all", "auxiliary_train"],
        ),
        deps=(mmlu_raw,),
    )

    boolq_eval = ArtifactStep(
        name="evaluation/boolq-eval",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="google/boolq",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=ctx.artifact_path(boolq_raw),
            hf_path="google/boolq",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            answer_label_key="answer",
            answer_labels=[True, False],
            answer_text_ignore=True,
        ),
        deps=(boolq_raw,),
    )

    piqa_eval = ArtifactStep(
        name="evaluation/piqa",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="ybisk/piqa",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=ctx.artifact_path(piqa_raw),
            hf_path="ybisk/piqa",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="goal",
            options_keys=["sol1", "sol2"],
            answer_idx_key="label",
            answer_labels=["1", "2"],
        ),
        deps=(piqa_raw,),
    )

    winogrande_eval = ArtifactStep(
        name="evaluation/winogrande",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="allenai/winogrande",
            subsets=["default"],
            splits=["train", "validation"],
            input_path=ctx.artifact_path(winogrande_raw),
            hf_path="allenai/winogrande",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="sentence",
            options_keys=["option1", "option2"],
            answer_label_key="answer",
            answer_labels=["1", "2"],
        ),
        deps=(winogrande_raw,),
    )

    arc_easy_eval = ArtifactStep(
        name="evaluation/arc-easy",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="allenai/ai2_arc",
            subsets=["ARC-Easy"],
            splits=["train", "validation"],
            input_path=ctx.artifact_path(arc_raw),
            hf_path="allenai/ai2_arc",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices.text",
            answer_labels_key="choices.label",
            answer_label_key="answerKey",
        ),
        deps=(arc_raw,),
    )

    arc_challenge_eval = ArtifactStep(
        name="evaluation/arc-challenge",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="allenai/ai2_arc",
            subsets=["ARC-Challenge"],
            splits=["train", "validation"],
            input_path=ctx.artifact_path(arc_raw),
            hf_path="allenai/ai2_arc",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices.text",
            answer_labels_key="choices.label",
            answer_label_key="answerKey",
        ),
        deps=(arc_raw,),
    )

    openbookqa_eval = ArtifactStep(
        name="evaluation/openbookqa-eval",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="allenai/openbookqa",
            subsets=["main"],
            splits=["train", "validation"],
            input_path=ctx.artifact_path(openbookqa_raw),
            hf_path="allenai/openbookqa",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question_stem",
            options_key="choices.text",
            answer_label_key="answerKey",
            answer_labels_key="choices.label",
        ),
        deps=(openbookqa_raw,),
    )

    hellaswag_eval = ArtifactStep(
        name="evaluation/hellaswag-eval",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="Rowan/hellaswag",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=ctx.artifact_path(hellaswag_raw),
            hf_path="Rowan/hellaswag",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="ctx",
            options_key="endings",
            answer_labels=["A", "B", "C", "D"],
            answer_idx_key="label",
        ),
        deps=(hellaswag_raw,),
    )

    humaneval_eval = ArtifactStep(
        name="evaluation/humaneval-eval",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="openai/openai_humaneval",
            subsets=["*"],
            splits=["test"],
            input_path=ctx.artifact_path(humaneval_raw),
            hf_path="openai/openai_humaneval",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="prompt",
            answer_text_key="canonical_solution",
        ),
        deps=(humaneval_raw,),
    )

    mbpp_eval = ArtifactStep(
        name="evaluation/mbpp-eval",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="google-research-datasets/mbpp",
            subsets=["*"],
            splits=["train", "test", "validation"],
            input_path=f"{ctx.artifact_path(mbpp_raw)}/4bb6404/full",
            hf_path="google-research-datasets/mbpp",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="text",
            answer_text_key="code",
        ),
        deps=(mbpp_raw,),
    )

    return [
        # tags are used to group datasets together for averaging
        EvalDataset("cais", "mmlu", [mmlu_aux_eval, mmlu_subject_eval]),
        EvalDataset("google", "boolq", [boolq_eval], ["core"]),
        EvalDataset("Rowan", "hellaswag", [hellaswag_eval], ["core"]),
        EvalDataset("ybisk", "piqa", [piqa_eval], ["core"]),
        EvalDataset("allenai", "winogrande", [winogrande_eval], ["core"]),
        EvalDataset("allenai", "ai2_arc_easy", [arc_easy_eval], ["core", "arc"]),
        EvalDataset("allenai", "ai2_arc_challenge", [arc_challenge_eval], ["core", "arc"]),
        EvalDataset("allenai", "openbookqa", [openbookqa_eval], ["core"]),
        EvalDataset("openai", "openai_humaneval", [humaneval_eval]),
        EvalDataset("google-research-datasets", "mbpp", [mbpp_eval]),
    ]


def decontamination_datasets() -> list[ArtifactStep[Artifact]]:
    """Build and return datasets in dolma text format for decontamination pipelines."""
    mmlu_raw = hf_download(
        "raw/cais/mmlu",
        hf_id="cais/mmlu",
        revision="c30699e",
        urls_glob=["**/*.parquet", "*.md"],
        pin="raw/cais/mmluhf",
        version="2026.06.28",
    )

    mmlu_convert_dolma = ArtifactStep(
        name="decontamination/mmlu-dolma",
        version="2026.06.28",
        artifact_type=Artifact,
        run=hf_dataset_to_jsonl,
        build_config=lambda ctx: DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["all"],
            splits=["dev", "test", "validation"],
            input_path=ctx.artifact_path(mmlu_raw),
            hf_path="cais/mmlu",
            output_path=ctx.output_path,
            output_format=OutputFormatOptions("decontamination"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        ),
        deps=(mmlu_raw,),
    )

    return [mmlu_convert_dolma]


def extra_raw_downloads() -> dict[str, ArtifactStep[TokenizedCache]]:
    """Build and return standalone raw download handles for additional evaluation datasets."""
    return {
        "mmlu_pro_raw": hf_download(
            "raw/TIGER-Lab/MMLU-Pro",
            hf_id="TIGER-Lab/MMLU-Pro",
            revision="3373e0b",
            urls_glob=["**/*.parquet", "*.md"],
            pin="raw/TIGER-Lab/MMLU-Prohf",
            version="2026.06.28",
        ),
        "lingoly": hf_download(
            "raw/ambean/lingOly",
            hf_id="ambean/lingOly",
            revision="6aff4c2",
            version="2026.06.28",
        ),
        "gsm8k_raw": hf_download(
            "raw/gsm8k",
            hf_id="openai/gsm8k",
            revision="e53f048",
            urls_glob=["**/*.parquet", "*.md"],
            pin="raw/gsm8k/mainhf",
            version="2026.06.28",
        ),
        "math_raw": hf_download(
            "raw/hendrycks_math",
            hf_id="EleutherAI/hendrycks_math",
            revision="21a5633",
            urls_glob=["**/*.parquet", "*.md"],
            pin="raw/hendrycks/mathhf",
            version="2026.06.28",
        ),
        "truthful_qa_raw": hf_download(
            "raw/truthful_qa",
            hf_id="truthfulqa/truthful_qa",
            revision="741b827",
            urls_glob=["**/*.parquet", "*.md"],
            pin="raw/truthful_qa/multiple_choicehf",
            version="2026.06.28",
        ),
        "bbh_raw": hf_download(
            "raw/bbh",
            hf_id="SaylorTwift/bbh",
            revision="b5306be",
            urls_glob=["**/*.parquet", "*.md"],
            pin="raw/SaylorTwift/bbhhf",
            version="2026.06.28",
        ),
        "gpqa_raw": hf_download(
            "raw/gpqa",
            hf_id="Idavidrein/gpqa",
            revision="90b8e5b",
            urls_glob=["**/*.csv", "*.csv"],
            pin="raw/Idavidrein/gpqa",
            version="2026.06.28",
        ),
        "instruction_following_raw": hf_download(
            "raw/instruction_following_eval",
            hf_id="wis-k/instruction-following-eval",
            revision="5a5661c",
            urls_glob=["**/*.jsonl", "*.jsonl"],
            pin="raw/wis-k/instruction-following-evalhf",
            version="2026.06.28",
        ),
        "musr_raw": hf_download(
            "raw/musr",
            hf_id="WillHeld/MuSRDecontam",
            revision="39b4f56",
            urls_glob=["**/*.parquet", "*.parquet"],
            pin="raw/WillHeld/MuSRDecontamhf",
            version="2026.06.28",
        ),
        "winograd_wsc_raw": hf_download(
            "raw/winograd_wsc",
            hf_id="marcov/winograd_wsc_wsc273_promptsource",
            revision="63befd8",
            urls_glob=["**/*.parquet", "*.parquet"],
            pin="raw/marcov/winograd_wsc_wsc273_promptsourcehf",
            version="2026.06.28",
        ),
        "commonsense_qa_raw": hf_download(
            "raw/commonsense_qa",
            hf_id="tau/commonsense_qa",
            revision="94630fe",
            urls_glob=["**/*.parquet", "*.parquet"],
            pin="raw/tau/commonsense_qahf",
            version="2026.06.28",
        ),
        "lambada_openai_raw": hf_download(
            "raw/lambada_openai",
            hf_id="EleutherAI/lambada_openai",
            revision="879e19a",
            urls_glob=["**/*.jsonl", "*.jsonl"],
            pin="raw/EleutherAI/lambada_openaihf",
            version="2026.06.28",
        ),
        "piqa_baber_raw": hf_download(
            "raw/baber/piqa",
            hf_id="baber/piqa",
            revision="142f6d7",
            urls_glob=["**/*.parquet", "*.parquet"],
            pin="raw/baber/piqahf",
            version="2026.06.28",
        ),
    }
