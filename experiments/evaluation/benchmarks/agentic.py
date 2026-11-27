# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic benchmarks lowered from Harbor task packages.

Each dataset pins the packages its Harbor policy ran; the agent and its token limits form the
protocol. Refresh a dataset's image lock after changing its source::

    uv run python -m experiments.evaluation.benchmarks.agentic lock <lock>
"""

import argparse
import json
import logging
from dataclasses import replace

from taskcompendium.evaluation.spec import MiniSweAgent, PiAgent, TerminusAgent
from taskcompendium.importers.harbor.package import JobSettings

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.harbor import (
    IMAGE_LOCKS,
    PI_MAX_OUTPUT_TOKENS,
    TMUX_INSTALL,
    HarborDataset,
    PackageCopy,
    PackageEdit,
    github_packages,
    harbor_source,
    huggingface_packages,
    lock_images,
    pi_install,
)
from experiments.evaluation.benchmarks.harbor_judges import bixbench_answer, financeagent_answer, simpleqa_answer
from experiments.evaluation.benchmarks.swebench_pro import (
    SWEBENCH_PRO_DATASETS,
    SWEBENCH_PRO_JOB,
    SWEBENCH_PRO_MAX_INPUT_TOKENS,
    SWEBENCH_PRO_MAX_OUTPUT_TOKENS,
)
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.post_training.task_curation.harbor_packages import package_rows
from experiments.post_training.task_curation.pipeline import HarborHubSource
from experiments.post_training.task_curation.source import RlDataSource

logger = logging.getLogger(__name__)

VERSION = "2026.11.17.2"
# The version of the datasets whose rule renders a judge spec into each task. Their recipe identity
# does not cover the rendered judge, so a change to it bumps this version.
JUDGED_VERSION = "2026.11.23"
# SWE-bench Multilingual's version; a change to how its package edits run or grade the agent bumps it.
SWEBENCH_MULTILINGUAL_VERSION = "2026.11.26"
# Terminal-Bench's build-pov-ray allows its agent and its verifier 12,000 seconds each.
TERMINAL_BENCH_ATTEMPT_TIMEOUT = 8 * 60 * 60

TERMINAL_BENCH_2 = HarborDataset(
    recipe="terminal-bench-2.0",
    benchmark="terminal-bench-2.0",
    title="Terminal-Bench 2.0",
    source=huggingface_packages("DCAgent2/terminal_bench_2", "693231ec029249e7c91ed2e414bcc9c45d7cd879"),
    version=VERSION,
    count=89,
    attempt_timeout=TERMINAL_BENCH_ATTEMPT_TIMEOUT,
)
SWEBENCH_VERIFIED = HarborDataset(
    recipe="swebench-verified",
    benchmark="swebench-verified",
    title="SWE-bench Verified",
    source=huggingface_packages("DCAgent/swebench-verified", "0e4345c345395a3371f0b79d891a59fdbf74e9a8"),
    version=VERSION,
    count=500,
)
SWEBENCH_VERIFIED_100 = HarborDataset(
    recipe="swebench-verified-100",
    benchmark="swebench-verified-100",
    title="SWE-bench Verified, random 100",
    source=huggingface_packages(
        "DCAgent2/swebench-verified-random-100-folders", "0c553bd6d05d451907afb8db31dba84759ccc993"
    ),
    version=VERSION,
    count=100,
    images=SWEBENCH_VERIFIED.recipe,
)
OT_TBLITE = HarborDataset(
    recipe="ot-tblite",
    benchmark="ot-tblite",
    title="OpenThoughts-TBLite",
    source=huggingface_packages("DCAgent/dev_set_v2", "377118ff3031c934f5a647ae2c425eb74eef3b21"),
    version=VERSION,
    count=100,
)
BFCL_PARITY = HarborDataset(
    recipe="bfcl-parity",
    benchmark="bfcl-parity",
    title="BFCL parity",
    source=huggingface_packages("DCAgent2/bfcl-parity", "c43075f07c969fa30658d28f7f48c3bc9f2b02b2"),
    version=VERSION,
    count=123,
)
GAIA = HarborDataset(
    recipe="gaia",
    benchmark="gaia",
    title="GAIA",
    source=huggingface_packages("DCAgent/gaia_127", "55a2fa83dfaaba08f95e6a8f07fb2edfa4353856"),
    version=VERSION,
    count=127,
)
AIDER_POLYGLOT = HarborDataset(
    recipe="aider-polyglot",
    benchmark="aider-polyglot",
    title="Aider polyglot",
    source=huggingface_packages("DCAgent2/aider_polyglot", "6d5c1dcdc69752027347bc25b524b70b771404c7"),
    version=VERSION,
    count=225,
)
OPENAI_SDK = "openai==2.24.0"
# FinanceAgent's retrieve_information tool calls its model through LiteLLM, which has had compromised
# PyPI releases. These edits install the OpenAI SDK in LiteLLM's place and send the tool's Chat Completions
# request through it. The tool still reads MODEL_FOR_TOOLS, an OpenAI model with or without LiteLLM's
# "openai/" prefix, and MODEL_API_KEY, and prints the same JSON.
FINANCEAGENT_EDITS = (
    PackageEdit("environment/Dockerfile", "    litellm==1.55.3\n", f"    {OPENAI_SDK}\n"),
    PackageEdit(
        "environment/retrieve_information.py",
        '    """Call LLM using litellm."""\n    import litellm\n',
        '    """Call the tools model through the OpenAI SDK."""\n    from openai import OpenAI\n',
    ),
    PackageEdit(
        "environment/retrieve_information.py",
        "    response = litellm.completion(\n        model=model_name,\n",
        "    response = OpenAI(api_key=api_key).chat.completions.create(\n"
        '        model=model_name.removeprefix("openai/"),\n',
    ),
    PackageEdit(
        "environment/retrieve_information.py",
        '        messages=[{"role": "user", "content": prompt}],\n        api_key=api_key,\n',
        '        messages=[{"role": "user", "content": prompt}],\n',
    ),
    PackageEdit(
        "environment/retrieve_information.py",
        "        # Call LLM (need to import litellm)\n",
        "        # Call LLM (need to import openai)\n",
    ),
    PackageEdit(
        "environment/retrieve_information.py",
        'raise ImportError("litellm is not installed. Please install it with: pip install litellm")',
        'raise ImportError("openai is not installed. Please install it with: pip install openai")',
    ),
)
# Its tools need SEC EDGAR and SerpAPI keys and its judge needs an OpenAI key, which an evaluation
# launch must supply as task secrets.
FINANCEAGENT = HarborDataset(
    recipe="financeagent",
    benchmark="financeagent",
    title="FinanceAgent",
    source=huggingface_packages("DCAgent/financeagent_terminal", "86f788bc2fe544de818f26a71e7f91c247dd39e9"),
    version=JUDGED_VERSION,
    count=50,
    rule=financeagent_answer,
    requires_judge=True,
    edits=FINANCEAGENT_EDITS,
)
MEDAGENTBENCH = HarborDataset(
    recipe="medagentbench",
    benchmark="medagentbench",
    title="MedAgentBench",
    source=huggingface_packages("DCAgent/medagentbench", "d25e5bd284ebdd812c21a6430598e9d7a73589ec"),
    version=VERSION,
    count=300,
)
AIME_AGENT = HarborDataset(
    recipe="aime-agent",
    benchmark="aime-agent",
    title="AIME, agentic",
    # Harbor registry aime@1.0.
    source=github_packages(
        "laude-institute/harbor-datasets",
        "414014c23ce4d32128073d12b057252c918cccf4",
        "96f339b3eb3543610db9b86ce2b0395f874b829cb2e720945ecd575d3776f9c3",
        root="datasets/aime",
    ),
    version=VERSION,
    count=60,
)
SIMPLEQA_AGENT = HarborDataset(
    recipe="simpleqa-agent",
    benchmark="simpleqa-agent",
    title="SimpleQA, agentic",
    # Harbor registry simpleqa@1.1 in marin-community/harbor@ec23913e's registry.json.
    source=github_packages(
        "karansag/harbor-datasets",
        "3936c3e71d2e50b190e105c18b877dce7917cb38",
        "89165b1b7221a6c588880ffd761eb959008b468f25f7864c00a73777b85a30c0",
        root="datasets/simpleqa",
    ),
    version=JUDGED_VERSION,
    count=4326,
    rule=simpleqa_answer,
    requires_judge=True,
)
TERMINAL_BENCH_2_1 = HarborDataset(
    recipe="terminal-bench-2.1",
    benchmark="terminal-bench-2.1",
    title="Terminal-Bench 2.1",
    # Harbor Hub terminal-bench/terminal-bench-2-1, ref 6.
    source=HarborHubSource(
        "terminal-bench/terminal-bench-2-1",
        "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a",
        read=package_rows,
    ),
    version=VERSION,
    count=89,
    attempt_timeout=TERMINAL_BENCH_ATTEMPT_TIMEOUT,
)
LONG_MAX_INPUT_TOKENS = 65536
LONG_MAX_OUTPUT_TOKENS = 16384
"""The context and reply limits of the Terminal-Bench 2.1 and BixBench policies."""
# The 2026-09-25 evaluation campaign's Harbor policies, which ran datasets of marin-community/harbor@38742b0f's
# registry.json. Each gives the agent 1,800 seconds and the verifier 300, whatever the package sets.
CAMPAIGN_JOB = JobSettings(agent_timeout=1800, verifier_timeout=300)
CAMPAIGN_TRIALS = 3
"""The trials of the campaign's Terminal-Bench policies."""
CAMPAIGN_PROTOCOL = "campaign-2026-09-25"
"""The protocol of a campaign policy whose agent is also the benchmark's default protocol."""
CAMPAIGN_MAX_INPUT_TOKENS = 32768
CAMPAIGN_PI = PiAgent(version="0.87.0", setup_timeout=360)
TERMINAL_BENCH_2_1_3TRIAL = replace(
    TERMINAL_BENCH_2_1,
    recipe="terminal-bench-2.1-3trial",
    title="Terminal-Bench 2.1, three trials",
    images=TERMINAL_BENCH_2_1.recipe,
)
OT_TBLITE_3TRIAL = HarborDataset(
    recipe="ot-tblite-2.0-3trial",
    benchmark="ot-tblite",
    title="OpenThoughts-TBLite 2.0, three trials",
    # Registry openthoughts-tblite@2.0. OT_TBLITE's copy of these packages adds stray README, .DS_Store and Git files.
    source=github_packages(
        "open-thoughts/OpenThoughts-TBLite",
        "f075e463472c7790b85793b392dff1fff20cc0e3",
        "f0bb3760be900930f1bfe93d21129bda9a1d8354c9cba5070b330a72f6f4e840",
        root="",
    ),
    version=VERSION,
    count=100,
    images=OT_TBLITE.recipe,
)
DS_1000_MINI_200_PACKAGES = frozenset(
    """
    3 6 11 25 27 30 32 44 46 48 57 65 67 71 73 80 81 89 93 94 95 99 103 104 112 114 117 127 141 142 146 156
    159 161 163 166 167 175 196 203 214 216 217 220 223 224 225 228 233 234 238 250 252 257 269 270 271 273
    274 276 281 284 296 300 322 323 332 344 348 352 363 367 370 373 379 387 388 389 390 394 405 408 410 429
    432 438 445 459 464 469 470 473 479 505 511 517 521 541 546 549 551 558 565 566 570 574 580 591 597 598
    603 604 610 616 618 623 633 642 643 650 654 655 656 658 663 665 671 677 686 692 696 698 699 701 704 714
    718 721 733 735 736 738 739 746 747 754 758 759 762 764 768 773 777 781 786 787 791 794 811 824 825 826
    828 830 841 849 854 863 867 875 881 882 887 890 899 906 907 908 913 925 930 931 936 947 954 955 956 958
    963 964 970 974 978 982 983 986 990 992 994 998
    """.split()
)
"""Registry ds-1000@mini-200: a seeded random 200 of DS-1000's 1,000 problems."""
DS_1000_MINI_200 = HarborDataset(
    recipe="ds-1000-mini-200",
    benchmark="ds-1000",
    title="DS-1000, random 200",
    source=github_packages(
        "penfever/harbor-datasets",
        "cedf53d7193626b466c856b5021c4056106bfd84",
        "0f5f1022b497ebf9a70b7ab4803a1e77048cf2bca43081a1c9e08ede0ce29872",
        root="datasets/ds1000",
        packages=DS_1000_MINI_200_PACKAGES,
    ),
    version="2026.11.25",
    count=200,
)
BFCL_PARITY_PI = HarborDataset(
    recipe="bfcl-parity-1.0-pi",
    benchmark="bfcl-parity",
    title="BFCL parity 1.0, Pi",
    # Registry bfcl_parity@1.0, an earlier revision of BFCL_PARITY's packages.
    source=github_packages(
        "laude-institute/harbor-datasets",
        "6bedd7878dc5d6f3456b4d80b781eb3c2d84f262",
        "f184ca6f3b79a2e0a0c5491e18ede5e3a381b0492a8855e7804c51c91852a780",
        root="datasets/bfcl_parity",
    ),
    version=VERSION,
    count=123,
    images=BFCL_PARITY.recipe,
)
BIXBENCH = HarborDataset(
    recipe="bixbench",
    benchmark="bixbench",
    title="BixBench",
    # Registry bixbench-cli-marin-corrections@3.
    source=github_packages(
        "penfever/harbor-datasets",
        "454d04b5427fd90de78d7bec9dfe57561a21ff03",
        "0a5edab87016e32c6760f813cca681d8215ac9e48f271608516eb61656766658",
        root="datasets/bixbench-cli-marin-corrections-1",
    ),
    version=JUDGED_VERSION,
    count=205,
    rule=bixbench_answer,
    requires_judge=True,
)
SWEBENCH_VERIFIED_MINI_SWE = HarborDataset(
    recipe="swebench-verified-1.0-mini-swe",
    benchmark="swebench-verified",
    title="SWE-bench Verified 1.0, mini-swe-agent",
    # Registry swebench-verified@1.0. These verifiers run `git clean -fd` before the tests; SWEBENCH_VERIFIED's
    # copy of the packages removes only the untracked files the test patch touches.
    source=github_packages(
        "laude-institute/harbor-datasets",
        "86723674f04e4209ac479d0fb75d9d9f44b4377e",
        "8fbd43bada1b6a00da8eb4ebf3d0641fed904d113dfb0545584be8b75098d7c2",
        root="datasets/swebench-verified",
    ),
    version=VERSION,
    count=500,
    images=SWEBENCH_VERIFIED.recipe,
)
SWEBENCH_WORKSPACE = "/testbed"
SWEBENCH_GIT = f"git -c safe.directory={SWEBENCH_WORKSPACE}"
SWEBENCH_WORKSPACE_TREE = "/opt/marin/workspace-tree"
"""The git tree of the image's workspace, untracked files included, that the agent's changes are diffed against."""
SWEBENCH_ARTIFACTS = "/logs/artifacts"
SWEBENCH_PATCH = f"{SWEBENCH_ARTIFACTS}/model.patch"
SWEBENCH_ENVIRONMENT = "[environment]\nbuild_timeout_sec = 1800.0\ncpus = 4\nmemory = '8G'\nstorage = '10G'\n"
SWEBENCH_LARGE_FILE_BYTES = 1024 * 1024
"""New files above this size, such as core dumps and build output, stay out of the agent's patch."""
SWEBENCH_COLLECT = (
    f'set -e; cd {SWEBENCH_WORKSPACE}; git="{SWEBENCH_GIT} -c core.quotePath=false"; '
    f"tree=$(cat {SWEBENCH_WORKSPACE_TREE}); $git add -A; "
    '$git diff --cached --name-only --diff-filter=A "$tree" | while IFS= read -r file; do '
    f'if [ -f "$file" ] && [ "$(wc -c < "$file")" -gt {SWEBENCH_LARGE_FILE_BYTES} ]; then '
    '$git rm -q --cached -- "$file"; fi; '
    "done; "
    f'mkdir -p {SWEBENCH_ARTIFACTS}; $git diff --cached --binary "$tree" > {SWEBENCH_PATCH}'
)
# A patch taken against the verifier's own starting tree fails to apply only when grading itself fails, so the
# script then exits without a reward, which records an infrastructure error rather than a zero.
SWEBENCH_APPLY = (
    f"if [ -s {SWEBENCH_PATCH} ]; then\n"
    f'    echo "Applying the agent\'s patch, $(wc -c < {SWEBENCH_PATCH}) bytes"\n'
    f"    {SWEBENCH_GIT} -C {SWEBENCH_WORKSPACE} apply --binary {SWEBENCH_PATCH} || exit 1\n"
    "fi\n"
)
# The agent works offline, so it cannot fetch the upstream fix. Its tests run in Harbor's separate verifier: a fresh
# machine from the same image, with the network they need to install their log parser, which receives the agent's
# changes as a patch in /logs/artifacts. The patch is taken against the image's whole workspace, so files the image
# leaves untracked apply to the verifier's identical copy of them.
SWEBENCH_MULTILINGUAL_EDITS = (
    PackageCopy("environment/Dockerfile", "tests/Dockerfile"),
    PackageEdit(
        "environment/Dockerfile",
        "RUN mkdir -p /logs\n",
        "RUN mkdir -p /logs\n"
        f"RUN mkdir -p {SWEBENCH_WORKSPACE_TREE.rsplit('/', 1)[0]} && cd {SWEBENCH_WORKSPACE} && {SWEBENCH_GIT} add -A"
        f" && {SWEBENCH_GIT} write-tree > {SWEBENCH_WORKSPACE_TREE} && {SWEBENCH_GIT} reset -q\n",
    ),
    PackageEdit(
        "task.toml",
        SWEBENCH_ENVIRONMENT,
        SWEBENCH_ENVIRONMENT.replace("[environment]", "[verifier.environment]")
        + "allow_internet = true\n\n"
        + f"[[verifier.collect]]\ncommand = '{SWEBENCH_COLLECT}'\ntimeout_sec = 600\n\n"
        + SWEBENCH_ENVIRONMENT
        + "allow_internet = false\n",
    ),
    PackageEdit(
        "tests/test.sh", 'exec > >(tee "$LOG_FILE") 2>&1\n', f'exec > >(tee "$LOG_FILE") 2>&1\n\n{SWEBENCH_APPLY}'
    ),
)
SWEBENCH_MULTILINGUAL = HarborDataset(
    recipe="swebench-multilingual",
    benchmark="swebench-multilingual",
    title="SWE-bench Multilingual",
    # Registry swebench_multilingual@1.0, which pins tokio-rs__tokio-4384 at this commit and the other packages
    # at its ancestor 61366953; between them only that package's tests/test.sh changes in this dataset.
    source=github_packages(
        "laude-institute/harbor-datasets",
        "19fb29e2b4d1e24615f620541b1a91461c00f5c5",
        "acf34269d8d282fc2fba779f973ef4cd0cf80c9745076294cbfc128601574325",
        root="datasets/swebench_multilingual",
    ),
    version=SWEBENCH_MULTILINGUAL_VERSION,
    count=300,
    edits=SWEBENCH_MULTILINGUAL_EDITS,
)
SWEBENCH_MULTILINGUAL_100_PACKAGES = frozenset(
    {
        "apache__druid-16875",
        "apache__lucene-11760",
        "apache__lucene-12022",
        "apache__lucene-12196",
        "apache__lucene-12212",
        "apache__lucene-13170",
        "apache__lucene-13494",
        "astral-sh__ruff-15394",
        "astral-sh__ruff-15543",
        "axios__axios-5892",
        "babel__babel-14532",
        "babel__babel-15445",
        "babel__babel-16130",
        "briannesbitt__carbon-2665",
        "briannesbitt__carbon-3041",
        "briannesbitt__carbon-3073",
        "caddyserver__caddy-6051",
        "caddyserver__caddy-6115",
        "caddyserver__caddy-6350",
        "faker-ruby__faker-2705",
        "fastlane__fastlane-20975",
        "fluent__fluentd-3608",
        "fluent__fluentd-3640",
        "fluent__fluentd-3917",
        "fluent__fluentd-4311",
        "fmtlib__fmt-1683",
        "fmtlib__fmt-3248",
        "fmtlib__fmt-3729",
        "fmtlib__fmt-3901",
        "gin-gonic__gin-3227",
        "gin-gonic__gin-3741",
        "gohugoio__hugo-12448",
        "gohugoio__hugo-12562",
        "google__gson-1100",
        "hashicorp__terraform-34814",
        "hashicorp__terraform-35543",
        "hashicorp__terraform-35611",
        "jekyll__jekyll-8167",
        "jekyll__jekyll-9141",
        "jordansissel__fpm-1850",
        "jqlang__jq-2235",
        "jqlang__jq-2681",
        "jqlang__jq-2750",
        "jqlang__jq-2839",
        "laravel__framework-46234",
        "laravel__framework-48573",
        "laravel__framework-51195",
        "laravel__framework-53206",
        "laravel__framework-53914",
        "micropython__micropython-10095",
        "micropython__micropython-12158",
        "micropython__micropython-13039",
        "mrdoob__three.js-25687",
        "mrdoob__three.js-26589",
        "nushell__nushell-13246",
        "nushell__nushell-13605",
        "nushell__nushell-13831",
        "php-cs-fixer__php-cs-fixer-7635",
        "php-cs-fixer__php-cs-fixer-7663",
        "php-cs-fixer__php-cs-fixer-8331",
        "php-cs-fixer__php-cs-fixer-8367",
        "phpoffice__phpspreadsheet-4214",
        "phpoffice__phpspreadsheet-4313",
        "preactjs__preact-2757",
        "preactjs__preact-2896",
        "preactjs__preact-3763",
        "preactjs__preact-4182",
        "preactjs__preact-4245",
        "preactjs__preact-4316",
        "projectlombok__lombok-3350",
        "projectlombok__lombok-3371",
        "projectlombok__lombok-3422",
        "projectlombok__lombok-3479",
        "projectlombok__lombok-3571",
        "projectlombok__lombok-3602",
        "projectlombok__lombok-3674",
        "prometheus__prometheus-12874",
        "prometheus__prometheus-13845",
        "prometheus__prometheus-14861",
        "prometheus__prometheus-9248",
        "redis__redis-13338",
        "redis__redis-9733",
        "rubocop__rubocop-13362",
        "rubocop__rubocop-13424",
        "rubocop__rubocop-13503",
        "rubocop__rubocop-13560",
        "rubocop__rubocop-13668",
        "rubocop__rubocop-13705",
        "sharkdp__bat-2650",
        "sharkdp__bat-3108",
        "sharkdp__bat-562",
        "tokio-rs__axum-1119",
        "tokio-rs__axum-691",
        "tokio-rs__tokio-4898",
        "tokio-rs__tokio-6551",
        "uutils__coreutils-6377",
        "uutils__coreutils-6690",
        "valkey-io__valkey-1842",
        "vuejs__core-11589",
        "vuejs__core-11739",
    }
)
"""100 packages drawn at random within each language, in proportion to its share of the 300."""
SWEBENCH_MULTILINGUAL_100 = replace(
    SWEBENCH_MULTILINGUAL,
    recipe="swebench-multilingual-100",
    benchmark="swebench-multilingual-100",
    title="SWE-bench Multilingual, stratified 100",
    source=replace(SWEBENCH_MULTILINGUAL.source, packages=SWEBENCH_MULTILINGUAL_100_PACKAGES),
    count=100,
    images=SWEBENCH_MULTILINGUAL.recipe,
)
BIXBENCH_PI = PiAgent(version="0.87.1", setup_timeout=360, compaction_reserve_tokens=2 * LONG_MAX_OUTPUT_TOKENS)
"""The campaign's ``pi-65k-16k`` profile, whose reserve is twice the per-reply limit, at the Pi its runs installed."""
BIXBENCH_32K_PROTOCOL = "pi-32k"
BIXBENCH_PI_32K = PiAgent(version="0.87.1", setup_timeout=360, compaction_reserve_tokens=LONG_MAX_OUTPUT_TOKENS)
"""The same profile for models with a 32,768-token window, whose reserve the campaign halved."""
PYTHON3_INSTALL = (
    "RUN if ! command -v python3 >/dev/null; then apt-get update"
    " && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3"
    " && rm -rf /var/lib/apt/lists/*; fi\n"
)
"""Dockerfile instructions that install Python 3, which the machine-local model relay of a Pi episode runs on."""
TERMINUS_DATASETS = (
    TERMINAL_BENCH_2,
    SWEBENCH_VERIFIED,
    SWEBENCH_VERIFIED_100,
    OT_TBLITE,
    BFCL_PARITY,
    GAIA,
    AIDER_POLYGLOT,
    FINANCEAGENT,
    MEDAGENTBENCH,
    AIME_AGENT,
    SIMPLEQA_AGENT,
)
"""Datasets whose Harbor policies run Terminus 2 with Harbor's default token limits."""
HARBOR_DATASETS = {
    dataset.recipe: dataset
    for dataset in (
        *TERMINUS_DATASETS,
        TERMINAL_BENCH_2_1,
        TERMINAL_BENCH_2_1_3TRIAL,
        OT_TBLITE_3TRIAL,
        DS_1000_MINI_200,
        BFCL_PARITY_PI,
        BIXBENCH,
        SWEBENCH_VERIFIED_MINI_SWE,
        SWEBENCH_MULTILINGUAL,
        SWEBENCH_MULTILINGUAL_100,
        *SWEBENCH_PRO_DATASETS,
    )
}


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    terminus = TerminusAgent()
    return [
        *(harbor_source(dataset, terminus) for dataset in TERMINUS_DATASETS),
        # mini-swe-agent runs on the host and sends each command through the task machine, so the image needs no install.
        harbor_source(
            SWEBENCH_VERIFIED_MINI_SWE, MiniSweAgent(), max_input_tokens=CAMPAIGN_MAX_INPUT_TOKENS, job=CAMPAIGN_JOB
        ),
        harbor_source(
            TERMINAL_BENCH_2_1,
            terminus,
            max_input_tokens=LONG_MAX_INPUT_TOKENS,
            max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
        ),
        harbor_source(
            TERMINAL_BENCH_2_1_3TRIAL,
            terminus,
            max_input_tokens=LONG_MAX_INPUT_TOKENS,
            max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
            trials=CAMPAIGN_TRIALS,
            job=CAMPAIGN_JOB,
            protocol=CAMPAIGN_PROTOCOL,
        ),
        harbor_source(
            OT_TBLITE_3TRIAL,
            terminus,
            max_input_tokens=CAMPAIGN_MAX_INPUT_TOKENS,
            trials=CAMPAIGN_TRIALS,
            job=CAMPAIGN_JOB,
            protocol=CAMPAIGN_PROTOCOL,
        ),
        # Terminus 2 as Harbor ran it, with the 16,384-token replies of the Harbor runs this port reproduces,
        # and tmux in the image because the agent's machine is offline.
        *(
            harbor_source(
                dataset,
                terminus,
                max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
                job=JobSettings(agent_install=TMUX_INSTALL),
            )
            for dataset in (SWEBENCH_MULTILINGUAL, SWEBENCH_MULTILINGUAL_100)
        ),
        *(
            harbor_source(
                dataset,
                MiniSweAgent(),
                max_input_tokens=SWEBENCH_PRO_MAX_INPUT_TOKENS,
                max_output_tokens=SWEBENCH_PRO_MAX_OUTPUT_TOKENS,
                job=SWEBENCH_PRO_JOB,
            )
            for dataset in SWEBENCH_PRO_DATASETS
        ),
        # Pi beside mini-swe-agent at SWE-bench Pro's context, with BixBench's profile, Python 3 and Pi installed over
        # each task's image, under the packages' own timeouts.
        *(
            harbor_source(
                dataset,
                BIXBENCH_PI,
                max_input_tokens=SWEBENCH_PRO_MAX_INPUT_TOKENS,
                max_output_tokens=SWEBENCH_PRO_MAX_OUTPUT_TOKENS,
                job=replace(SWEBENCH_PRO_JOB, agent_install=PYTHON3_INSTALL + pi_install(BIXBENCH_PI)),
            )
            for dataset in SWEBENCH_PRO_DATASETS
        ),
        # Pi with BixBench's profile, installed with Python 3 over each task's image, under the packages' own timeouts.
        *(
            harbor_source(
                dataset,
                BIXBENCH_PI,
                max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
                job=JobSettings(agent_install=PYTHON3_INSTALL + pi_install(BIXBENCH_PI)),
            )
            for dataset in (SWEBENCH_MULTILINGUAL, SWEBENCH_MULTILINGUAL_100)
        ),
        harbor_source(DS_1000_MINI_200, terminus, max_input_tokens=CAMPAIGN_MAX_INPUT_TOKENS, job=CAMPAIGN_JOB),
        # Pi beside Terminus 2 at the context Terminus 2 gets, with BixBench's profile for that context, under the
        # packages' own timeouts.
        harbor_source(
            TERMINAL_BENCH_2_1_3TRIAL,
            BIXBENCH_PI,
            max_input_tokens=LONG_MAX_INPUT_TOKENS,
            max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
            trials=CAMPAIGN_TRIALS,
            job=JobSettings(agent_install=PYTHON3_INSTALL + pi_install(BIXBENCH_PI)),
        ),
        *(
            harbor_source(
                dataset,
                BIXBENCH_PI_32K,
                max_input_tokens=CAMPAIGN_MAX_INPUT_TOKENS,
                max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
                trials=trials,
                job=JobSettings(agent_install=PYTHON3_INSTALL + pi_install(BIXBENCH_PI_32K)),
            )
            for dataset, trials in ((OT_TBLITE_3TRIAL, CAMPAIGN_TRIALS), (DS_1000_MINI_200, 1))
        ),
        harbor_source(
            BFCL_PARITY_PI,
            CAMPAIGN_PI,
            max_input_tokens=CAMPAIGN_MAX_INPUT_TOKENS,
            max_output_tokens=PI_MAX_OUTPUT_TOKENS,
            job=replace(CAMPAIGN_JOB, agent_install=pi_install(CAMPAIGN_PI)),
        ),
        harbor_source(
            BIXBENCH,
            BIXBENCH_PI,
            max_input_tokens=LONG_MAX_INPUT_TOKENS,
            max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
            job=replace(CAMPAIGN_JOB, storage_mb=10240, agent_install=pi_install(BIXBENCH_PI)),
        ),
        harbor_source(
            BIXBENCH,
            BIXBENCH_PI_32K,
            max_input_tokens=CAMPAIGN_MAX_INPUT_TOKENS,
            max_output_tokens=LONG_MAX_OUTPUT_TOKENS,
            job=replace(CAMPAIGN_JOB, storage_mb=10240, agent_install=pi_install(BIXBENCH_PI_32K)),
            protocol=BIXBENCH_32K_PROTOCOL,
        ),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock", help="Resolve and write the images of every dataset sharing a lock")
    lock.add_argument("lock", choices=sorted({dataset.image_lock_name for dataset in HARBOR_DATASETS.values()}))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    images: dict[str, str] = {}
    for dataset in HARBOR_DATASETS.values():
        if dataset.image_lock_name == arguments.lock:
            images.update(lock_images(dataset))
    path = IMAGE_LOCKS / f"{arguments.lock}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(sorted(images.items())), indent=2) + "\n")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
