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
    HarborDataset,
    PackageEdit,
    github_packages,
    harbor_source,
    huggingface_packages,
    lock_images,
    pi_install,
)
from experiments.evaluation.benchmarks.harbor_judges import bixbench_answer, financeagent_answer, simpleqa_answer
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.post_training.task_curation.harbor_packages import package_rows
from experiments.post_training.task_curation.pipeline import HarborHubSource
from experiments.post_training.task_curation.source import RlDataSource

logger = logging.getLogger(__name__)

VERSION = "2026.11.17.2"
# The version of the datasets whose rule renders a judge spec into each task. Their recipe identity
# does not cover the rendered judge, so a change to it bumps this version.
JUDGED_VERSION = "2026.11.23"
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
        "791c0607c15a6d7946aa52e9c1a427d104fc0916",
        "970d8e5e3e61f6c5f955be5815eb0f73729bbcd739f5657d3e71cee9725d9385",
        root="datasets/ds1000",
        packages=DS_1000_MINI_200_PACKAGES,
    ),
    version=VERSION,
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
SWEBENCH_PRO_SOURCE = github_packages(
    "laude-institute/harbor-datasets",
    "c8e8f3fac7097accaacf261d74c3d6f441de45b1",
    "504adb77fd404b767da2019444f13afc3958c59a1110bf98f5d8f18ff32b7682",
    root="datasets/swebenchpro",
)
"""Registry swebenchpro@1.0: Scale AI's public SWE-Bench Pro set, each task on Scale's prebuilt image."""
SWEBENCH_PRO = HarborDataset(
    recipe="swebench-pro",
    benchmark="swebench-pro",
    title="SWE-Bench Pro",
    source=SWEBENCH_PRO_SOURCE,
    version=VERSION,
    count=731,
)
SWEBENCH_PRO_200_PACKAGES = frozenset(
    """
    instance_ansible__ansible-164881d871964aa64e0f911d03ae270acbad253c-v390e508d27db7a51eece36bb6d9698b63a5b638a
    instance_ansible__ansible-1b70260d5aa2f6c9782fd2b848e8d16566e50d85-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-1ee70fc272aff6bf3415357c6e13c5de5b928d9b-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-395e5e20fab9cad517243372fa3c3c5d9e09ab2a-v7eee2454f617569fd6889f2211f75bc02a35f9f8
    instance_ansible__ansible-39bd8b99ec8c6624207bf3556ac7f9626dad9173-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-5260527c4a71bfed99d803e687dd19619423b134-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-5e88cd9972f10b66dd97e1ee684c910c6a2dd25e-v906c969b551b346ef54a2c0b41e04f632b7b73c2
    instance_ansible__ansible-5f4e332e3762999d94af27746db29ff1729252c1-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-709484969c8a4ffd74b839a673431a8c5caa6457-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-77658704217d5f166404fc67997203c25381cb6e-v390e508d27db7a51eece36bb6d9698b63a5b638a
    instance_ansible__ansible-7e1a347695c7987ae56ef1b6919156d9254010ad-v390e508d27db7a51eece36bb6d9698b63a5b638a
    instance_ansible__ansible-83909bfa22573777e3db5688773bda59721962ad-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-935528e22e5283ee3f63a8772830d3d01f55ed8c-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-984216f52e76b904e5b0fa0fb956ab4f1e0a7751-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-a02e22e902a69aeb465f16bf03f7f5a91b2cb828-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-a20a52701402a12f91396549df04ac55809f68e9-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-b8025ac160146319d2b875be3366b60c852dd35d-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-be2c376ab87e3e872ca21697508f12c6909cf85a-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-bec27fb4c0a40c5f8bbcf26a475704227d65ee73-v30a923fb5c164d6cd18280c02422f75e611e8fb2
    instance_ansible__ansible-bf98f031f3f5af31a2d78dc2f0a58fe92ebae0bb-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-cb94c0cc550df9e98f1247bc71d8c2b861c75049-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-d58e69c82d7edd0583dd8e78d76b075c33c3151e-v173091e2e36d38c978002990795f66cfc0af30ad
    instance_ansible__ansible-de01db08d00c8d2438e1ba5989c313ba16a145b0-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-e0c91af45fa9af575d10fd3e724ebc59d2b2d6ac-v30a923fb5c164d6cd18280c02422f75e611e8fb2
    instance_ansible__ansible-e64c6c1ca50d7d26a8e7747d8eb87642e767cd74-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_element-hq__element-web-1077729a19c0ce902e713cf6fab42c91fb7907f1-vnan
    instance_element-hq__element-web-27139ca68eb075a4438c18fca184887002a4ffbc-vnan
    instance_element-hq__element-web-33299af5c9b7a7ec5a9c31d578d4ec5b18088fb7-vnan
    instance_element-hq__element-web-4fec436883b601a3cac2d4a58067e597f737b817-vnan
    instance_element-hq__element-web-56c7fc1948923b4b3f3507799e725ac16bcf8018-vnan
    instance_element-hq__element-web-6205c70462e0ce2e1e77afb3a70b55d0fdfe1b31-vnan
    instance_element-hq__element-web-6961c256035bed0b7640a6e5907652c806968478-vnan
    instance_element-hq__element-web-776ffa47641c7ec6d142ab4a47691c30ebf83c2e
    instance_element-hq__element-web-7c63d52500e145d6fff6de41dd717f61ab88d02f-vnan
    instance_element-hq__element-web-8f3c8b35153d2227af45f32e46bd1e15bd60b71f-vnan
    instance_element-hq__element-web-9bf77963ee5e036d54b2a3ca202fbf6378464a5e-vnan
    instance_element-hq__element-web-ad26925bb6628260cfe0fcf90ec0a8cba381f4a4-vnan
    instance_element-hq__element-web-b7fea97bb68c6628a644580076f840109132f074-vnan
    instance_element-hq__element-web-ca58617cee8aa91c93553449bfdf9b3465a5119b-vnan
    instance_element-hq__element-web-ecfd1736e5dd9808e87911fc264e6c816653e1a9-vnan
    instance_flipt-io__flipt-02e21636c58e86c51119b63e0fb5ca7b813b07b1
    instance_flipt-io__flipt-292fdaca9be39e6a921aaa8874c011d0fdd3e874
    instance_flipt-io__flipt-2ca5dfb3513e4e786d2b037075617cccc286d5c3
    instance_flipt-io__flipt-2ce8a0331e8a8f63f2c1b555db8277ffe5aa2e63
    instance_flipt-io__flipt-36e62baffae2132f78f9d34dc300a9baa2d7ae0e
    instance_flipt-io__flipt-40007b9d97e3862bcef8c20ae6c87b22ea0627f0
    instance_flipt-io__flipt-406f9396ad65696d58865b3a6283109cd4eaf40e
    instance_flipt-io__flipt-56a620b8fc9ef7a0819b47709aa541cdfdbba00b
    instance_flipt-io__flipt-5ffba3406a7993d97ced4cc13658bee66150fcca
    instance_flipt-io__flipt-690672523398c2b6f6e4562f0bf9868664ab894f
    instance_flipt-io__flipt-6fd0f9e2587f14ac1fdd1c229f0bcae0468c8daa
    instance_flipt-io__flipt-96820c3ad10b0b2305e8877b6b303f7fafdf815f
    instance_flipt-io__flipt-9d25c18b79bc7829a6fb08ec9e8793d5d17e2868
    instance_flipt-io__flipt-9f8127f225a86245fa35dca4885c2daef824ee55
    instance_flipt-io__flipt-b2cd6a6dd73ca91b519015fd5924fde8d17f3f06
    instance_flipt-io__flipt-c12967bc73fdf02054cf3ef8498c05e25f0a18c0
    instance_flipt-io__flipt-c6a7b1fd933e763b1675281b30077e161fa115a1
    instance_flipt-io__flipt-c8d71ad7ea98d97546f01cce4ccb451dbcf37d3b
    instance_flipt-io__flipt-cd2f3b0a9d4d8b8a6d3d56afab65851ecdc408e8
    instance_flipt-io__flipt-db1c3b100e231c62f0c90c2ab037614f20a2a63b
    instance_flipt-io__flipt-e42da21a07a5ae35835ec54f74004ebd58713874
    instance_flipt-io__flipt-ea9a2663b176da329b3f574da2ce2a664fc5b4a1
    instance_flipt-io__flipt-ee02b164f6728d3227c42671028c67a4afd36918
    instance_future-architect__vuls-1832b4ee3a20177ad313d806983127cb6e53f5cf
    instance_future-architect__vuls-2c84be80b65d022c262956cd26fc79d8bb2f7010
    instance_future-architect__vuls-457a3a9627fb9a0800d0aecf1d4713fb634a9011
    instance_future-architect__vuls-54e73c2f5466ef5daec3fb30922b9ac654e4ed25
    instance_future-architect__vuls-6682232b5c8a9d08c0e9f15bd90d41bff3875adc
    instance_future-architect__vuls-8659668177f1feb65963db7a967347a79c5f9c40
    instance_future-architect__vuls-9aa0d87a21bede91c2b45c32187456bb69455e92
    instance_future-architect__vuls-a76302c11174ca081f656c63a000ffa746e350af
    instance_future-architect__vuls-abd80417728b16c6502067914d27989ee575f0ee
    instance_future-architect__vuls-ad2edbb8448e2c41a097f1c0b52696c0f6c5924d
    instance_future-architect__vuls-d18e7a751d07260d75ce3ba0cd67c4a6aebfd967
    instance_future-architect__vuls-e3c27e1817d68248043bd09d63cc31f3344a6f2c
    instance_future-architect__vuls-e4728e388120b311c4ed469e4f942e0347a2689b-v264a82e2f4818e30f5a25e4da53b27ba119f62b5
    instance_future-architect__vuls-edb324c3d9ec3b107bf947f00e38af99d05b3e16
    instance_future-architect__vuls-f0b3a8b1db98eb1bd32685f1c36c41a99c3452ed
    instance_future-architect__vuls-fd18df1dd4e4360f8932bc4b894bd8b40d654e7c
    instance_future-architect__vuls-fe8d252c51114e922e6836055ef86a15f79ad042
    instance_gravitational__teleport-02d1efb8560a1aa1c72cfb1c08edd8b84a9511b4-vce94f93ad1030e3136852817f2423c1b3ac37bc4
    instance_gravitational__teleport-0ac7334939981cf85b9591ac295c3816954e287e
    instance_gravitational__teleport-0cb341c926713bdfcbb490c69659a9b101df99eb
    instance_gravitational__teleport-1316e6728a3ee2fc124e2ea0cc6a02044c87a144-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-1330415d33a27594c948a36d9d7701f496229e9f
    instance_gravitational__teleport-1b08e7d0dbe68fe530a0f08ad408ec198b7c53fc-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-3ff19cf7c41f396ae468797d3aeb61515517edc9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-3ff75e29fb2153a2637fe7f83e49dc04b1c99c9f
    instance_gravitational__teleport-4f771403dc4177dc26ee0370f7332f3fe54bee0f-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-5dca072bb4301f4579a15364fcf37cc0c39f7f6c
    instance_gravitational__teleport-6a14edcf1ff010172fdbac622d0a474ed6af46de
    instance_gravitational__teleport-769b4b5eec7286b7b14e179f2cc52e6b15d2d9f3-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-82185f232ae8974258397e121b3bc2ed0c3729ed-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-a95b3ae0667f9e4b2404bf61f51113e6d83f01cd
    instance_gravitational__teleport-ac2fb2f9b4fd1896b554d3011df23d3d71295779
    instance_gravitational__teleport-b4e7cd3a5e246736d3fe8d6886af55030b232277
    instance_gravitational__teleport-ba6c4a135412c4296dd5551bd94042f0dc024504-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-c1b1c6a1541c478d7777a48fca993cc8206c73b9
    instance_gravitational__teleport-d6ffe82aaf2af1057b69c61bf9df777f5ab5635a-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-e6d86299a855687b21970504fbf06f52a8f80c74-vce94f93ad1030e3136852817f2423c1b3ac37bc4
    instance_gravitational__teleport-f432a71a13e698b6e1c4672a2e9e9c1f32d35c12
    instance_internetarchive__openlibrary-08ac40d050a64e1d2646ece4959af0c42bf6b7b5-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-1351c59fd43689753de1fca32c78d539a116ffc1-v29f82c9cf21d57b242f8d8b0e541525d259e2d63
    instance_internetarchive__openlibrary-2abe28b472ffed563a87cfe83685b161b35263b0-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_internetarchive__openlibrary-431442c92887a3aece3f8aa771dd029738a80eb1-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c
    instance_internetarchive__openlibrary-4b7ea2977be2747496ba792a678940baa985f7ea-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-53e02a22972e9253aeded0e1981e6845e1e521fe-vfa6ff903cb27f336e17654595dd900fa943dcd91
    instance_internetarchive__openlibrary-58999808a17a26b387f8237860a7a524d1e2d262-v08d8e8889ec945ab821fb156c04c7d2e2810debb
    instance_internetarchive__openlibrary-5fb312632097be7e9ac6ab657964af115224d15d-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-60725705782832a2cb22e17c49697948a42a9d03-v298a7a812ceed28c4c18355a091f1b268fe56d86
    instance_internetarchive__openlibrary-6fdbbeee4c0a7e976ff3e46fb1d36f4eb110c428-v08d8e8889ec945ab821fb156c04c7d2e2810debb
    instance_internetarchive__openlibrary-72321288ea790a3ace9e36f1c05b68c93f7eec43-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-798055d1a19b8fa0983153b709f460be97e33064-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_internetarchive__openlibrary-7edd1ef09d91fe0b435707633c5cc9af41dedddf-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c
    instance_internetarchive__openlibrary-89e4b4431fe7506c365a6f6eb6f6d048d04c044c-v08d8e8889ec945ab821fb156c04c7d2e2810debb
    instance_internetarchive__openlibrary-91efee627df01e32007abf2d6ebf73f9d9053076-vbee42ad1b72fb23c6a1c874868a720b370983ed2
    instance_internetarchive__openlibrary-a48fd6ba9482c527602bc081491d9e8ae6e8226c-vfa6ff903cb27f336e17654595dd900fa943dcd91
    instance_internetarchive__openlibrary-a7b7dc5735a1b3a9824376b1b469b556dd413981-va4315b5dc369c1ef66ae22f9ae4267aa3114e1b3
    instance_internetarchive__openlibrary-b4f7c185ae5f1824ac7f3a18e8adf6a4b468459c-v08d8e8889ec945ab821fb156c04c7d2e2810debb
    instance_internetarchive__openlibrary-b67138b316b1e9c11df8a4a8391fe5cc8e75ff9f-ve8c8d62a2b60610a3c4631f5f23ed866bada9818
    instance_internetarchive__openlibrary-bb152d23c004f3d68986877143bb0f83531fe401-ve8c8d62a2b60610a3c4631f5f23ed866bada9818
    instance_internetarchive__openlibrary-c506c1b0b678892af5cb22c1c1dbc35d96787a0a-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-c8996ecc40803b9155935fd7ff3b8e7be6c1437c-ve8fc82d8aae8463b752a211156c5b7b59f349237
    instance_internetarchive__openlibrary-f0341c0ba81c790241b782f5103ce5c9a6edf8e3-ve8fc82d8aae8463b752a211156c5b7b59f349237
    instance_internetarchive__openlibrary-f3b26c2c0721f8713353fe4b341230332e30008d-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-fad4a40acf5ff5f06cd7441a5c7baf41a7d81fe4-vfa6ff903cb27f336e17654595dd900fa943dcd91
    instance_navidrome__navidrome-09ae41a2da66264c60ef307882362d2e2d8d8b89
    instance_navidrome__navidrome-0a650de357babdcc8ce910fe37fee84acf4ed2fe
    instance_navidrome__navidrome-23bebe4e06124becf1000e88472ae71a6ca7de4c
    instance_navidrome__navidrome-27875ba2dd1673ddf8affca526b0664c12c3b98b
    instance_navidrome__navidrome-29b7b740ce469201af0a0510f3024adc93ef4c8e
    instance_navidrome__navidrome-29bc17acd71596ae92131aca728716baf5af9906
    instance_navidrome__navidrome-3977ef6e0f287f598b6e4009876239d6f13b686d
    instance_navidrome__navidrome-56303cde23a4122d2447cbb266f942601a78d7e4
    instance_navidrome__navidrome-669c8f4c49a7ef51ac9a53c725097943f67219eb
    instance_navidrome__navidrome-6b3b4d83ffcf273b01985709c8bc5df12bbb8286
    instance_navidrome__navidrome-8d56ec898e776e7e53e352cb9b25677975787ffc
    instance_navidrome__navidrome-b3980532237e57ab15b2b93c49d5cd5b2d050013
    instance_navidrome__navidrome-d0dceae0943b8df16e579c2d9437e11760a0626a
    instance_navidrome__navidrome-d21932bd1b2379b0ebca2d19e5d8bae91040268a
    instance_navidrome__navidrome-dfa453cc4ab772928686838dc73d0130740f054e
    instance_navidrome__navidrome-f78257235ec3429ef42af6687738cd327ec77ce8
    instance_nodebb__nodebb-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan
    instance_nodebb__nodebb-0c81642997ea1d827dbd02c311db9d4976112cd4-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_nodebb__nodebb-0e07f3c9bace416cbab078a30eae972868c0a8a3-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_nodebb__nodebb-18c45b44613aecd53e9f60457b9812049ab2998d-v0495b863a912fbff5749c67e860612b91825407c
    instance_nodebb__nodebb-3c85b944e30a0ba8b3ec9e1f441c74f383625a15-v4fbcfae8b15e4ce5d132c408bca69ebb9cf146ed
    instance_nodebb__nodebb-82562bec444940608052f3e4149e0c61ec80bf3f-vd59a5728dfc977f44533186ace531248c2917516
    instance_nodebb__nodebb-84e065752f6d7fbe5c08cbf50cb173ffb866b8fa-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_nodebb__nodebb-bd80d36e0dcf78cd4360791a82966078b3a07712-v4fbcfae8b15e4ce5d132c408bca69ebb9cf146ed
    instance_nodebb__nodebb-cfc237c2b79d8c731bbfc6cadf977ed530bfd57a-v0495b863a912fbff5749c67e860612b91825407c
    instance_nodebb__nodebb-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_nodebb__nodebb-eb49a64974ca844bca061744fb3383f5d13b02ad-vnan
    instance_nodebb__nodebb-f083cd559d69c16481376868c8da65172729c0ca-vnan
    instance_protonmail__webclients-01ea5214d11e0df8b7170d91bafd34f23cb0f2b1
    instance_protonmail__webclients-0200ce0fc1d4dbd35178c10d440a284c82ecc858
    instance_protonmail__webclients-08bb09914d0d37b0cd6376d4cab5b77728a43e7b
    instance_protonmail__webclients-09fcf0dbdb87fa4f4a27700800ee4a3caed8b413
    instance_protonmail__webclients-0ec14e36ceb01ba45602a563e12352af8171ed39
    instance_protonmail__webclients-1501eb765873b2884b6f1944fd242ecfc9d6b103
    instance_protonmail__webclients-2dce79ea4451ad88d6bfe94da22e7f2f988efa60
    instance_protonmail__webclients-2f2f6c311c6128fe86976950d3c0c2db07b03921
    instance_protonmail__webclients-2f66db85455f4b22a47ffd853738f679b439593c
    instance_protonmail__webclients-6e1873b06df6529a469599aa1d69d3b18f7d9d37
    instance_protonmail__webclients-715dbd4e6999499cd2a576a532d8214f75189116
    instance_protonmail__webclients-815695401137dac2975400fc610149a16db8214b
    instance_protonmail__webclients-a6e6f617026794e7b505d649d2a7a9cdf17658c8
    instance_protonmail__webclients-b530a3db50cb33e5064464addbcbef1465856ce6
    instance_protonmail__webclients-bf2e89c0c488ae1a87d503e5b09fe9dd2f2a635f
    instance_protonmail__webclients-c5a2089ca2bfe9aa1d85a664b8ad87ef843a1c9c
    instance_protonmail__webclients-cba6ebbd0707caa524ffee51c62b197f6122c902
    instance_protonmail__webclients-e9677f6c46d5ea7d277a4532a4bf90074f125f31
    instance_qutebrowser__qutebrowser-0fc6d1109d041c69a68a896db87cf1b8c194cef7-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-1943fa072ec3df5a87e18a23b0916f134c131016-vafb3e8e01b31319c66c4e666b8a3b1d8ba55db24
    instance_qutebrowser__qutebrowser-1a9e74bfaf9a9db2a510dc14572d33ded6040a57-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-1af602b258b97aaba69d2585ed499d95e2303ac2-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-2dd8966fdcf11972062c540b7a787e4d0de8d372-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-3fd8e12949b8feda401930574facf09dd4180bba
    instance_qutebrowser__qutebrowser-44e64199ed38003253f0296badd4a447645067b6-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-473a15f7908f2bb6d670b0e908ab34a28d8cf7e2-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-52708364b5f91e198defb022d1a5b4b3ebd9b563-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-5cef49ff3074f9eab1da6937a141a39a20828502-v02ad04386d5238fe2d1a1be450df257370de4b6a
    instance_qutebrowser__qutebrowser-66cfa15c372fa9e613ea5a82d3b03e4609399fb6-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-7b603dd6bf195e3e723ce08ff64a82b406e3f6b6-v9f8e9d96c85c85a605e382f1510bd08563afc566
    instance_qutebrowser__qutebrowser-96b997802e942937e81d2b8a32d08f00d3f4bc4e-v5fc38aaf22415ab0b70567368332beee7955b367
    instance_qutebrowser__qutebrowser-996487c43e4fcc265b541f9eca1e7930e3c5cf05-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-a84ecfb80a00f8ab7e341372560458e3f9cfffa2-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-c09e1439f145c66ee3af574386e277dd2388d094-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-c580ebf0801e5a3ecabc54f327498bb753c6d5f2-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-cf06f4e3708f886032d4d2a30108c2fddb042d81-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-deeb15d6f009b3ca0c3bd503a7cef07462bd16b4-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-e34dfc68647d087ca3175d9ad3f023c30d8c9746-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-e64622cd2df5b521342cf4a62e0d4cb8f8c9ae5a-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-ff1c025ad3210506fc76e1f604d8c8c27637d88e-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_tutao__tutanota-12a6cbaa4f8b43c2f85caca0787ab55501539955-vc4e41fd0029957297843cb9dec4a25c7c756f029
    instance_tutao__tutanota-40e94dee2bcec2b63f362da283123e9df1874cc1-vc4e41fd0029957297843cb9dec4a25c7c756f029
    instance_tutao__tutanota-4b4e45949096bb288f2b522f657610e480efa3e8-vee878bb72091875e912c52fc32bc60ec3760227b
    instance_tutao__tutanota-fbdb72a2bd39b05131ff905780d9d4a2a074de26-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf
    instance_tutao__tutanota-fe240cbf7f0fdd6744ef7bef8cb61676bcdbb621-vc4e41fd0029957297843cb9dec4a25c7c756f029
    """.split()
)
"""A fixed 200 of SWE-Bench Pro's 731 tasks; each repository keeps its share of the 731."""
SWEBENCH_PRO_SMOKE_PACKAGES = frozenset(
    """
    instance_ansible__ansible-984216f52e76b904e5b0fa0fb956ab4f1e0a7751-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-a20a52701402a12f91396549df04ac55809f68e9-v1055803c3a812189a1133297f7f5468579283f86
    instance_element-hq__element-web-18c03daa865d3c5b10e52b669cd50be34c67b2e5-vnan
    instance_element-hq__element-web-56c7fc1948923b4b3f3507799e725ac16bcf8018-vnan
    instance_flipt-io__flipt-c6a7b1fd933e763b1675281b30077e161fa115a1
    instance_flipt-io__flipt-cd2f3b0a9d4d8b8a6d3d56afab65851ecdc408e8
    instance_future-architect__vuls-7e91f5ef7e5712b1a3d7d5066ad6607e9debc21c
    instance_future-architect__vuls-f6509a537660ea2bce0e57958db762edd3a36702
    instance_gravitational__teleport-6eaaf3a27e64f4ef4ef855bd35d7ec338cf17460-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-c335534e02de143508ebebc7341021d7f8656e8f
    instance_internetarchive__openlibrary-798055d1a19b8fa0983153b709f460be97e33064-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_internetarchive__openlibrary-c506c1b0b678892af5cb22c1c1dbc35d96787a0a-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_navidrome__navidrome-3972616585e82305eaf26aa25697b3f5f3082288
    instance_navidrome__navidrome-8383527aaba1ae8fa9765e995a71a86c129ef626
    instance_nodebb__nodebb-1ea9481af6125ffd6da0592ed439aa62af0bca11-vd59a5728dfc977f44533186ace531248c2917516
    instance_nodebb__nodebb-70b4a0e2aebebe8f2f559de6680093d96a697b2f-vnan
    instance_protonmail__webclients-2f2f6c311c6128fe86976950d3c0c2db07b03921
    instance_protonmail__webclients-3f22e2172cbdfd7b9abb2b1d8fd80c16d38b4bbe
    instance_qutebrowser__qutebrowser-5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-ed19d7f58b2664bb310c7cb6b52c5b9a06ea60b2-v059c6fdc75567943479b23ebca7c07b5e9a7f34c
    instance_tutao__tutanota-219bc8f05d7b980e038bc1524cb021bf56397a1b-vee878bb72091875e912c52fc32bc60ec3760227b
    instance_tutao__tutanota-fe240cbf7f0fdd6744ef7bef8cb61676bcdbb621-vc4e41fd0029957297843cb9dec4a25c7c756f029
    """.split()
)
"""Two tasks from each of SWE-Bench Pro's 11 repositories, drawn with ``random.Random(0)``."""
SWEBENCH_PRO_200 = HarborDataset(
    recipe="swebench-pro-200",
    benchmark="swebench-pro-200",
    title="SWE-Bench Pro, stratified 200",
    source=replace(SWEBENCH_PRO_SOURCE, packages=SWEBENCH_PRO_200_PACKAGES),
    version=VERSION,
    count=200,
    images=SWEBENCH_PRO.recipe,
)
SWEBENCH_PRO_SMOKE = HarborDataset(
    recipe="swebench-pro-smoke",
    benchmark="swebench-pro-smoke",
    title="SWE-Bench Pro, 22-task smoke",
    source=replace(SWEBENCH_PRO_SOURCE, packages=SWEBENCH_PRO_SMOKE_PACKAGES),
    version=VERSION,
    count=22,
    images=SWEBENCH_PRO.recipe,
)
SWEBENCH_PRO_DATASETS = (SWEBENCH_PRO, SWEBENCH_PRO_200, SWEBENCH_PRO_SMOKE)
SWEBENCH_PRO_MAX_INPUT_TOKENS = 131072
SWEBENCH_PRO_MAX_OUTPUT_TOKENS = 16384
"""Marin's agentic eval policy for mini-swe-agent: a 128K context and 16K replies."""
BIXBENCH_PI = PiAgent(version="0.87.1", setup_timeout=360, compaction_reserve_tokens=2 * LONG_MAX_OUTPUT_TOKENS)
"""The campaign's ``pi-65k-16k`` profile, whose reserve is twice the per-reply limit, at the Pi its runs installed."""
BIXBENCH_32K_PROTOCOL = "pi-32k"
BIXBENCH_PI_32K = PiAgent(version="0.87.1", setup_timeout=360, compaction_reserve_tokens=LONG_MAX_OUTPUT_TOKENS)
"""The same profile for models with a 32,768-token window, whose reserve the campaign halved."""
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
        *(
            harbor_source(
                dataset,
                MiniSweAgent(),
                max_input_tokens=SWEBENCH_PRO_MAX_INPUT_TOKENS,
                max_output_tokens=SWEBENCH_PRO_MAX_OUTPUT_TOKENS,
            )
            for dataset in SWEBENCH_PRO_DATASETS
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
        harbor_source(DS_1000_MINI_200, terminus, max_input_tokens=CAMPAIGN_MAX_INPUT_TOKENS, job=CAMPAIGN_JOB),
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
