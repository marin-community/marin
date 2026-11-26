# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-Bench Pro V2 (Scale AI's public set) under Scale's locked protocol.

Scale's V2 task images keep only each task's starting commit, so the fix is not in the repository's history.
The agent works offline. Its changes are collected as a patch and graded in a fresh machine from the same
image, with the network the tests need, as Scale's patch-replay regrade does; the package's own ``test.sh``
then resets the test files and applies the gold test patch. https://github.com/scaleapi/SWE-bench_Pro-os
"""

from dataclasses import replace

from taskcompendium.importers.harbor.package import JobSettings

from experiments.evaluation.benchmarks.harbor import HarborDataset, PackageCopy, PackageEdit, github_packages

VERSION = "2026.11.26.1"
WORKSPACE_TREE = "/opt/marin/workspace-tree"
"""The git tree of the image's workspace, untracked files included, that the agent's changes are diffed against."""
PATCH = "/logs/artifacts/model.patch"
SETUP_TIMEOUT = 300
GIT = "git -c safe.directory=/app"
RECORD_WORKSPACE = (
    f"mkdir -p /opt/marin && cd /app && {GIT} add -A && {GIT} write-tree > {WORKSPACE_TREE} && {GIT} reset -q"
)
COLLECT_PATCH = (
    f'set -e; cd /app; {GIT} add -A; mkdir -p /logs/artifacts; {GIT} diff --cached --binary "$(cat {WORKSPACE_TREE})"'
    f" > {PATCH}"
)
APPLY_PATCH = (
    f"if [ -s {PATCH} ] && ! {GIT} apply --binary {PATCH}; then\n"
    "    echo \"The agent's changes do not apply to the task's workspace\"\n"
    "    exit 1\n"
    "fi\n"
)
VERIFIER = '[verifier]\nnetwork_mode = "public"\ntimeout_sec = 3000.0\n'
TEST_WORKDIR = (
    'cd /app 2>/dev/null || cd /testbed 2>/dev/null || { echo "ERROR: Neither /app nor /testbed exists"; exit 1; }\n'
)
# The packages grade in the agent's container, whose tests need the network the offline agent lacks. They now
# record the image's workspace before the agent starts, and grade in Harbor's separate verifier: a fresh machine
# from the same image and resources, which applies the agent's changes before the package's tests run.
SWEBENCH_PRO_EDITS = (
    PackageCopy("environment/Dockerfile", "tests/Dockerfile"),
    PackageEdit(
        "task.toml",
        VERIFIER + "collect = []\n",
        VERIFIER
        + f"\n[[verifier.collect]]\ncommand = '{COLLECT_PATCH}'\ntimeout_sec = 600\n"
        + "\n[verifier.environment]\ncpus = 1\nmemory_mb = 4096\nstorage_mb = 10240\n",
    ),
    PackageEdit(
        "task.toml",
        "[environment.env]\n",
        f"[environment.setup]\ncommand = '{RECORD_WORKSPACE}'\ntimeout_sec = {SETUP_TIMEOUT}\n\n[environment.env]\n",
    ),
    PackageEdit("tests/test.sh", TEST_WORKDIR, TEST_WORKDIR + APPLY_PATCH),
)
SWEBENCH_PRO_JOB = JobSettings(setup_timeout=SETUP_TIMEOUT)

SWEBENCH_PRO_SOURCE = github_packages(
    "scaleapi/SWE-bench_Pro-os",
    "66f92766bba642462d4bbe5479e83f91f9211862",
    "a459a929729c95d15cb7b4c5c599d36ea4e9a067bedc9d018cee6032b577e6cc",
    root="v2/tasks",
)
"""Scale's v2.0.0 release: the 642 public V2 tasks, each on its own sanitized image."""
SWEBENCH_PRO = HarborDataset(
    recipe="swebench-pro",
    benchmark="swebench-pro",
    title="SWE-Bench Pro V2",
    source=SWEBENCH_PRO_SOURCE,
    version=VERSION,
    count=642,
    edits=SWEBENCH_PRO_EDITS,
)

SWEBENCH_PRO_200_PACKAGES = frozenset(
    """
    instance_NodeBB__NodeBB-05f2236193f407cf8e2072757fbd6bb170bc13f0-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_NodeBB__NodeBB-397835a05a8e2897324e566b41c5e616e172b4af-v89631a1cdb318276acb48860c5d78077211397c6
    instance_NodeBB__NodeBB-70b4a0e2aebebe8f2f559de6680093d96a697b2f-vnan
    instance_NodeBB__NodeBB-767973717be700f46f06f3e7f4fc550c63509046-vnan
    instance_NodeBB__NodeBB-97c8569a798075c50e93e585ac741ab55cb7c28b-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_NodeBB__NodeBB-9c576a0758690f45a6ca03b5884c601e473bf2c1-vd59a5728dfc977f44533186ace531248c2917516
    instance_NodeBB__NodeBB-a5afad27e52fd336163063ba40dcadc80233ae10-vd59a5728dfc977f44533186ace531248c2917516
    instance_NodeBB__NodeBB-bad15643013ca15affe408b75eba9e47cc604bb2-vd59a5728dfc977f44533186ace531248c2917516
    instance_NodeBB__NodeBB-be43cd25974681c9743d424238b7536c357dc8d3-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_NodeBB__NodeBB-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_NodeBB__NodeBB-f083cd559d69c16481376868c8da65172729c0ca-vnan
    instance_NodeBB__NodeBB-f48ed3658aab7be0f1165d4c1f89af48d7865189-v0495b863a912fbff5749c67e860612b91825407c
    instance_ansible__ansible-12734fa21c08a0ce8c84e533abdc560db2eb1955-v7eee2454f617569fd6889f2211f75bc02a35f9f8
    instance_ansible__ansible-164881d871964aa64e0f911d03ae270acbad253c-v390e508d27db7a51eece36bb6d9698b63a5b638a
    instance_ansible__ansible-185d41031660a676c43fbb781cd1335902024bfe-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-34db57a47f875d11c4068567b9ec7ace174ec4cf-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-3b823d908e8a5d17674f8c26d337d3114b7493b1-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-3db08adbb1cc6aa9941be5e0fc810132c6e1fa4b-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-42355d181a11b51ebfc56f6f4b3d9c74e01cb13b-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-5e369604e1930b1a2e071fecd7ec5276ebd12cb1-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-709484969c8a4ffd74b839a673431a8c5caa6457-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-748f534312f2073a25a87871f5bd05882891b8c4-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-8127abbc298cabf04aaa89a478fc5e5e3432a6fc-v30a923fb5c164d6cd18280c02422f75e611e8fb2
    instance_ansible__ansible-9142be2f6cabbe6597c9254c5bb9186d17036d55-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-935528e22e5283ee3f63a8772830d3d01f55ed8c-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-984216f52e76b904e5b0fa0fb956ab4f1e0a7751-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-a02e22e902a69aeb465f16bf03f7f5a91b2cb828-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
    instance_ansible__ansible-a1569ea4ca6af5480cf0b7b3135f5e12add28a44-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-a7d2a4e03209cff1e97e59fd54bb2b05fdbdbec6-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-be59caa59bf47ca78a4760eb7ff38568372a8260-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-bec27fb4c0a40c5f8bbcf26a475704227d65ee73-v30a923fb5c164d6cd18280c02422f75e611e8fb2
    instance_ansible__ansible-c616e54a6e23fa5616a1d56d243f69576164ef9b-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-d30fc6c0b359f631130b0e979d9a78a7b3747d48-v1055803c3a812189a1133297f7f5468579283f86
    instance_ansible__ansible-d33bedc48fdd933b5abd65a77c081876298e2f07-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-d62496fe416623e88b90139dc7917080cb04ce70-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-d6d2251929c84c3aa883bad7db0f19cc9ff0339e-v30a923fb5c164d6cd18280c02422f75e611e8fb2
    instance_ansible__ansible-de5858f48dc9e1ce9117034e0d7e76806f420ca8-v1055803c3a812189a1133297f7f5468579283f86
    instance_element-hq__element-web-18c03daa865d3c5b10e52b669cd50be34c67b2e5-vnan
    instance_element-hq__element-web-44b98896a79ede48f5ad7ff22619a39d5f6ff03c-vnan
    instance_element-hq__element-web-53a9b6447bd7e6110ee4a63e2ec0322c250f08d1-vnan
    instance_element-hq__element-web-582a1b093fc0b77538052f45cbb9c7295f991b51-vnan
    instance_element-hq__element-web-5e8488c2838ff4268f39db4a8cca7d74eecf5a7e-vnan
    instance_element-hq__element-web-6205c70462e0ce2e1e77afb3a70b55d0fdfe1b31-vnan
    instance_element-hq__element-web-71fe08ea0f159ccb707904d87f0a4aef205a167c-vnan
    instance_element-hq__element-web-772df3021201d9c73835a626df8dcb6334ad9a3e-vnan
    instance_element-hq__element-web-776ffa47641c7ec6d142ab4a47691c30ebf83c2e
    instance_element-hq__element-web-7c63d52500e145d6fff6de41dd717f61ab88d02f-vnan
    instance_element-hq__element-web-9a31cd0fa849da810b4fac6c6c015145e850b282-vnan
    instance_element-hq__element-web-9bf77963ee5e036d54b2a3ca202fbf6378464a5e-vnan
    instance_element-hq__element-web-a692fe21811f88d92e8f7047fc615e4f1f986b0f-vnan
    instance_element-hq__element-web-b7fea97bb68c6628a644580076f840109132f074-vnan
    instance_element-hq__element-web-ee13e23b156fbad9369d6a656c827b6444343d4f-vnan
    instance_element-hq__element-web-f14374a51c153f64f313243f2df6ea4971db4e15
    instance_flipt-io__flipt-1737085488ecdcd3299c8e61af45a8976d457b7e
    instance_flipt-io__flipt-1dceb5edf3fa8f39495b939ef9cc0c3dd38fa17d
    instance_flipt-io__flipt-406f9396ad65696d58865b3a6283109cd4eaf40e
    instance_flipt-io__flipt-492cc0b158200089dceede3b1aba0ed28df3fb1d
    instance_flipt-io__flipt-6fe76d024ee0c50ddb09c86f4ae0bd4c208fd65f
    instance_flipt-io__flipt-8bd3604dc54b681f1f0f7dd52cbc70b3024184b6
    instance_flipt-io__flipt-967855b429f749c28c112b8cb1b15bc79157f973
    instance_flipt-io__flipt-9f8127f225a86245fa35dca4885c2daef824ee55
    instance_flipt-io__flipt-a42d38a1bb1df267c53d9d4a706cf34825ae3da9
    instance_flipt-io__flipt-abaa5953795afb9c621605bb18cb32ac48b4508c
    instance_flipt-io__flipt-b2cd6a6dd73ca91b519015fd5924fde8d17f3f06
    instance_flipt-io__flipt-b4bb5e13006a729bc0eed8fe6ea18cff54acdacb
    instance_flipt-io__flipt-b68b8960b8a08540d5198d78c665a7eb0bea4008
    instance_flipt-io__flipt-c12967bc73fdf02054cf3ef8498c05e25f0a18c0
    instance_flipt-io__flipt-c1728053367c753688f114ec26e703c8fdeda125
    instance_flipt-io__flipt-cd2f3b0a9d4d8b8a6d3d56afab65851ecdc408e8
    instance_flipt-io__flipt-d966559200183b713cdf3ea5007a7e0ba86a5afb
    instance_flipt-io__flipt-dae029cba7cdb98dfb1a6b416c00d324241e6063
    instance_flipt-io__flipt-dbe263961b187e1c5d7fe34c65b000985a2da5a0
    instance_flipt-io__flipt-e2bd19dafa7166c96b082fb2a59eb54b4be0d778
    instance_flipt-io__flipt-e5fe37c379e1eec2dd3492c5737c0be761050b26
    instance_flipt-io__flipt-e91615cf07966da41756017a7d571f9fc0fdbe80
    instance_flipt-io__flipt-ea9a2663b176da329b3f574da2ce2a664fc5b4a1
    instance_flipt-io__flipt-ee02b164f6728d3227c42671028c67a4afd36918
    instance_future-architect__vuls-0ec945d0510cdebf92cdd8999f94610772689f14
    instance_future-architect__vuls-3c1489e588dacea455ccf4c352a3b1006902e2d4
    instance_future-architect__vuls-4c04acbd9ea5b073efe999e33381fa9f399d6f27
    instance_future-architect__vuls-5af1a227339e46c7abf3f2815e4c636a0c01098e
    instance_future-architect__vuls-6682232b5c8a9d08c0e9f15bd90d41bff3875adc
    instance_future-architect__vuls-6eff6a9329a65cc412e79b8f82444dfa3d0f0b5a
    instance_future-architect__vuls-7eb77f5b5127c847481bcf600b4dca2b7a85cf3e
    instance_future-architect__vuls-8659668177f1feb65963db7a967347a79c5f9c40
    instance_future-architect__vuls-86b60e1478e44d28b1aff6b9ac7e95ceb05bc5fc
    instance_future-architect__vuls-878c25bf5a9c9fd88ac32eb843f5636834d5712d
    instance_future-architect__vuls-9aa0d87a21bede91c2b45c32187456bb69455e92
    instance_future-architect__vuls-a76302c11174ca081f656c63a000ffa746e350af
    instance_future-architect__vuls-aaea15e516ece43978cf98e09e52080478b1d39f
    instance_future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a-v1151a6325649aaf997cd541ebe533b53fddf1b07
    instance_future-architect__vuls-d576b6c6c15e56c47cc3e26f5878867677d4a9ea
    instance_future-architect__vuls-e4728e388120b311c4ed469e4f942e0347a2689b-v264a82e2f4818e30f5a25e4da53b27ba119f62b5
    instance_future-architect__vuls-edb324c3d9ec3b107bf947f00e38af99d05b3e16
    instance_future-architect__vuls-f6509a537660ea2bce0e57958db762edd3a36702
    instance_gravitational__teleport-0cb341c926713bdfcbb490c69659a9b101df99eb
    instance_gravitational__teleport-0ecf31de0e98b272a6a2610abe1bbedd379a38a3-vce94f93ad1030e3136852817f2423c1b3ac37bc4
    instance_gravitational__teleport-1316e6728a3ee2fc124e2ea0cc6a02044c87a144-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-2b15263e49da5625922581569834eec4838a9257-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-3587cca7840f636489449113969a5066025dd5bf
    instance_gravitational__teleport-3a5c1e26394df2cb4fb3f01147fb9979662972c5-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-3ff75e29fb2153a2637fe7f83e49dc04b1c99c9f
    instance_gravitational__teleport-47530e1fd8bfb84ec096ebcbbc29990f30829655-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-65438e6e44b6ce51458d09b7bb028a2797cfb0ea-vce94f93ad1030e3136852817f2423c1b3ac37bc4
    instance_gravitational__teleport-6eaaf3a27e64f4ef4ef855bd35d7ec338cf17460-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-73cc189b0e9636d418c4470ecce0d9af5dae2f02-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-82185f232ae8974258397e121b3bc2ed0c3729ed-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-87a593518b6ce94624f6c28516ce38cc30cbea5a
    instance_gravitational__teleport-b5d8169fc0a5e43fee2616c905c6d32164654dc6
    instance_gravitational__teleport-b8fbb2d1e90ffcde88ed5fe9920015c1be075788-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-ba6c4a135412c4296dd5551bd94042f0dc024504-v626ec2a48416b10a88641359a169d99e935ff037
    instance_gravitational__teleport-bb69574e02bd62e5ccd3cebb25e1c992641afb2a
    instance_gravitational__teleport-c1b1c6a1541c478d7777a48fca993cc8206c73b9
    instance_gravitational__teleport-db89206db6c2969266e664c7c0fb51b70e958b64
    instance_gravitational__teleport-e6d86299a855687b21970504fbf06f52a8f80c74-vce94f93ad1030e3136852817f2423c1b3ac37bc4
    instance_gravitational__teleport-fb0ab2b9b771377a689fd0d0374777c251e58bbf
    instance_internetarchive__openlibrary-0a90f9f0256e4f933523e9842799e39f95ae29ce-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c
    instance_internetarchive__openlibrary-0d13e6b4bf80bced6c0946b969b9a1b6963f6bce-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c
    instance_internetarchive__openlibrary-0dc5b20fa186f9714f8a838178597e69f549d026-v2d9a6c849c60ed19fd0858ce9e40b7cc8e097e59
    instance_internetarchive__openlibrary-25858f9f0c165df25742acf8309ce909773f0cdd-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_internetarchive__openlibrary-322d7a46cdc965bfabbf9500e98fde098c9d95b2-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_internetarchive__openlibrary-3aeec6afed9198d734b7ee1293f03ca94ff970e1-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_internetarchive__openlibrary-3f7db6bbbcc7c418b3db72d157c6aed1d45b2ccf-v430f20c722405e462d9ef44dee7d34c41e76fe7a
    instance_internetarchive__openlibrary-5069b09e5f64428dce59b33455c8bb17fe577070-v8717e18970bcdc4e0d2cea3b1527752b21e74866
    instance_internetarchive__openlibrary-5de7de19211e71b29b2f2ba3b1dff2fe065d660f-v08d8e8889ec945ab821fb156c04c7d2e2810debb
    instance_internetarchive__openlibrary-5fb312632097be7e9ac6ab657964af115224d15d-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-6afdb09df692223c3a31df65cfa92f15e5614c01-v08d8e8889ec945ab821fb156c04c7d2e2810debb
    instance_internetarchive__openlibrary-6fdbbeee4c0a7e976ff3e46fb1d36f4eb110c428-v08d8e8889ec945ab821fb156c04c7d2e2810debb
    instance_internetarchive__openlibrary-72321288ea790a3ace9e36f1c05b68c93f7eec43-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-7c8dc180281491ccaa1b4b43518506323750d1e4-v298a7a812ceed28c4c18355a091f1b268fe56d86
    instance_internetarchive__openlibrary-7edd1ef09d91fe0b435707633c5cc9af41dedddf-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c
    instance_internetarchive__openlibrary-7f6b722a10f822171501d027cad60afe53337732-ve8c8d62a2b60610a3c4631f5f23ed866bada9818
    instance_internetarchive__openlibrary-8a5a63af6e0be406aa6c8c9b6d5f28b2f1b6af5a-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-910b08570210509f3bcfebf35c093a48243fe754-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-92db3454aeaa02f89b4cdbc3103f7e95c9759f92-v2c55207218fb8a0138425cbf7d9675272e240b90
    instance_internetarchive__openlibrary-9bdfd29fac883e77dcbc4208cab28c06fd963ab2-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c
    instance_internetarchive__openlibrary-a48fd6ba9482c527602bc081491d9e8ae6e8226c-vfa6ff903cb27f336e17654595dd900fa943dcd91
    instance_internetarchive__openlibrary-ba3abfb6af6e722185d3715929ab0f3e5a134eed-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c
    instance_internetarchive__openlibrary-bdba0af0f6cbaca8b5fc3be2a3080f38156d9c92-ve8c8d62a2b60610a3c4631f5f23ed866bada9818
    instance_internetarchive__openlibrary-c12943be1db80cf1114bc267ddf4f9933aca9b28-v2c55207218fb8a0138425cbf7d9675272e240b90
    instance_internetarchive__openlibrary-c4eebe6677acc4629cb541a98d5e91311444f5d4-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_internetarchive__openlibrary-d40ec88713dc95ea791b252f92d2f7b75e107440-v13642507b4fc1f8d234172bf8129942da2c2ca26
    instance_navidrome__navidrome-09ae41a2da66264c60ef307882362d2e2d8d8b89
    instance_navidrome__navidrome-28389fb05e1523564dfc61fa43ed8eb8a10f938c
    instance_navidrome__navidrome-3977ef6e0f287f598b6e4009876239d6f13b686d
    instance_navidrome__navidrome-3f2d24695e9382125dfe5e6d6c8bbeb4a313a4f9
    instance_navidrome__navidrome-55730514ea59d5f1d0b8e3f8745569c29bdbf7b4
    instance_navidrome__navidrome-55bff343cdaad1f04496f724eda4b55d422d7f17
    instance_navidrome__navidrome-669c8f4c49a7ef51ac9a53c725097943f67219eb
    instance_navidrome__navidrome-677d9947f302c9f7bba8c08c788c3dc99f235f39
    instance_navidrome__navidrome-69e0a266f48bae24a11312e9efbe495a337e4c84
    instance_navidrome__navidrome-6b3b4d83ffcf273b01985709c8bc5df12bbb8286
    instance_navidrome__navidrome-8383527aaba1ae8fa9765e995a71a86c129ef626
    instance_navidrome__navidrome-874b17b8f614056df0ef021b5d4f977341084185
    instance_navidrome__navidrome-87d4db7638b37eeb754b217440ab7a372f669205
    instance_navidrome__navidrome-9c3b4561652a15846993d477003e111f0df0c585
    instance_navidrome__navidrome-de90152a7173039677ac808f5bfb1e644d761336
    instance_navidrome__navidrome-e12a14a87d392ac70ee4cc8079e3c3e0103dbcb2
    instance_protonmail__webclients-0200ce0fc1d4dbd35178c10d440a284c82ecc858
    instance_protonmail__webclients-1917e37f5d9941a3459ce4b0177e201e2d94a622
    instance_protonmail__webclients-3a6790f480309130b5d6332dce6c9d5ccca13ee3
    instance_protonmail__webclients-51742625834d3bd0d10fe0c7e76b8739a59c6b9f
    instance_protonmail__webclients-5e815cfa518b223a088fa9bb232a5fc90ab15691
    instance_protonmail__webclients-5f0745dd6993bb1430a951c62a49807c6635cd77
    instance_protonmail__webclients-6f8916fbadf1d1f4a26640f53b5cf7f55e8bedb7
    instance_protonmail__webclients-715dbd4e6999499cd2a576a532d8214f75189116
    instance_protonmail__webclients-7b833df125859e5eb98a826e5b83efe0f93a347b
    instance_protonmail__webclients-7e54526774e577c0ebb58ced7ba8bef349a69fec
    instance_protonmail__webclients-8afd9ce04c8dde9e150e1c2b50d32e7ee2efa3e7
    instance_protonmail__webclients-8be4f6cb9380fcd2e67bcb18cef931ae0d4b869c
    instance_protonmail__webclients-944adbfe06644be0789f59b78395bdd8567d8547
    instance_protonmail__webclients-c5a2089ca2bfe9aa1d85a664b8ad87ef843a1c9c
    instance_protonmail__webclients-dfe5604193d63bfcb91ce60d62db2f805c43bf11
    instance_protonmail__webclients-f080ffc38e2ad7bddf2e93e5193e82c20c7a11e7
    instance_protonmail__webclients-fc9d535e9beb3ae30a52a7146398cadfd6e30606
    instance_qutebrowser__qutebrowser-0fc6d1109d041c69a68a896db87cf1b8c194cef7-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-16de05407111ddd82fa12e54389d532362489da9-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-1a9e74bfaf9a9db2a510dc14572d33ded6040a57-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-322834d0e6bf17e5661145c9f085b41215c280e8-v488d33dd1b2540b234cbb0468af6b6614941ce8f
    instance_qutebrowser__qutebrowser-36ade4bba504eb96f05d32ceab9972df7eb17bcc-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-3d01c201b8aa54dd71d4f801b1dd12feb4c0a08a-v5fc38aaf22415ab0b70567368332beee7955b367
    instance_qutebrowser__qutebrowser-6b320dc18662580e1313d2548fdd6231d2a97e6d-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-6dd402c0d0f7665d32a74c43c5b4cf5dc8aff28d-v5fc38aaf22415ab0b70567368332beee7955b367
    instance_qutebrowser__qutebrowser-7f9713b20f623fc40473b7167a082d6db0f0fd40-va0fd88aac89cde702ec1ba84877234da33adce8a
    instance_qutebrowser__qutebrowser-85b867fe8d4378c8e371f055c70452f546055854-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-8cd06741bb56cdca49f5cdc0542da97681154315-v5149fcda2a9a6fe1d35dfed1bade1444a11ef271
    instance_qutebrowser__qutebrowser-9b71c1ea67a9e7eb70dd83214d881c2031db6541-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-a84ecfb80a00f8ab7e341372560458e3f9cfffa2-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-bedc9f7fadf93f83d8dee95feeecb9922b6f063f-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-c0be28ebee3e1837aaf3f30ec534ccd6d038f129-v9f8e9d96c85c85a605e382f1510bd08563afc566
    instance_qutebrowser__qutebrowser-cc360cd4a34a126274c7b51f3b63afbaf3e05a02-v5fc38aaf22415ab0b70567368332beee7955b367
    instance_qutebrowser__qutebrowser-e15d26630934d0b6415ed2295ac42fd570a57620-va0fd88aac89cde702ec1ba84877234da33adce8a
    instance_qutebrowser__qutebrowser-e34dfc68647d087ca3175d9ad3f023c30d8c9746-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-e64622cd2df5b521342cf4a62e0d4cb8f8c9ae5a-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_qutebrowser__qutebrowser-e70f5b03187bdd40e8bf70f5f3ead840f52d1f42-v02ad04386d5238fe2d1a1be450df257370de4b6a
    instance_qutebrowser__qutebrowser-ec2dcfce9eee9f808efc17a1b99e227fc4421dea-v5149fcda2a9a6fe1d35dfed1bade1444a11ef271
    instance_qutebrowser__qutebrowser-ed19d7f58b2664bb310c7cb6b52c5b9a06ea60b2-v059c6fdc75567943479b23ebca7c07b5e9a7f34c
    instance_qutebrowser__qutebrowser-ef5ba1a0360b39f9eff027fbdc57f363597c3c3b-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d
    instance_tutao__tutanota-f3ffe17af6e8ab007e8d461355057ad237846d9d-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf
    """.split()
)
"""A fixed 200 of SWE-Bench Pro V2's 642 tasks; each repository keeps its share of the 642."""
SWEBENCH_PRO_SMOKE_PACKAGES = frozenset(
    """
    instance_NodeBB__NodeBB-97c8569a798075c50e93e585ac741ab55cb7c28b-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
    instance_NodeBB__NodeBB-a5afad27e52fd336163063ba40dcadc80233ae10-vd59a5728dfc977f44533186ace531248c2917516
    instance_ansible__ansible-a7d2a4e03209cff1e97e59fd54bb2b05fdbdbec6-v0f01c69f1e2528b935359cfe578530722bca2c59
    instance_ansible__ansible-be59caa59bf47ca78a4760eb7ff38568372a8260-v1055803c3a812189a1133297f7f5468579283f86
    instance_element-hq__element-web-772df3021201d9c73835a626df8dcb6334ad9a3e-vnan
    instance_element-hq__element-web-f14374a51c153f64f313243f2df6ea4971db4e15
    instance_flipt-io__flipt-b68b8960b8a08540d5198d78c665a7eb0bea4008
    instance_flipt-io__flipt-c1728053367c753688f114ec26e703c8fdeda125
    instance_future-architect__vuls-8659668177f1feb65963db7a967347a79c5f9c40
    instance_future-architect__vuls-f6509a537660ea2bce0e57958db762edd3a36702
    instance_gravitational__teleport-b8fbb2d1e90ffcde88ed5fe9920015c1be075788-vee9b09fb20c43af7e520f57e9239bbcf46b7113d
    instance_gravitational__teleport-c1b1c6a1541c478d7777a48fca993cc8206c73b9
    instance_internetarchive__openlibrary-8a5a63af6e0be406aa6c8c9b6d5f28b2f1b6af5a-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4
    instance_internetarchive__openlibrary-92db3454aeaa02f89b4cdbc3103f7e95c9759f92-v2c55207218fb8a0138425cbf7d9675272e240b90
    instance_navidrome__navidrome-677d9947f302c9f7bba8c08c788c3dc99f235f39
    instance_navidrome__navidrome-e12a14a87d392ac70ee4cc8079e3c3e0103dbcb2
    instance_protonmail__webclients-715dbd4e6999499cd2a576a532d8214f75189116
    instance_protonmail__webclients-fc9d535e9beb3ae30a52a7146398cadfd6e30606
    instance_qutebrowser__qutebrowser-bedc9f7fadf93f83d8dee95feeecb9922b6f063f-v2ef375ac784985212b1805e1d0431dc8f1b3c171
    instance_qutebrowser__qutebrowser-cc360cd4a34a126274c7b51f3b63afbaf3e05a02-v5fc38aaf22415ab0b70567368332beee7955b367
    instance_tutao__tutanota-de49d486feef842101506adf040a0f00ded59519-v10a26bfb45a064b93f4fc044a0254925037b88f1
    instance_tutao__tutanota-f3ffe17af6e8ab007e8d461355057ad237846d9d-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf
    """.split()
)
"""Two tasks from each of SWE-Bench Pro V2's 11 repositories, drawn with ``random.Random(0)``."""
SWEBENCH_PRO_200 = replace(
    SWEBENCH_PRO,
    recipe="swebench-pro-200",
    benchmark="swebench-pro-200",
    title="SWE-Bench Pro V2, stratified 200",
    source=replace(SWEBENCH_PRO_SOURCE, packages=SWEBENCH_PRO_200_PACKAGES),
    count=200,
    images=SWEBENCH_PRO.recipe,
)
SWEBENCH_PRO_SMOKE = replace(
    SWEBENCH_PRO,
    recipe="swebench-pro-smoke",
    benchmark="swebench-pro-smoke",
    title="SWE-Bench Pro V2, 22-task smoke",
    source=replace(SWEBENCH_PRO_SOURCE, packages=SWEBENCH_PRO_SMOKE_PACKAGES),
    count=22,
    images=SWEBENCH_PRO.recipe,
)
SWEBENCH_PRO_DATASETS = (SWEBENCH_PRO, SWEBENCH_PRO_200, SWEBENCH_PRO_SMOKE)
SWEBENCH_PRO_MAX_INPUT_TOKENS = 131072
SWEBENCH_PRO_MAX_OUTPUT_TOKENS = 16384
"""Marin's agentic eval policy for mini-swe-agent: a 128K context and 16K replies."""
