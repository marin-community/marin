# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Approved source-config fingerprints for dated verified eval cohorts."""

from collections.abc import Mapping
from types import MappingProxyType

SEPTEMBER_16_VERSION = "eval-policy-2026-09-16-verified"
SEPTEMBER_24_VERSION = "eval-policy-2026-09-24-verified"

POLICY_SOURCE_DIGESTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        SEPTEMBER_16_VERSION: MappingProxyType(
            {
                "math500": "sha256:622f156a16b9c1d73981265af8e24cadde2408c5ff406c27758ce2bac72c3e94",
                "aime24": "sha256:fdd8a49682d1b0819f11e4a286618f794b32ebb23b988ef9bcd48c820693955a",
                "humanevalplus": "sha256:65d3c6f2bca9644c4b046e4a890486977c3d668dea13c8805b138ad3fd115381",
                "mbppplus": "sha256:56776f854d49021ef37612385a952d88f626e5a9ade946572018f6b587acfe04",
                "olympiadbench": "sha256:10381f2a6c1ed37d875f7d15c04f9e8cc8b2bf85b8e9932257677184974e3be1",
                "mmlu-pro": "sha256:b90aa9313a83fe5401f63dbc2adf84dde3c00a8a64d26f77a961bc501bf8e7d1",
                "gpqa-diamond": "sha256:06db58b9303fb7a995e538db17d963074d124ad2ab90d953186a5018abd060ed",
                "cruxeval": "sha256:1dc5459ee474b0f36ffe910d8d8b7bc09eea99d63aa753eed90d05368f53c962",
                "financebench": "sha256:8196f11bd745da1a6f250ea5a5fdf476295157fe83b3176a8ffcead9bcdfc4c4",
                "ifbench": "sha256:03436f9e8ae68a63a0ad878dfe55eb9c2f3b8db86da754b0db7b122819950310",
                "mrcr": "sha256:4d6a2ce2dd2b66de185a97606a100b9e42c95fb13a489497f312cde32f13109b",
                "gsm8k-0shot": "sha256:36b3187f0c8a1690dc7ccf2cda94dc5cd81e2c447555c2b4cc7c163ff0c2ec9b",
                "mmlu": "sha256:70a0955acd05773be7148f287793ceef548b7c850726efdee353cee30c5c6088",
                "piqa": "sha256:1fb08bab8459043cdbcaad32c914164a9ae10602647a3294b097e95d8cdfe0ca",
                "winogrande": "sha256:a91b3e3e01d08f5db27e8ffde92ace2303639a04d810cea2c88709b4393a7e5b",
                "openbookqa": "sha256:2fe5f82fd4d648874950e9d9f4ff9e022bf7550370878b46709fe9d685197d55",
                "boolq": "sha256:db3c4eb96299bfe818eae7f84b2ecbe625ee941f8597a8e5eb5605dda9d3bba9",
                "truthfulqa": "sha256:ad8f2533c225ae92ecf71b3dd099e50bb0fe96328a5b93df45e2cce5cd1136ff",
                "triviaqa": "sha256:c8f7ed5096413c3952cd4aa19f6807c349101165d8fe1a6a920c973d3913c15c",
                "swebench-recovery": "sha256:2562df5138e42d1695a9fa4b7e3b963270cb9a2ec92647a54e870bc575c65629",
                "ot-tblite-recovery": "sha256:62f1fa60131fb54ca111c08d92d82e76f79ac3b3f7582814159a7a5b069f5421",
                "tb2-recovery": "sha256:21ab39e53800aad6d0b89370bfb5409ab2c5b6e8a08c410d439dc687dc81a64e",
                "simpleqa-recovery": "sha256:549710acdc1c80dbddea2b53a4bf02f49bb2f3914d8a044fdfee483402d0e401",
                "ds-1000-local": "sha256:925e0762285639d16764e98b8bbc4f6d834689fb41c766f998178830875d025c",
            }
        ),
        SEPTEMBER_24_VERSION: MappingProxyType(
            {
                "math500": "sha256:6b6af99263e464b14b0fc18eb7affbf629063e113b11b5cacb488843103cab42",
                "humanevalplus": "sha256:324195d75222508fabdc0a7f1a9287c0b0674589fc229ef709898e424751743e",
                "mbppplus": "sha256:e98cffecfe9a1cf87708022412c153d6220246f595f86d73b884b642f3ba7191",
                "olympiadbench": "sha256:c3e63ede049b6c6f6f5f0a56c254ee7e139adae0c7acb1698ca4f6cd8e7a730a",
                "gsm8k-0shot": "sha256:90e0dfe23b8cd523c61782227dd47ccf0f2fb6dfab5629e997c2fcf281bd923f",
                "piqa": "sha256:1fb08bab8459043cdbcaad32c914164a9ae10602647a3294b097e95d8cdfe0ca",
                "winogrande": "sha256:a91b3e3e01d08f5db27e8ffde92ace2303639a04d810cea2c88709b4393a7e5b",
                "boolq": "sha256:db3c4eb96299bfe818eae7f84b2ecbe625ee941f8597a8e5eb5605dda9d3bba9",
                "truthfulqa": "sha256:ad8f2533c225ae92ecf71b3dd099e50bb0fe96328a5b93df45e2cce5cd1136ff",
                "triviaqa": "sha256:2ee4dddf7f70e172b1e859b0602da7e991e339f76e6968d6b02128d485eb13c8",
                "aime24": "sha256:5bf65d0b7b1628c04734a304b38d53234087f3179f70d85853cbe33012d35592",
                "mmlu-pro": "sha256:3f738359c5beb11ca0b6f35b86e1960be1d84a36d10e0fff00a240650a8bc3b5",
                "gpqa-diamond": "sha256:28438fbe808ba4a0e11ffaa952e82c6516102493a662c2c02d89051125605a1d",
                "cruxeval": "sha256:917e049fcaa9f21e4186b81a58846e43e5cc5b28a65e1c944c2d349da579194e",
                "financebench": "sha256:d695003c7eb6071afa74483d38f26726c39be1a606bbed822223c2eaa1ee893c",
                "ifbench": "sha256:cbe79d3b33dffcfc5189b9f349c548236bd979de3d7c504c766ff8b162e682c6",
                "mrcr": "sha256:86edabf46ad8bb610e2d1fc420ad29ba4778dfa41944efd6cda8f1978d920739",
                "nupa": "sha256:5e49ca859e3ea638bf1f9650a8739f90ce52ce64293bb94b9da6206b8fe29f52",
                "swebench-recovery": "sha256:c56bc78c6eebdbcf17dfb30ff952923f763388540a49893949758ad1f1fe260d",
                "ot-tblite-recovery": "sha256:1af2649c0cb5814d4c8804267ef74df0304c581814eca070b154920c2e326122",
                "tb2-recovery": "sha256:549b1afe346af5fbbf257b0a17699432c792e5ac22cfde87c2d0f47647a19364",
                "ds-1000-local": "sha256:6c6d14384f18de269c73b1d38293187ecb26628e63ac8d6ee0f93ede3f2ae9f3",
                "bfclparity-pi": "sha256:1f7128d01f83a2f0eaade116aeaeae964731377526c9e6c6746d24bf7d1f7bfc",
                "bixbench-pi": "sha256:495ff3f66915e994bbd62b89d0db6018e70967fb6bad1a15253d1b161827ce40",
                "tau3-pi": "sha256:f7448d65774840de7959ce6306e5de66ce56069a7a2ba663baed3e73d8a9f3ae",
                "sotopia-hard": "sha256:fe88ddac6d912cb98b1571fbd2abf67bd1571c2979ac11034502f824b62b9849",
            }
        ),
    }
)
