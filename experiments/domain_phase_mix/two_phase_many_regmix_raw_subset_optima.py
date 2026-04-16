# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact subset-fit raw-optimum deployments for the RegMix baseline."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import cache
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)

REGMIX_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_regmix_raw_subset_optima_uncheatable_bpb"
)
REGMIX_RAW_SUBSET_OPTIMA_BASE_RUN_ID = 450
REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES = (20, 40, 60, 80, 100, 140, 180, 220, 242)
REGMIX_RAW_SUBSET_OPTIMA_POLICY = "feature_bayes_linear_regmix_raw_optimum"
REGMIX_RAW_SUBSET_OPTIMA_VARIANT = "regmix"
REGMIX_RAW_SUBSET_OPTIMA_TUNING_METHOD = "LightGBM+sample_bank"
_SUMMARY_DIR = Path(__file__).resolve().parent / "exploratory" / "two_phase_many"
SUMMARY_JSON_PATH = _SUMMARY_DIR / "two_phase_many_regmix_raw_summary.json"
_EMBEDDED_ROWS_GZIP_BASE64 = (
    "H4sIAIzp3GkC/+Wd6XIcyZGE34W/ZbN5H3qVtTUYCPYMIeGgAJCjWdm8+/qXjaO7uiszC8LMj9WKmqWkPqqrMiM8PDw8//tfny6v"
    "nr5f3lz8uLy5/nL5tPty8fnb509/vft+c/OXTz/rn4+/Xj7cXlx9vX/c3V08fL+7uLu83X3666fPl4+7m+u73cW3h/tv9w9P1/d3"
    "lzefzrxHn/xdb7A/mWqLj6GWYmqxIR6+9mH3y8Pu6eLy6cJ++qv5yfiQfMo+Vpusz6nqxXe7y4fd49PF/efH3cMPXei5i3l8erh8"
    "uv75evfl05l3PP24+HL9+HR5d7XjW6wp0ZgUQjY+1Zrzmbe8XX0uVb8glmxSSrqe+29P17ffby9u73/sLm53l3cX377qMviSH4+6"
    "K7sfL7ex/ffm4vbynxe/7q5/+frUfmIo0ZbqUrU+FN2Q+PrCx+/fuKMXn3c3979e2F3Q65//R7v8FF9rdLa6Wn1MzubX13U+5Pn9"
    "j5/++q/n/8Lw1y/3N7eX/uLy4Z/XP9pn22z4ybWaWIytwf/l5TVXV/91qc++vPty8WX3eP3L3cVXfWJ7k7PGWleSjcnbmFPqvElX"
    "tv8RscSgN+hd3hgbDt9y9XB9u2tv9N9uvj8+7+yoDacnrEVhbTnYN0prV3+/UDzaf2ZJQQHIsv2N9+5tm/x6/ffrb7sv15fPId75"
    "TKgxWfHa2+fPu767v7i6v71VXLt6uPxVj+8f+5+qAOG0wJQxtXhrenu1/t/91cPFty8/P768WA+XDFJc0rOw9e21r5eqX3i7f6kC"
    "iRIs+0h/e9me+9fubi+01H/8tr+S50gea6yKHEmbM3t/cNWPv93phl3df9k97yDFJpOsdhzZ5SWOvr3y+k4L7vsVIfw5n/I8ozZe"
    "Kvt8evx6nsfzNWSFd6NHEHPOeu3ylf+4fEYyTnE8OWW44mw6+m3tdU9fr+/+rrizv96kFWsVIfRcbC3p95eUbc9n6BIUR2pQwDe6"
    "opJmMrTumt7kSOhaRrFOJGiXhMS0TrTCtQfNcdRay9DFRMU4bcSgFZCnEnTlnuuPnm60dTZBO91VFxT+HD/LziZoRVd+UFHU9tHG"
    "uDFBCwWaqGiunyhc6+OmDK01m7PV2klZAMGFmQSt9BVdYb8GrVKbJhK0gm3JwSkwZ5cU5+YTtDZO0C/k9gAJ/HyCVnRI2hPaFZUV"
    "U6YStFcY1l4KSQ9CAM3O5GfFOJuCTZ73lcUDXCToU5i8lqBtqmxqbw0frAzdS9C66GwVuAnIoZxN5m9PXPmbDOGCnn06xoOj/KyI"
    "b30Ldfql0W/Jz6qdclYyUFI00ddRetYm0lZXKhS2UkV0jPnP5Wf9ai0zVVBFV6mbNszPQdENvK+loSs7zrfn8nPbXokdpkVsyub0"
    "DGZX8Bca1pr0C3Azk5/1Tu0kRSceYfUHpcaZBO2U6/Q1AGmFWUGQlQStWE3sUA5VkIxhNT9rbbcErSwubJldPz/rKkMxApZ6JkFb"
    "w3UTNB8O9FME9Eo8L8jlfIoWmhBCEFCK3FGTRylaEDer7rGKkbqmmFZTNLhZ6CtET7h6id7rGVoYoYB19S9woe+kaF1y1l42oDun"
    "H7mWorUqXCkgQEetlbopmjBCmlZNGPUD8+/k6Ac9sSvYkxdW4JU3sMJ02he6WqG7zNU+fv/8uKMwP+QZGutifzJJpamJNmgXUuis"
    "vPqA+uCvhl3y6fW1j9f/q//Fqdx/+t4i5tUPJfRvurT2MaX9DJWm2lAqK4QOlYp8K7vfXv/z/c2XRmosmRkXvWl33ik8OOWHw3fp"
    "8fMtLyzKU0MpQTWRF4gX/Mq2AaWDNyw+vaiU0MtUxCh51Hz00tvH5wXjFceCbpPTi0ILCc+v0o7dPVwoDd7sn8P14+0zym0lreWa"
    "HWn39R33n/+mTH39Y//JnmQsRKVtrnxp3O9/+Y/mx0xh51qnysupBoUS+GB+zPwkrKsKV/FB0K3qI9bZMmVtlcsqPoy17Qe/hy0T"
    "CMjCJ7aRTf5D2TIhz8A98E4/RpkiT7Fl2rgqHSPvIl6FGTBOQDOpwlqqsDn+ohUw7ijGnH6zMIDQSpxB49EX1cbeBd1vAaE4DceD"
    "cptXTHLkS4XVWcIsusju181TSnDH0GMCkCvfZcWZVoqrnE6bALnTfmdpqThyZHA3hchV1ikaRj1w1YAmTCBybSRvEtWzcp2exgbKT"
    "FBLRYdR6U29Xso8ZZbJ3cqBwltOK2aOMoNe1XLWahFKW2DbVUjeSDA9yaLi3PoeItezFhDW6opwGSl1ILlKHiXR5IR+9emuS5mp2"
    "Nei09U7ytsWiFYhuTCScFIi1sMMbkHkZ8LRLCL3egRa5FWZy/E8RojcQanrDugNChSKXiNErhpdW5YLVClTR3gc1EWc91xTPebez"
    "9JlXgkBOi42gmI7W6b9magZfNsCbjNd5lT7auNVJXLljHBAdJ6D47CmisuKX8ouh+D/GI47WHInmK096Q+472M0ToBWTBSaq9qNM"
    "cUBW2YIRLWooNUiq4cvP4PGtSGopSjXhPZzLl003nJMinq1Qt5rTbIOx/WI4ekt2UZlbVyF4547q6uNiqLeltNXnuBxFXQCLi7kq"
    "n9rY6zjcX0aJJXuNFTrKh5v69Fq7QoOKWi50sfjUNAhwTh4Cvc84MyE9PWz9ERqNSkcF5SraVp3o8DgBVtqDDnPpGkhXiXbrLpYyy"
    "VNcWYuae/Cz2kp2GzzRJZWkBYKUBT1ggN5PkmroMwKJwbQ5V6f2gRpBtto4KL0F1O2JmnqDQVOQm8G9Wzra1UCthayMKPq5gUkWUn"
    "SFONZEBMOrCzv0NkkbeGuKAqTJ9C7tCVJJ1szRE+0ENN5njY7gaYztBmASfWVMnuk1TeTo5UfsoKSHryijGCE7/a1FGgSnTqVu+6Y"
    "kV32tZShBVAKhHtwxyTsCW2mta2cqSXEj+7naC1Ra/iZ+uAYj3feiDZTThfibcFJGeN4q/eTtNInuynuu7d13NUSMlE1rd2rte2OW"
    "bAzOZrUTGNPOaQIR4xJs5zgABVSYuMH4iBL2xbqhd4KjaCY/OY0TS/IEYiDAh+d881punV9UmV1Vn+Ap89maZ+BxkrmwdQDHL3gz"
    "OAEaMiGqPKgrqZpJfACq24AjcUMelrCt0oDQtVKfFqRrp+lvdEmNaS5KBgScuxm6WQUW/WDatElG+cHWVq5USmGtBGEyoIJnb5WBQ"
    "5bokYJ3oVhX8sqXtNu12MNR/dkkaS1dBLCCqtnro2T13J00G5nLSo6aH2MulqKvbpUKF9t+VTGnJkwogOz6Tmym/4kziz0ObMpgoy"
    "+mgf1aYnoDh7TamcJMi80GaIpeugJcq1HkCnrlWxyRJZAxz6dp8jovgnleFJ7OOC7VhmyoABGl0Q3nude1xgyrUs9b8Far2iX4r/"
    "DkO2fgsB5lxVTGDMUrcDz0LbEOiume4jqJbBu4cf+EFJM16E6QHci6HaxkT6eE1NWVv6xuVWKrWW8QoophwDTU2HtO6qAd5BiQQF"
    "VW1PpruqjXnVoH0KKnbnCCbStnFGrR/cisD5FiWlTaK0ryCnp5ZqmGDGjciGhPaKCiBNQu2Ggktiljh7LPCFGm8y0DpZAT5zlwxJ"
    "gEqYvtSJrI9JW4hEcQ13laelt6087FH8OOjrH5MxMf1qBySdDCyoWF2ba04aFV71STY4K3BtgNjS5qrDWew+bqDDSNlIqqwqzxOPF"
    "tc6FGdXlirc06+pyFa/ibG6g1n0OGbxnXQ9n55J0LxydfYH5Y23iCRmGEgphYfLhmA06IcNUYgkmIbIrQmK+25822qVwk56ub45bg"
    "LbXR7cmadGCC7ZskY/pzurN2sD8/mF7WnswaD/QgWPd1KF6TAEE9aYwKKrUieY0ytKkEFLKERG0woY5tldRvQeZV4+r6Dk6LAojF4"
    "R3SlgxvANn89O8V9Qj4WfbB9qKP46bp6WTbDZ5jQ5TKPVWYackUNsBK7ggxAhOtNTgUkMZ0GEBOK5QpEcNL5a7QFtPQgCzwXjtOJP"
    "6QBsmEOUUwuMQy1A/li0sl0CXInI2vtOcDtwqFRLWof8dkmFcRGL9aD/XGjrN6UiX71lrISQVV4G2Mo3VDkG4ioyjT4axKxQ/FOIU"
    "/+OIChM4LAL6IRAdapprWVkeY32B8yVO5eckGKe6I+vGxwUrv8qFqUZuTQvFpGXlv5Khtd2rCoZQoPYWjH43Qyvk6a5FrTq4qfmWV"
    "VEOYweEaOHpt3asKgV9SUoQKDDDxo4VIcqSd7OCyKTIOwsU0CQrbPCpjpWCi+K6MJyPBsZwAxmmDKtyUqusqlYxaT5LR0M/RgE1s9"
    "TKrMgbwKhEprpIpdQxabOapT2CZAhngSXvU1fl7VqDS5BeYam6XscqqPKPUbWUknSxvpuktbpVtWY9FHTUIffYMIU3IUm4Cl/tMW8"
    "0IsNUU/hE0SjIVH0IWzRkdKnBBSgU6nHT8nyW9mjVWmOvyftHaVrLX4WHfhJd1NdE0snTpGhE5yHCf/o4FJExxVIU7gniC9Q0macF"
    "hl0mXwPRQtmcpyNzBOgEre5krv22lQIlBHL22j+g3ZU8fVq6rfStAkpwKiolypxGKjJLgmZahmadKX1CLCoupKiEbvSG2k3T2msQ"
    "S0xThPKqPF5P00qRCOrpo2nVrifpQDsA8aEekC2nirDTllVQ5koo2avWdPYdCRkKT49OLkG5rCVpCnx9lKE1kWu/YWWrHr/TGlBe"
    "DwJQQzbMZBCCnmEN1v55ArL0EWQYooXaWsuCMO3OjMgwRThAtWI/4MKFHht2lvyCv1IVQo8W5X12Q/bLNPKraRH23bq4Qn+pXMkWC"
    "QbEp7Zo+ff5L8XKgSpMS7okR2815T7/hdy7QrEqPamK+WNEYRV8BKUf6R5Rgv0BBBisgMpcgLez5M/zBFhBdqO1q8q1oFJ6F/+FQI"
    "CpqRqrivvsP1QUhrySRo/uGAKOuRHKHBgs4da1CaapbnNkBq+S3Qv87JQoTClGUVgwPmUvQDGFsFWp6KkKpuj3GJfmOTAblcmKZVF"
    "mdhNk2CRcpGQZkJcCNdmpjQiPfVIEw/Gtmyb0rChMK5GzUICmEHYEXWebqzKHd3fOAOwizYsFOl+nnIeX5+u/mlBGOhMBUtRItQ6"
    "m5vRUG1ZEKdGFFbRzJFgenqMyASkNbrO0IHXdPf1cgE12DIbeySYMriwn1ZjULpfcMQnALvB3gDyEYbzbiQJ02faLFxjKLm3IGylH"
    "MOcl2pPJjA3AGyleL4PVsRNwOtAOiiot5XmxiSYzYzLJXpHAJ9xszkwcAAEFPQJQ2xNx8WohFEgRXb3Dg4sAOEEcir0eMmbOTAlF9"
    "UBin2KE8EfjsGdw9YOxkNYNRDGklujwIpKuMZvkO8Oy+slttbP99pFzBhkW4aSMEYWDIMfSNL7irDMzBaTWDazKN1gPgN9fSPGVQ"
    "OXoSLMt9k8/aGkSXa916yNm8ktHhmp8XWIrhnLZFhJ+ZTR2rKOrkOikR302IXCa01r8Doyvhno1nDBIwpMXwue1ZeT2POYBDMqtpi"
    "qZUqhznWoChIAiH2lhjpFgUERW6/HlJqE0M9RYLr1SckkwGzUmfxMdqbAQIgSp9OzQkRpU4XciBKmW1RM8SsS6+H4BJOxLTvrjZFi"
    "uiat3RjKRsm2IhTa+Ypowh9Dq7XszKhTU5oEwekw5XGQVa+lqFioPV62jFB6tMgWbSIiahfn0zNSQ8bvEkAixrnsrMhJf8opiJYF"
    "j7WanVFHqajQWo5UmKWr19ZNDizHxgXZTnKOzH8yeMXgYQyum5xbixJNyH5WK3STs9X2R/miC16yaqMJyty6jfxapn435ObMnLCB"
    "YXXoxIfZWRW96lfAe6X5V0ZaMEep6xHlouQ0YZie6ccyz0RAxAtjyH3RtFQ48MI/9T3Ul1VYqFxeZdAhbm9RIctVGcTQHqq3bnbWp"
    "tbOzDgCKOutNajo0yHBUPxF2Lk2PqniGS1FCWyo/Ar/V7MzU4UBNaXCSii1z3zp0RJGclPwxSN+6iQ7n9aYA+YLxYQiOXyEy2ldro"
    "0dijZlgCzTkx4nZyE79zwQK8Synpvb0I8KoJRAV6fs20tqBsgwLc2NUMb1/dycI+MPUSEKWbwdDk/SB6BS0arZm3/8SdxX+QDuS0u"
    "0ibqS/o9pQjfmviL5kB2O/Dt1hWAB5BnJfnRfTVyhwhpfXSKjEt6nMRMmaFJUssUKfjJNlLHChDVnC0FZOjU1+A9gwtpj6DBhJ49"
    "3nQlDRW0Zz0lR5XuiM/dHzEdSbRjXeiKu6VY/nAoTLKU/iMMMcyHwJitaMCEUuidWbyBVv4sKU0DzHvGnakuY74+lwhyeNVjB1Kg"
    "a000BbaMHCINj9eNTzVNIG62tMimzNFopbooKa84QzCagHVzIndegdkYrmyO0DVLueSqMQgofH93uWjYQYZVuARyG4rLb2mvO+GWp"
    "1ME4LZaNajCnn6r0DX4QZp+C2vvRJJr+JIfjhtkK1oZEENCITYcdgt+AtS16aEolOLuwAWt7SH+hKZrV1ds5KkyXqMhOS8Px7PsE"
    "2huSFBzJfI9HlGu7fiWRTm9RndhmgnpMmGXoQgkVcwx/bLd3ArZtDdj/CM8y2RJSd/BCj405QnwCvIkbe82NQHIRXc7CFGwAt2Mu"
    "kbRsd/wlYtQsyYgp8XlCeDgFUh07Twb5mPEoz23P9PHTmKEA1DX3jXLI3TGPmNIdH3LXYkTlfy2UycFQC1lGhxpyW6GDcMzihOQ29"
    "mTZuIsHIag2nleYNHkNiRnxks47hehTV3QIX3JWGGIL7cDhljIc0yYRbnCbtYPYz52Jjm3O4jXbGC2OISesruRNAicwRzR2l5uRiw"
    "V2knCfmkysEjNrVfMMaD0g3THekSYbQ5QHHzfFFlhS2q2bQolBoaAKwfCb+hUZUzfHT4yeB+MNWPtUK0SVewAxMuIB7NtEMOTpJVF"
    "XRgeOmljbgeJKO2adpTi6NBJr1hgUPzlVgPXzfkZ11yG8GlLc4LK5vxscb723H1aNf2joS3SPjggNLyHHPeCBzOBVYDAzDB1v5aeu"
    "XRY00JJld3h2Y1nebAC7a0nwTmoSqR91RjixULU9K2hE3rC7jNFZp8IQzRW6Bfhm5mS7wi7HVHNNHcSjkoZpmg9Ck7koLulEj6vE2"
    "FYqGlxIuvXpi6rRvtK84GTNYA3MH19Jsz6dtA4Q2Ha9QrGQyZMeFy1EafXm8YUd5mwE3Zkjgnbsy5HTJhzAyaMORoKhkSfQGsszcn"
    "GkqdN6Bj0fT7keUSGcRgqzVYFYDpP3dFKlN+F+hPiwK6RYcFn0CL2SkriYcyGMWSGOQuNK+X6VQcy28adY+TcFJXt///pMJVuKvXJ"
    "XipbYxvi+ng6TMFAeB1nDtMOfFnlwxRgBKU5P4ODN9519CRlnwU+IkaK5mOd93GLh1JxYAk/OUtJjVUBC8pD0bsZNkwhA1rYQrm7R"
    "c5agdsclcfttbhYlkWuXzvnHZtp1a0cy5vTvELMpuaEg81HznlaIMZAuvaW5dgRlvvW3nMmiKC70d4MIW809nWtu4liqYYZNgwbgb"
    "I/JgXP55hmbPc5UCC2vh3YaEvzOaLu2p8GhRYibDh+krup+gXL/sU5Rz1nX+R9kB88+jzXew54YuGBgIY/2t65Vp7BrtwUlZidx57"
    "b2L5RqUSdBFxK7NNhrolrGLbD+K87pRGopj11Yd54yLvqK2UBQEYKi7Gq0dmTvvIcOUCbjt2YDWOSzFnPI8y47AwJsbL3A8JV4NiZ"
    "7dzeDvjpcDaSw4pieMa7a0d5q2ykeeHq/MmRhzxrO/i86jtjKtuPtDLN9zK3/A446RuNaY0VhtstX7smDDtNK+dnNBTQycO6/JLqE"
    "cg9S4cJ6SdvsB1qZ6TEgdUYuINxIWaXj6D8GT4MIZseAQcWV3c0xnB+RgMVK8f0FOiWjomJb8fHl+Yyhj/7WBrmMNNH5ac9lE6bxG"
    "9d58QALcZ8hTPHT91JXr3GuLtMUWIVZUfiMPSBioq+edYPz55sAxDK+vvTnnyZE4fpPjBTxjmP2OuFqSFK034DAjeFoTJlB4q3reA"
    "n3gjIO8JUgm6iImZ7dXnzfFi7DTWyJNiL8w0rrKQprVI+mUocJ2jDqTEM0VvMnOpGqwNsBGI74B4f/akpyoCZPSd1aPMvuwNrGbrS"
    "+1VgwZA4bzkZxyEaFYDTk+MIsS3yMOUSRVNtvAbyw6Q8DHdqRNAoxPKCXlqXh7VhTdt0Ja5vNYazCcc+B8iuXDoCbs6/Y/8pjwNPT"
    "D9F42eIQxIgaOG0dSIPa5QmBZlWTnbbBikRBSosKKaGtKVfhX+fESxQzkDPMVRvu2ZsWhpOFuiPI/N9TF8rxmxwY3nhyXre7SCRco"
    "XzGa72zo+ytNVlx+Yy1sYT3zFJKXiCKiQF/E2Dze8QcHuik0G+ag4dPM6ePBkYNWcmOHEe56qAO0DRwVWWZF4N7U/zNAf7mIpsquR"
    "w1KQ57wi6T88Kg0jKBkdEY+LKUkeTXF2IPVbstNQcpGnLIRzadBE1wKnP5uEJ0Ua4rdjW0nUxj0kxQhNGpqE53dre8TgMxiEvpE3s"
    "10XcmQLP4prnX2u1tTwNSYkqOOpPNBPqMIuiruDx3/w9/yxKLLgPOH6y7TvkBBx5Fo9ps7MEmGN0Fls/5ZyQy7/Pf/mmKhKwZ7bP"
    "5zihBhMsa6eT10AnYI3/QvqB1EB33Dbu9Pf/+T+hi7Afz6sAAA=="
)


@dataclass(frozen=True)
class RegmixRawSubsetOptimumSummary:
    """Summary for one exact subset-fit raw-optimum deployment."""

    subset_size: int
    run_id: int
    run_name: str
    policy: str
    objective_metric: str
    variant_name: str
    tuning_method: str
    predicted_optimum_value: float
    subset_best_observed_run_name: str
    subset_best_observed_bpb: float
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    optimum_move_mean_phase_tv_vs_prev: float | None
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_regret_at_1: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_lower_tail_optimism: float
    tuning_cv_depopt_best8: float
    tuning_cv_rawopt_nearest_tv: float
    phase0_max_weight: float
    phase1_max_weight: float
    phase0_support_below_1e4: int
    phase1_support_below_1e4: int
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def regmix_raw_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one exact subset-fit raw optimum."""
    if subset_size not in REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return REGMIX_RAW_SUBSET_OPTIMA_BASE_RUN_ID + REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES.index(subset_size)


def regmix_raw_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one exact subset-fit raw optimum."""
    return f"baseline_regmix_raw_optimum_k{subset_size:03d}_uncheatable_bpb"


def _weights_from_phase_weights(
    phase_weights: dict[str, dict[str, float]],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _embedded_rows() -> list[dict[str, Any]]:
    payload = gzip.decompress(base64.b64decode(_EMBEDDED_ROWS_GZIP_BASE64))
    return json.loads(payload.decode("utf-8"))


def _load_rows() -> list[dict[str, Any]]:
    if SUMMARY_JSON_PATH.exists():
        payload = json.loads(SUMMARY_JSON_PATH.read_text())
        rows = payload.get("rows")
        if isinstance(rows, list):
            return rows
    return _embedded_rows()


@cache
def _raw_subset_rows_by_size() -> dict[int, dict[str, Any]]:
    rows = _load_rows()
    rows_by_size = {int(row["subset_size"]): row for row in rows}
    missing = set(REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES).difference(rows_by_size)
    if missing:
        raise ValueError(f"Missing embedded/disk RegMix subset summaries for sizes: {sorted(missing)}")
    return rows_by_size


def _summary_to_dict(summary: RegmixRawSubsetOptimumSummary) -> dict[str, Any]:
    return {
        "subset_size": summary.subset_size,
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "policy": summary.policy,
        "objective_metric": summary.objective_metric,
        "variant_name": summary.variant_name,
        "tuning_method": summary.tuning_method,
        "predicted_optimum_value": summary.predicted_optimum_value,
        "subset_best_observed_run_name": summary.subset_best_observed_run_name,
        "subset_best_observed_bpb": summary.subset_best_observed_bpb,
        "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
        "fullswarm_chosen_value": summary.fullswarm_chosen_value,
        "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
        "nearest_observed_run_name": summary.nearest_observed_run_name,
        "nearest_observed_value": summary.nearest_observed_value,
        "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
        "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
        "tuning_objective": summary.tuning_objective,
        "tuning_cv_rmse": summary.tuning_cv_rmse,
        "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
        "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
        "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
        "tuning_cv_depopt_best8": summary.tuning_cv_depopt_best8,
        "tuning_cv_rawopt_nearest_tv": summary.tuning_cv_rawopt_nearest_tv,
        "phase0_max_weight": summary.phase0_max_weight,
        "phase1_max_weight": summary.phase1_max_weight,
        "phase0_support_below_1e4": summary.phase0_support_below_1e4,
        "phase1_support_below_1e4": summary.phase1_support_below_1e4,
        "family_shares": summary.family_shares,
        "phase_weights": summary.phase_weights,
    }


@cache
def regmix_raw_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> tuple[RegmixRawSubsetOptimumSummary, ...]:
    """Return exact subset-fit raw-optimum summaries for the RegMix baseline."""
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    rows_by_size = _raw_subset_rows_by_size()
    summaries: list[RegmixRawSubsetOptimumSummary] = []
    for subset_size in subset_sizes:
        if subset_size not in REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES:
            raise ValueError(f"Unsupported subset size: {subset_size}")
        row = rows_by_size[subset_size]
        phase_weights = row["phase_weights"]
        weights = _weights_from_phase_weights(phase_weights, packet.base.domain_names)
        summaries.append(
            RegmixRawSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=regmix_raw_subset_optimum_run_id(subset_size),
                run_name=regmix_raw_subset_optimum_run_name(subset_size),
                policy=REGMIX_RAW_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                variant_name=REGMIX_RAW_SUBSET_OPTIMA_VARIANT,
                tuning_method=REGMIX_RAW_SUBSET_OPTIMA_TUNING_METHOD,
                predicted_optimum_value=float(row["predicted_optimum_value"]),
                subset_best_observed_run_name=str(row["subset_best_observed_run_name"]),
                subset_best_observed_bpb=float(row["subset_best_observed_bpb"]),
                fullswarm_chosen_run_name=str(row["fullswarm_chosen_run_name"]),
                fullswarm_chosen_value=float(row["fullswarm_chosen_value"]),
                fullswarm_regret_at_1=float(row["fullswarm_regret_at_1"]),
                nearest_observed_run_name=str(row["nearest_observed_run_name"]),
                nearest_observed_value=float(row["nearest_observed_value"]),
                nearest_observed_tv_distance=float(row["nearest_observed_tv_distance"]),
                optimum_move_mean_phase_tv_vs_prev=(
                    None
                    if row["optimum_move_mean_phase_tv_vs_prev"] is None
                    else float(row["optimum_move_mean_phase_tv_vs_prev"])
                ),
                tuning_objective=float(row["tuning_objective"]),
                tuning_cv_rmse=float(row["tuning_cv_rmse"]),
                tuning_cv_regret_at_1=float(row["tuning_cv_regret_at_1"]),
                tuning_cv_foldmean_regret_at_1=float(row["tuning_cv_foldmean_regret_at_1"]),
                tuning_lower_tail_optimism=float(row["tuning_lower_tail_optimism"]),
                tuning_cv_depopt_best8=float(row["tuning_cv_depopt_best8"]),
                tuning_cv_rawopt_nearest_tv=float(row["tuning_cv_rawopt_nearest_tv"]),
                phase0_max_weight=float(np.max(weights[0])),
                phase1_max_weight=float(np.max(weights[1])),
                phase0_support_below_1e4=int(row["phase0_support_below_1e4"]),
                phase1_support_below_1e4=int(row["phase1_support_below_1e4"]),
                family_shares=family_shares(packet, weights),
                phase_weights=phase_weights,
            )
        )
    return tuple(summaries)


def regmix_raw_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> str:
    """Return the exact subset-fit raw-optimum summaries as JSON."""
    return json.dumps(
        [_summary_to_dict(summary) for summary in regmix_raw_subset_optima_summaries(subset_sizes)],
        indent=2,
        sort_keys=True,
    )


def regmix_raw_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> pd.DataFrame:
    """Return a flat summary frame for the exact subset-fit raw-optimum sweep."""
    return pd.DataFrame(
        [
            {
                "subset_size": summary.subset_size,
                "run_id": summary.run_id,
                "run_name": summary.run_name,
                "policy": summary.policy,
                "variant_name": summary.variant_name,
                "tuning_method": summary.tuning_method,
                "predicted_optimum_value": summary.predicted_optimum_value,
                "subset_best_observed_run_name": summary.subset_best_observed_run_name,
                "subset_best_observed_bpb": summary.subset_best_observed_bpb,
                "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
                "fullswarm_chosen_value": summary.fullswarm_chosen_value,
                "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_value": summary.nearest_observed_value,
                "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
                "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
                "tuning_objective": summary.tuning_objective,
                "tuning_cv_rmse": summary.tuning_cv_rmse,
                "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
                "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
                "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
                "tuning_cv_depopt_best8": summary.tuning_cv_depopt_best8,
                "tuning_cv_rawopt_nearest_tv": summary.tuning_cv_rawopt_nearest_tv,
                "phase0_max_weight": summary.phase0_max_weight,
                "phase1_max_weight": summary.phase1_max_weight,
                "phase0_support_below_1e4": summary.phase0_support_below_1e4,
                "phase1_support_below_1e4": summary.phase1_support_below_1e4,
            }
            for summary in regmix_raw_subset_optima_summaries(subset_sizes)
        ]
    )


def create_regmix_raw_subset_optimum_weight_config(subset_size: int) -> WeightConfig:
    """Return the weight config for one exact subset-fit raw optimum."""
    summary = next(
        summary for summary in regmix_raw_subset_optima_summaries((subset_size,)) if summary.subset_size == subset_size
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_regmix_raw_subset_optima_weight_configs(
    subset_sizes: tuple[int, ...] = REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> tuple[WeightConfig, ...]:
    """Return weight configs for the exact subset-fit raw-optimum sweep."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in regmix_raw_subset_optima_summaries(subset_sizes)
    )
