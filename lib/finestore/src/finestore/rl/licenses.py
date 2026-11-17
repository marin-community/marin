# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Marin license policy and revision-bound Atlas license receipts."""

import fnmatch
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from finestore.schema import SHA256_PATTERN

# Licenses a candidate may have: every rating of the Blue Oak Council's list of permissive licenses
# (https://blueoakcouncil.org/list.json, version 16), by SPDX id, and OpenMDW, a permissive license for models and
# data. A lowercased SPDX id is the Hugging Face card id. The list holds software licenses only, so the data licenses
# that only ask for attribution, or for derivatives under the same terms, are in DATA_LICENSES.
BLUE_OAK_LICENSES_BY_RATING = {
    "Model": "BlueOak-1.0.0",
    "Gold": "BSD-2-Clause-Patent",
    "Silver": (
        "ADSL Apache-2.0 APAFML BSD-1-Clause BSD-2-Clause BSD-2-Clause-FreeBSD BSD-2-Clause-NetBSD "
        "BSD-2-Clause-Views BSL-1.0 DSDP ECL-1.0 ECL-2.0 hdparm ImageMagick Intel-ACPI ISC Linux-OpenIB MIT "
        "MIT-Modern-Variant MIT-testregex MIT-Wu MS-PL MulanPSL-1.0 Mup PostgreSQL SHL-0.5 Spencer-99 UPL-1.0 "
        "Xerox Xfig"
    ),
    "Bronze": (
        "0BSD AFL-1.1 AFL-1.2 AFL-2.0 AFL-2.1 AFL-3.0 AMDPLPA AML AMPAS ANTLR-PD ANTLR-PD-fallback Apache-1.0 "
        "Apache-1.1 Artistic-2.0 Bahyph Barr bcrypt-Solar-Designer BSD-3-Clause BSD-3-Clause-Attribution "
        "BSD-3-Clause-Clear BSD-3-Clause-HP BSD-3-Clause-LBNL BSD-3-Clause-Modification "
        "BSD-3-Clause-No-Nuclear-License-2014 BSD-3-Clause-No-Nuclear-Warranty BSD-3-Clause-Open-MPI "
        "BSD-3-Clause-Sun BSD-4-Clause BSD-4-Clause-Shortened BSD-4-Clause-UC BSD-Source-Code bzip2-1.0.5 "
        "bzip2-1.0.6 CC0-1.0 CFITSIO Clips CNRI-Jython CNRI-Python CNRI-Python-GPL-Compatible Cube curl eGenix "
        "Entessa FTL fwlw HPND-Fenneberg-Livingston HPND-sell-regexpr HTMLTIDY IBM-pibs ICU Info-ZIP Intel "
        "JasPer-2.0 Libpng libpng-2.0 libtiff LPPL-1.3c LZMA-SDK-9.22 MIT-0 MIT-advertising MIT-CMU MIT-enna "
        "MIT-feh MIT-open-group MITNFA MTLL MulanPSL-2.0 Multics Naumen NCSA Net-SNMP NetCDF NICTA-1.0 "
        "NIST-Software NTP OGL-Canada-2.0 OLDAP-2.0 OLDAP-2.0.1 OLDAP-2.1 OLDAP-2.2 OLDAP-2.2.1 OLDAP-2.2.2 "
        "OLDAP-2.3 OLDAP-2.4 OLDAP-2.5 OLDAP-2.6 OLDAP-2.7 OLDAP-2.8 OML OpenSSL PHP-3.0 PHP-3.01 Plexus PSF-2.0 "
        "Python-2.0 Ruby Saxpath SGI-B-2.0 SMLNJ SunPro SWL Symlinks TCL TCP-wrappers UCAR Unicode-3.0 "
        "Unicode-DFS-2015 Unicode-DFS-2016 UnixCrypt Unlicense VSL-1.0 W3C X11 XFree86-1.1 xlock Xnet xpp Zlib "
        "zlib-acknowledgement ZPL-2.0 ZPL-2.1"
    ),
    "Lead": (
        "AAL Adobe-2006 Afmparse Artistic-1.0 Artistic-1.0-cl8 Artistic-1.0-Perl Beerware blessing Borceux "
        "BSD-2-Clause-Darwin CECILL-B check-cvs ClArtistic Condor-1.1 Crossword CrystalStacker diffmark DOC "
        "EFL-1.0 EFL-2.0 Fair FSFAP FSFUL FSFULLR Giftware GLWTPL HPND HPND-export-US IJG IJG-short Jam Kazlib "
        "Leptonica LPL-1.0 LPL-1.02 McPhee-slideshow MirOS mpich2 NAIST-2003 NASA-1.3 NBPL-1.0 Newsletr NLPL NRL "
        "NTP-0 OFFIS OGTSL OLDAP-1.1 OLDAP-1.2 OLDAP-1.3 OLDAP-1.4 psutils python-ldap Qhull Rdisc RSA-MD "
        "snprintf Spencer-86 Spencer-94 SSH-short TU-Berlin-1.0 TU-Berlin-2.0 Vim W3C-19980720 W3C-20150513 "
        "Wsuipa WTFPL xinetd XSkat Zed Zend-2.0 ZPL-1.1"
    ),
}
# Hugging Face card ids of CC BY, CC BY-SA (every version), ODC-By, and the
# public-domain dedications PDDL and ODC-PDDL.
DATA_LICENSES = frozenset(
    {f"cc-by{sa}-{v}" for sa in ("", "-sa") for v in ("1.0", "2.0", "2.5", "3.0", "4.0")}
    | {"odc-by", "odc-by-1.0", "pddl-1.0", "odc-pddl"}
)
ALLOWED_LICENSES = DATA_LICENSES | frozenset(
    spdx_id.lower() for spdx_ids in BLUE_OAK_LICENSES_BY_RATING.values() for spdx_id in spdx_ids.split()
)
# OpenMDW's card ids: openmdw-1.0, openmdw-1.1.
OPENMDW_LICENSE_PATTERN = "openmdw-[0-9].[0-9]"
# Card ids that name a license a person must read: custom terms, or no stated terms.
REVIEW_LICENSES = frozenset({"other", "unknown"})
# The evidence kind that names the task's own license, not the dataset card's.
TASK_LICENSE_FILE_KIND = "task_license_file"
# The license id a receipt reports when its card names no license.
MISSING_LICENSE_ID = "missing"
# Card id fragments that forbid a use Marin needs: non-commercial or no derivatives.
REJECTED_LICENSE_FRAGMENTS = ("-nc", "-nd", "noncommercial")


class LicenseVerdict(StrEnum):
    ALLOWED = "allowed"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


def license_verdict(licenses: tuple[str, ...]) -> LicenseVerdict:
    """Classify a dataset card's license ids; every id must be allowed for the dataset to be allowed."""
    if not licenses:
        return LicenseVerdict.NEEDS_REVIEW
    verdicts = {_one_license_verdict(card_id.strip().lower()) for card_id in licenses}
    for verdict in (LicenseVerdict.REJECTED, LicenseVerdict.NEEDS_REVIEW):
        if verdict in verdicts:
            return verdict
    return LicenseVerdict.ALLOWED


def _one_license_verdict(card_id: str) -> LicenseVerdict:
    if any(fragment in card_id for fragment in REJECTED_LICENSE_FRAGMENTS):
        return LicenseVerdict.REJECTED
    if card_id in REVIEW_LICENSES:
        return LicenseVerdict.NEEDS_REVIEW
    if card_id in ALLOWED_LICENSES or fnmatch.fnmatchcase(card_id, OPENMDW_LICENSE_PATTERN):
        return LicenseVerdict.ALLOWED
    return LicenseVerdict.NEEDS_REVIEW


LICENSE_RESOLVER_VERSION = "atlas-license-v2"
LICENSE_REGISTRY_ISSUE = "https://github.com/marin-community/marin/issues/11497"


class LicenseClass(StrEnum):
    PERMISSIVE = "permissive"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"
    MISSING = "missing"


class LicenseEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    kind: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    license_ids: list[str]
    quoted_lines: list[str] = Field(min_length=1)
    restrictive_terms: bool = False
    unidentified_third_party_terms: bool = False


class LicenseDecisionKind(StrEnum):
    EXCEPTION = "exception"
    BUNDLED_THIRD_PARTY = "bundled_third_party"
    MODEL_GENERATED_OUTPUTS = "model_generated_outputs"
    BAN = "ban"


def _timestamp_with_timezone(value: str, what: str) -> str:
    """Require an ISO timestamp that names its timezone."""
    if datetime.fromisoformat(value).tzinfo is None:
        raise ValueError(f"{what} must include a timezone")
    return value


def _registry_issue_url(value: str) -> str:
    """Require a blocked-source registry issue link, optionally with an anchor."""
    if value.split("#")[0] != LICENSE_REGISTRY_ISSUE:
        raise ValueError("License decisions must link to the blocked-source registry")
    return value


class LicenseDecision(BaseModel):
    """An explicit human/coordinator decision for one source revision and use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: LicenseDecisionKind
    approved: bool
    who: str = Field(min_length=1)
    when: str
    source_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    scope: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence_urls: list[str] = Field(min_length=1)
    issue_url: str = Field(pattern=r"^https://github.com/marin-community/marin/issues/[0-9]+(?:#.*)?$")

    @field_validator("when")
    @classmethod
    def validate_time(cls, value: str) -> str:
        return _timestamp_with_timezone(value, "License decision time")

    @field_validator("issue_url")
    @classmethod
    def validate_issue(cls, value: str) -> str:
        return _registry_issue_url(value)


class LicenseSourcePin(BaseModel):
    """The one source pin declared by a wrapper, with its declaration evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    url: str = Field(min_length=1)
    declaration: LicenseEvidence


class LicenseReceipt(BaseModel):
    """Primary evidence at an immutable pin; classification is derived, never asserted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    dataset_id: str
    revision: str = Field(min_length=1)
    resolved_at: str
    resolver_version: str
    evidence: list[LicenseEvidence]
    bundled_third_party: bool
    model_generated_outputs: bool
    decisions: list[LicenseDecision] = Field(default_factory=list)
    resolved_source: LicenseSourcePin | None = None
    wrapper_chain: list[LicenseSourcePin] = Field(default_factory=list)

    @field_validator("resolved_at")
    @classmethod
    def validate_time(cls, value: str) -> str:
        return _timestamp_with_timezone(value, "License resolution time")

    @model_validator(mode="after")
    def validate_decisions(self) -> "LicenseReceipt":
        for decision in self.decisions:
            if (decision.source_id, decision.dataset_id, decision.revision) != (
                self.source_id,
                self.dataset_id,
                self.revision,
            ):
                raise ValueError("License decision belongs to another source revision")
        return self

    @property
    def license_ids(self) -> tuple[str, ...]:
        ids = {item for evidence in self.evidence for item in evidence.license_ids} - {MISSING_LICENSE_ID}
        return tuple(sorted(ids)) or (MISSING_LICENSE_ID,)

    @property
    def classification(self) -> LicenseClass:
        if any(item.restrictive_terms for item in self.evidence):
            return LicenseClass.RESTRICTED
        licenses = self.license_ids
        verdict = license_verdict(licenses)
        if verdict is LicenseVerdict.REJECTED:
            return LicenseClass.RESTRICTED
        if licenses == (MISSING_LICENSE_ID,):
            return LicenseClass.MISSING
        if verdict is LicenseVerdict.ALLOWED:
            return LicenseClass.PERMISSIVE
        return LicenseClass.UNKNOWN


class LicenseBan(BaseModel):
    """A source-wide user decision; lifting it requires a new recorded permissive grant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    upstream_id: str = Field(min_length=1)
    upstream_revision: str = Field(min_length=1)
    classification: LicenseClass
    license_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_urls: list[str] = Field(min_length=1)
    decided_at: str = Field(min_length=1)
    decision_by: str = Field(min_length=1)
    issue_url: str

    @field_validator("issue_url")
    @classmethod
    def validate_issue(cls, value: str) -> str:
        return _registry_issue_url(value)


def require_license(
    receipt: LicenseReceipt | None,
    source_id: str,
    dataset_id: str | None,
    revision: str | None,
    use: str,
    *,
    ban: LicenseBan | None,
) -> None:
    """Fail closed before ingest or admission, including on stale evidence, bans and registry bans.

    ``ban`` is the source's registry ban, or None after the caller checked the
    registry; every caller states that it looked.
    """
    if ban is not None and ban.source_id != source_id:
        raise ValueError(f"The registry ban names another source: {ban.source_id}")
    if receipt is None or not receipt.evidence:
        raise ValueError(f"License evidence is missing: {source_id}")
    if (receipt.source_id, receipt.dataset_id, receipt.revision) != (source_id, dataset_id, revision):
        raise ValueError(f"License evidence belongs to another source revision: {source_id}")
    if receipt.resolver_version != LICENSE_RESOLVER_VERSION:
        raise ValueError(f"License receipt needs resolution with the current resolver: {source_id}")
    scoped = [item for item in receipt.decisions if use in item.scope]
    decisions = {item.kind: item for item in scoped}
    if any(not item.approved or item.kind is LicenseDecisionKind.BAN for item in scoped):
        raise ValueError(f"License decision bans this source: {source_id}")
    check = check_license(receipt, ban)
    if check.verdict is not LicenseCheckVerdict.PASS and LicenseDecisionKind.EXCEPTION not in decisions:
        raise ValueError(
            f"License {receipt.classification.value} blocks ingest and admission: {source_id}; "
            f"{check.verdict.value} {check.rule_id.value}: {check.quote} ({check.evidence_url})"
        )


class LicenseCheckVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ESCALATE = "ESCALATE"


class LicenseCheckRule(StrEnum):
    PERMISSIVE = "P1"
    RESTRICTED = "F1"
    MISSING = "F2"
    BANNED = "F3"
    CUSTOM = "E1"
    CONFLICT = "E2"
    THIRD_PARTY = "E3"
    UNREACHABLE = "E4"


class LicenseCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    dataset_id: str
    revision: str
    verdict: LicenseCheckVerdict
    rule_id: LicenseCheckRule
    quote: str
    evidence_url: str


def check_license(receipt: LicenseReceipt, ban: LicenseBan | None = None) -> LicenseCheck:
    """Apply the dataset-level rules to the resolver's pinned primary evidence."""
    if ban is not None:
        return LicenseCheck(
            source_id=receipt.source_id,
            dataset_id=receipt.dataset_id,
            revision=receipt.revision,
            verdict=LicenseCheckVerdict.FAIL,
            rule_id=LicenseCheckRule.BANNED,
            quote=ban.reason,
            evidence_url=ban.issue_url,
        )
    evidence = receipt.evidence
    restricted = next(
        (
            item
            for item in evidence
            if item.restrictive_terms or license_verdict(tuple(item.license_ids)) is LicenseVerdict.REJECTED
        ),
        None,
    )
    terms = next((item for item in evidence if item.unidentified_third_party_terms), None)
    signals = [item for item in evidence if set(item.license_ids) - {MISSING_LICENSE_ID}]
    permissive = [item for item in signals if license_verdict(tuple(item.license_ids)) is LicenseVerdict.ALLOWED]
    unknown = [item for item in signals if license_verdict(tuple(item.license_ids)) is not LicenseVerdict.ALLOWED]
    dataset_permissive = [item for item in permissive if item.kind != TASK_LICENSE_FILE_KIND]
    if restricted is not None:
        verdict, rule, selected = LicenseCheckVerdict.FAIL, LicenseCheckRule.RESTRICTED, restricted
    elif terms is not None:
        verdict, rule, selected = LicenseCheckVerdict.ESCALATE, LicenseCheckRule.THIRD_PARTY, terms
    elif not signals:
        verdict, rule, selected = LicenseCheckVerdict.FAIL, LicenseCheckRule.MISSING, evidence[0]
    elif dataset_permissive and unknown:
        verdict, rule, selected = LicenseCheckVerdict.ESCALATE, LicenseCheckRule.CONFLICT, unknown[0]
    elif dataset_permissive:
        verdict, rule, selected = LicenseCheckVerdict.PASS, LicenseCheckRule.PERMISSIVE, dataset_permissive[0]
    else:
        verdict, rule, selected = LicenseCheckVerdict.ESCALATE, LicenseCheckRule.CUSTOM, signals[0]
    return LicenseCheck(
        source_id=receipt.source_id,
        dataset_id=receipt.dataset_id,
        revision=receipt.revision,
        verdict=verdict,
        rule_id=rule,
        quote=selected.quoted_lines[0],
        evidence_url=selected.url,
    )
