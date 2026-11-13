# Remaining EU AI Act tasks

Review status: 4 November 2026. C-identifiers refer to [the detailed review](drafts/REVIEW.md). Complete the applicable items before release; no policy adoption or publication has occurred through this drafting work.

## Decisions and operational confirmation

| Task | Suggested owner | What closes it |
| --- | --- | --- |
| C01: assign policy and complaint ownership | Provider lead | Name the accountable internal owner; confirm privacy@openathena.ai receives and routes model/copyright enquiries and complaints. The provider name and US establishment are already confirmed. |
| C07/C10: adopt and implement the copyright policy | Provider lead / policy owner | Approve a policy and effective date, assign its procedures, and establish source-review, complaint and escalation handling. The draft's future procedures are not evidence of historical implementation. Review whether the confirmed reliance on upstream collection/rights handling adequately implements the applicable duty. |
| C05: finish section 2.6's other-source answer | Data owner | Confirm Yes/No for other sources outside the documented categories, such as offline or commissioned human data; describe any such sources. This is separate from the already resolved synthetic-data and user-data questions. |
| C02: establish release timing | Release owner | Record actual first availability and the relevant Union-market placement date, accounting for any other supply channels. 5 November 2026 is the planned date. Preserve the current gate and approvals until the release decision. |
| C08/C09: record the basis for the intended non-systemic OSS route | Training / release owner | Check actual license, access conditions, public weights, architecture and usage information. Close any remaining applicable cumulative-compute scope using existing evidence. The pretraining calculation is already checked; public synthetic inventory is confirmed. If the exemption does not apply, assess the additional Article 53 documentation and Article 54 representation. |

The checked pretraining estimate for the latest retained checkpoint, step 378000, is approximately 2.7 × 10²⁴ FLOP, below the 10²⁵ FLOP systemic-risk presumption. This does not alone close classification or designation questions. No new compute run or weight download is needed to finish the documentary scope check.

## Editorial and release work

- [ ] C02/C04: map every covered checkpoint/version to the correct corpus, phase and summary; fill dependencies or confirm none. Reuse a summary where its required content is identical; use phase/version deltas where it differs. Add immutable identifiers when available.
- [ ] Section 2.5: add available generator-model training-summary links from public documentation. Synthetic inventory completeness is already confirmed.
- [ ] C09: finish sufficient versioned architecture/use information for the intended OSS route. Preserve the statement that inference parity and generic loading/serving have not been validated.
- [ ] C11: choose permanent website and HF document URLs; set an update owner and first-version history.
- [ ] Resolve every CONFIRM/EDITORIAL marker and update time-sensitive access and upload status. Remove draft banners only after approval.
- [ ] Publish the required completed training summary at the applicable release stage. Publishing the full copyright policy or a short public policy summary is optional; adopting and implementing the policy is required.
- [ ] Apply approved card changes through a coordinated metadata update after upload verification; preserve verified weight commits, required warning, responsible-use paragraph and OpenMDW 1.1 LICENSE. Check all final links and checkpoint coverage.

## Already confirmed; no repeat question needed

- Provider: Open Athena AI Foundation Inc., United States. Proposed contact: privacy@openathena.ai.
- Latest acquisition/collection month: August 2026.
- No private third-party datasets; no provider-model or provider-service user data.
- Public recipe covers all synthetic data used; nothing was added outside it.
- No provider-operated or commissioned crawling. Common Crawl independently created the public Focus Crawl; Marin consulted with it. Marin's extraction output is withheld but reproducible from public WARCs.
- Historical rights handling relied on upstream dataset providers; illegal-content filtering used ML-based classifiers. Neither statement guarantees complete rights verification or removal.
- Prepared corpus: 23,106,103,007,622 postprocessed text tokens. The seven large packaged public datasets and threshold calculation are reconstructed.

## Optional choices and separate scope

Signing the GPAI Code of Practice is voluntary. Confirm actual signatory status for the form; do not claim adherence unless true. A non-signatory still needs adequate means to demonstrate compliance with applicable underlying duties. [Article 53(4)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-53).

This packet covers the requested GPAI disclosure and policy work. It does not complete a separate GDPR/personal-data, general copyright or other-law assessment. These adjacent reviews should not be presented as extra mandatory fields in the training-summary form.
