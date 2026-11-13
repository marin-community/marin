# Mochi EU disclosure drafts: publication review

Drafted 4 November 2026. These files are for review. They have not been published to Hugging Face or adopted as policy.

The proposed minimum package assumes that Marin Mochi qualifies for the non-systemic, free and open-source route under Articles 53(2) and 54(6). Under that route, the copyright policy and public training-content summary still apply. The exemption covers selected technical documentation and representative duties. Confirm the route before relying on it. See [Article 53](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-53) and [Article 54](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-54).

As of 4 November 2026, the provider reports that weight access is gated and no access requests have been approved. Repository visibility and completed uploads do not establish that another user has received the weights. This packet is being prepared for release; it makes no finding that the model has already been placed on the Union market.

Under [Article 3(9)–(10)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-3), market placement concerns the first supply for distribution or use on the Union market. C02 must establish the relevant facts and timing, including any other access channels or prior supplies. Complete the applicable release requirements before enabling external access; the legal trigger should not be assumed to depend only on the first approved download.

Article 53(1)(c) requires the provider to adopt and implement the copyright policy. Publishing the policy or a short public policy summary is encouraged, rather than required. The training-content summary must be public.

## Files and publication targets

| File | Purpose | Proposed destination after approval |
| --- | --- | --- |
| [TRAINING_CONTENT.md](TRAINING_CONTENT.md) | Public summary following the Commission's section numbering | Official provider website and the HF repository, linked from each covered checkpoint card |
| [COPYRIGHT_POLICY.md](COPYRIGHT_POLICY.md) | Proposed policy, including an electronic complaint route | Adopt internally; publish this short version or an approved summary on the official website and HF |
| [README.md](README.md) | Complete proposed HF main card | HF `main`, retaining the accurate upload status at publication |
| [CHECKPOINT_CARD_CHANGES.md](CHECKPOINT_CARD_CHANGES.md) | Exact addition and placement for checkpoint cards | Each covered `step-N` card after the upload verification workflow is finished or explicitly coordinated |

`CONFIRM Cxx` marks unresolved content. Remove the draft notices only after the relevant confirmations are complete. Do not upload this review file or the checkpoint editing instructions as public model documentation. Preserve version history and the evidence behind the final answers.

## Confirm before publication

| ID | Confirmation and evidence needed | Suggested owner |
| --- | --- | --- |
| C01 | Confirmed provider: Open Athena AI Foundation Inc., United States. Proposed model/copyright contact: privacy@openathena.ai. Remaining: confirm mailbox routing/monitoring and assign the accountable policy/complaint owner. | Legal / provider lead |
| C02 | Planned release date: 5 November 2026. Record actual first availability when access opens and confirm Union-market placement for each version. Check other channels and earlier supply against the gated/no-approved-access status reported by the provider. Confirm model dependencies and map each version to its summary. | Release owner + legal |
| C03 | Reconstructed: text modality, >10T size band, exact 23,106,103,007,622-token prepared corpus, source membership, seven required large datasets, selections and multilingual/content description. Provider confirmed latest actual acquisition/collection month: August 2026. No further C03 factual question; retain version/store-coverage review under C04. Exact language proportions are unavailable and are not asserted. | Data / training owner |
| C04 | Decide summary coverage per checkpoint. Shared ancestry alone is insufficient: reuse one summary only when its required content is identical. Otherwise publish version or phase summaries/deltas with explicit dependencies. | Data + release owner |
| C05 | Confirmed by provider: no private third-party training data, no provider-service user data, own synthetic data was public, and the public recipe covers all synthetic data used with nothing added outside it. Sections 2.2, 2.4 and the synthetic inventory in 2.5 have no further provider questions. Remaining: the separate other-source answer in 2.6. Available generator-summary links are an editorial task. | Data owner + release editor |
| C06 | Resolved by provider: no provider-operated or commissioned crawling. Marin consulted with Common Crawl on its independently created public Focus Crawl. The extracted corpus is withheld but reproducible. Report the upstream source under 2.1; no provider-crawler/domain-list attachment is applicable. | Data collection owner + legal |
| C07 | Provider confirmed ML-based classifiers for illegal-content filtering and historical reliance on upstream dataset providers for collection and rights-reservation handling. Describe that approach without claiming independent verification or complete removal. Remaining: policy adoption/effective date and confirmation of Code signatory status if any adherence is claimed. Signing the Code is voluntary; putting a copyright policy in place is required under Article 53(1)(c). | Data owner + policy owner |
| C08 | Close cumulative compute scope, especially applicable nonpublic synthetic generation (including discarded candidates), inherited/merged-weight training and other capability-contributing training. Confirm classification/designation status. The reviewed pre-training estimate is below the compute presumption; it is not a complete classification decision. | Training owner + legal |
| C09 | Confirm OpenMDW terms, intended gate/approval conditions at release, public availability of weights/architecture and sufficient model-use/integration information support the open-source exception. A public repository page alone does not establish qualification. Existing cards say a serving recipe and inference parity are unvalidated. Resolve documentation adequacy without claiming tests that have not run. If the exception is unavailable, prepare the additional Article 53 documentation and assess Article 54 representation. | Release owner + legal |
| C10 | Confirm proposed policy procedures have responsible owners and are operational, including complaints and proportionate output-risk measures. Complete separate personal-data/GDPR and other-law review; a training summary does not establish those obligations are met. | Policy owner + legal |
| C11 | Choose permanent public URLs, publish the completed summary on the official website and alongside distribution, check every card link, and establish an update owner. Preserve exact required warning, responsible-use paragraph and LICENSE. | Release owner |

The [Commission template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models) supplies the required summary fields. Its [FAQ](https://digital-strategy.ec.europa.eu/en/faqs/template-general-purpose-ai-model-providers-summarise-their-training-content) explains publication and updating. The [Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai) is voluntary; signing it is a separate decision. Do not claim signatory status or Code compliance from this draft.

## Evidence already available

| Evidence | Supported use and limit |
| --- | --- |
| [Source-pool reconstruction guide](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/docs/reproducibility/pretraining-source-pool.md) | Explains provider revisions and retrieval recipes. Explicitly requires the run's own revision and sampling configuration to identify its inputs. |
| [Datakit overview](https://marin.community/blog/2026/09/11/marin-data-pipeline-overview/) | Documents source attribution, normalization, deduplication and mixture construction. Its 152-source pool and 25.25T raw tokens are not a Mochi training-content inventory. |
| [Blog-pinned source registry](https://github.com/marin-community/marin/blob/64a9483cf1a928e6db97b83130d8a5d171a4aead/lib/marin/src/marin/datakit/sources.py) and download recipes | Upstream identities and source-specific selections, reconciled against the actual prepared-store membership. Presence in a moving catalog alone is insufficient evidence. |
| [Harrier recipe](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.py) and [public figure data](https://openathena.ai/assets/images/blog/marin-data-pipeline-overview/topic-quality-sankey.json) | Exact matching 23,106,103,007,622-token postprocessed corpus. Source-level figure counts match the frozen store; available-store volume is distinct from presented training tokens. |
| Existing reviewed checkpoint cards and export manifests | Architecture, context settings, tokenizer pin, export provenance and integrity checks. They disclose untested inference parity. |
| [Compute audit](../EVIDENCE.md#compute), [phase calculations](../EVIDENCE.md#compute) and [metric snapshot](../EVIDENCE.md#compute) | Local review evidence: context-corrected forward/backward estimate, source hashes and metric cross-check. Preserve these artifacts with the internal approval record; they are not public documentation links. |

## C03/C05 reconstruction

The [reconstruction audit](../EVIDENCE.md#training-content) records source-count checks and public evidence hashes. The public figure has 151 grouped rows; the frozen store has 292 source keys. Their counts match exactly after grouping the 142 Penfever trace datasets into one public figure row. This establishes source membership in the prepared corpus; it does not assert every record was sampled into every checkpoint.

Seven original packaged public datasets are above the reporting threshold, including Common Corpus at 716,822,128,569 postprocessed tokens. Subsets are combined by upstream collection, including both selected Dolma inputs. Their combined size is 18,301,410,547,202 tokens. The public-dataset denominator must be at least that large and at most the full 23,106,103,007,622-token corpus. Thus the 3% threshold lies between 549,042,316,416 and 693,183,090,229 tokens. All seven exceed the upper bound; the largest remaining packaged dataset, Nemotron-Pretraining-Code-v2 at 519,496,368,379 tokens, is below the lower bound. The required seven-dataset list is unchanged by classification of smaller own/generated/crawled sources. Exact section 2.1 shares are not asserted.

The public token-count viewer uses another catalog snapshot and tokenization-stage totals. It is not the postprocessing denominator. Source-language labels and actual decoded examples support a multilingual description without claiming language proportions. The recipe's `2026.08.18` version label and the public figure's 20 August 2026 generation timestamp alone do not establish an acquisition date; the provider has separately confirmed August 2026 as the latest actual acquisition/collection month.

The provider confirms no private third-party datasets and provider-generated public synthetic data, with all synthetic data covered by the public recipe and nothing added outside it. Public recipe/card evidence identifies identity conversations, KernelGym trajectories and contributor agent traces. Public third-party synthetic corpora remain under section 2.1; own/on-behalf generation is described under section 2.5 even when public. Available generator-summary links remain an editorial task, not another inventory-confirmation question. The identity dataset's public card does not assert a dataset license, so no blanket statement that every input is permissively licensed is supported.

The provider separately confirmed that no user data was used. Section 2.4 therefore records no training on interactions with the provider's models or other services. Public conversation-derived datasets and synthetic messages labeled `user` remain described under their source categories. Other sources under section 2.6 remain separately flagged; the synthetic inventory in section 2.5 is confirmed.

## C06 Focus Crawl clarification

The provider confirmed that Common Crawl independently created the Focus Crawl; Marin did not commission or direct that crawl. The WARCs are public; Marin's resiliparse extraction output is withheld but reproducible, as clarified by the provider. The public reproduction recipe documents archived HTML response extraction and normalization, including parser/boilerplate-processing dependencies. Do not claim that the processed corpus is itself publicly downloadable. It supplies neither evidence that Marin crawled the origin websites nor evidence about the original crawler's historical rights-reservation behavior. Do not identify the archive-download client's user agent as the web crawler.

Under the [Commission template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models), independent third-party public collections, including Common Crawl snapshots, belong under 2.1; preprocessing the same packaged collection does not create a separate source to report. Section 2.3 covers collection directly by the provider or on its behalf. The Focus Crawl therefore needs a public-source/provenance description under 2.1, without the 2.3 provider-crawler/domain attachment. Withholding Marin's own extraction output does not contradict the confirmation of no private datasets obtained from third parties. The provider subsequently confirmed no other provider-operated or commissioned crawling and historical reliance on upstream collection and rights-reservation handling. This closes the crawling question; C07 still covers policy adoption and the adequacy of the implemented approach.

## Compute wording

The retained release ends at step 378000. The reviewed estimate for that checkpoint is **2.6843 × 10²⁴ pre-training FLOP**, approximately **2.7 × 10²⁴**. The planned later step 390251 estimate, **2.7998 × 10²⁴**, must not be described as compute already used for the released step 378000.

| Phase interval | Context | Tokens per optimizer update | Estimated phase FLOP |
| --- | --- | --- | --- |
| 0–257553 | 4096 | 46,137,344 | 1.7793 × 10²⁴ |
| 257553–372000 | 16384 | 46,137,344 | 8.4839 × 10²³ |
| 372000–378000 | 65536 | 46,137,344 | 5.6586 × 10²² |

This counts active MoE computation, attention at the phase's context length and forward/backward work. It excludes activation recomputation as specified in the [GPAI compute guidance](https://digital-strategy.ec.europa.eu/en/library/guidelines-scope-obligations-general-purpose-ai-models-under-ai-act); BF16 operations are not discounted. The 48 checkpoint exports do not multiply the training compute by 48. Reviewed configurations show no inherited initialization or weight averaging, subject to closure of C08 across the full lineage.

The estimate is below the **10²⁵ FLOP** presumption in [Article 51](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-51). Independent additional-operation accounting gave approximately 2.7140 × 10²⁴ at step 378000. Neither estimate establishes that all regulatory compute scope is closed, or rules out capability-based classification or Commission designation. Retain the original audit, configurations, metric convention and source hashes for review.

## Release editing constraint

The running upload batch verifies checkpoint cards against frozen reviewed text. These drafts do not change that text, the runtime, or the plan. Publish approved disclosure files and card changes as a separate metadata update after coordinating with verification. Record that later card revision separately from the verified weight commits. Preserve public repository visibility and the existing gate; this package does not authorize access-setting changes.
