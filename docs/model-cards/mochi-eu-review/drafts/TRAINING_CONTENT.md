# Marin Mochi: public summary of training content

**DRAFT — incomplete; not approved for publication.** Resolve each `CONFIRM` field using the [publication review](REVIEW.md). This draft follows the Commission's [training-content template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models). It describes training content; export integrity checks have a separate scope.

Version: draft 1. Last updated: 4 November 2026.
Previous summary versions: **[CONFIRM C11: public version links, or “None; first version.”]**

## 1. General information

### 1.1. Provider identification

| Field | Information |
| --- | --- |
| Provider name and contact | Open Athena AI Foundation Inc., United States. Proposed public contact: [privacy@openathena.ai](mailto:privacy@openathena.ai). [CONFIRM C01: confirm routing and monitoring for model/copyright enquiries.] |
| Authorised representative | [CONFIRM C01/C09: applicability of Article 54, including the qualifying open-source exception; representative details if required.] |

### 1.2. Model identification

Marin Mochi 535B-A23B is a base text language model. Weights are being staged in [open-athena/Marin-Mochi-535B-A23B](https://huggingface.co/open-athena/Marin-Mochi-535B-A23B), with checkpoint identifiers of the form `step-N`. Access is gated; as of 4 November 2026, the provider reports that no access requests have been approved. The retained release inventory has 48 checkpoints between step 6000 and step 378000. Each branch has its own model configuration and card.

| Field | Information |
| --- | --- |
| Versions covered by this summary | [CONFIRM C02/C04: enumerate the covered branches and their immutable release revisions. This draft does not yet establish that one summary can cover all 48.] |
| Model dependencies | [CONFIRM C02/C08: identify any previously placed model used as a base, with version and summary link, or confirm none. Distinguish continuation of the same pre-training lineage from modification of another released model.] |
| Planned release date | 5 November 2026. Actual first availability will be recorded when access opens. |
| First placement on the Union market | Pending confirmation at release. [CONFIRM C02: record the actual date for each covered version when access opens, accounting for any earlier supply through other channels. Gated access with no approvals is reported; upload completion is not a placement date.] |

### 1.3. Modalities, overall training data size and other characteristics

| Field | Information |
| --- | --- |
| Modality | Text-token streams, including natural-language text, source code and structured text. PDF-origin material is ingested as extracted text; SVG markup and genomic sequences occur as text representations. The documented recipe does not train on image pixels, audio waveforms or video frames. |
| Training data size after preprocessing, before training | **More than 10 trillion text tokens.** The prepared corpus contains **23,106,103,007,622 tokens** in 200 topic/quality cells, tokenized with `marin-community/marin-tokenizer`. This is the available postprocessed corpus, before training repetition and sampling. |
| Types of content | General web prose, news, discussion/forum text, books, scientific and mathematical material, government and legal documents, source code, development logs and software-agent transcripts. Inputs include original text, machine translations, synthetic rewrites, question-answering material and generated trajectories. |
| Latest data acquisition/collection | August 2026, as confirmed by the provider. The recipe identifies the fixed corpus as version `2026.08.18`; the documented recipe reads that corpus without live collection during training. |
| Linguistic characteristics | Multilingual text including English. Nonempty FinePDFs source-language selections include Czech, Dutch, French, German, Hungarian, Italian, Polish, Portuguese, Romanian, Spanish and Swedish, alongside Arabic, Chinese, Indonesian, Japanese, Russian, Thai and Ukrainian. FineTranslations includes original-language text paired with English translations. These are source-language labels; an exhaustive language audit and corpus-wide language proportions are unavailable. |
| Other relevant characteristics | The corpus covers international web content, with substantial software, scientific and English-language material. It is not a representative demographic sample. No validated corpus-wide demographic breakdown is available. Web and conversation-derived material can contain personal information and copyright-protected content; no blanket claim of their absence is made. |

The [public postprocessing counts](https://openathena.ai/assets/images/blog/marin-data-pipeline-overview/topic-quality-sankey.json) sum to the same total as the [pinned training-store recipe](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.json). The [Datakit overview](https://marin.community/blog/2026/09/11/marin-data-pipeline-overview/) describes normalization, deduplication, benchmark decontamination and topic/quality assignment. These figures describe the prepared corpus. The cumulative 17.44 trillion tokens presented through step 378000 measure training exposure, including sampling and repetition, and are a separate quantity.

## 2. List of data sources

### 2.1. Publicly available datasets

Were publicly available third-party datasets used? **Yes.** Modality used: **text**, including extracted and structured text.

The following original packaged datasets exceed the 3% reporting threshold. Counts combine subsets belonging to the same upstream dataset after preprocessing. They describe the prepared corpus, not the fraction of optimizer updates sampled from each dataset.

| Dataset identifier and access link | Selected portion and preprocessing | Postprocessed text tokens |
| --- | --- | --- |
| [NVIDIA Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | Six quality, synthetic, diverse-QA and translated-QA subsets; text normalization followed by shared corpus processing. | 4,870,276,073,044 |
| [Stack v3 train](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train) | Repository-context source-code documents, serialized with repository/file headers and directory context. | 3,238,000,622,909 |
| [FineTranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) | Original web text and its English translation combined in one text document; deterministic ordering of the two sides. | 2,983,288,691,545 |
| [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | Extracted text from 19 language selections, combined here as one upstream dataset. | 2,370,345,797,540 |
| [NVIDIA Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | Nine quality, synthetic and translated-text subsets; shared corpus processing. | 2,348,152,996,040 |
| [Dolma 3.5 pool](https://huggingface.co/datasets/allenai/dolma3.5_pool) | Selected OLMo-crawled PDF text and license-partitioned S2ORC papers, plus code-prose material. The duplicate FinePDFs component is excluded. | 1,774,524,237,555 |
| [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) | English-language records labeled Open Science, Open Government or Open Culture by the source, followed by shared processing. | 716,822,128,569 |

Other public inputs include HPLT web text; Common Pile scientific papers, books, legal and government material; institutional books; financial filings; additional code corpora; mathematics problems and solutions; multilingual web/PDF text; software-development logs; public conversation-derived material; and synthetic instruction, code and reasoning text. The [source registry](https://github.com/marin-community/marin/blob/64a9483cf1a928e6db97b83130d8a5d171a4aead/lib/marin/src/marin/datakit/sources.py) and its linked source recipes identify the upstream collections and selections. Provider/contributor-generated synthetic sources are described in section 2.5. Collection periods are not uniformly documented; where not documented, they are not known.

These public inputs also include Common Crawl's independently created science-focused `CC-SUPPLEMENTAL-2026-22` collection. Its WARC data is public. Marin collaborated with Common Crawl and extracted HTML text from the archived WARC response records. Marin's processed extraction is withheld from publication; the provider states that it is reproducible from the public WARCs using the documented extraction process. The [public extraction recipe](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/lib/marin/src/marin/datakit/download/common_crawl_focus.py) identifies the 4,573-WARC collection, parser/boilerplate-processing dependencies and subsequent normalization. The prepared corpus retains 25,389,343,944 tokens from this source. Common Crawl created the crawl independently; it was not commissioned or operated on Marin's behalf.

The list above combines original packaged datasets rather than treating their subsets as separate datasets. Each listed collection exceeds 3% even using the entire prepared text corpus as the denominator. All other packaged collections remain below 3% when the denominator is restricted to the seven listed public collections, so classifying smaller own/generated sources separately does not change the required list. This does not assert that all source material is free of rights or personal information.

### 2.2. Private non-publicly available datasets obtained from third parties

#### 2.2.1. Commercially licensed by rightsholders or their representatives

Were transactional commercial licensing agreements concluded for private third-party training datasets? **No.** The provider confirms that no private third-party datasets were used.

#### 2.2.2. Other private third-party datasets

Were other private third-party datasets used? **No**, as confirmed by the provider. The public sources in section 2.1 may still carry dataset-specific terms and access requirements.

### 2.3. Data crawled and scraped from online sources

The Focus Crawl described in section 2.1 was independently created by Common Crawl, as confirmed by the provider. Marin downloaded public archived HTML records and extracted text; that extraction pipeline did not crawl the original websites. The extracted corpus is withheld but reproducible. This source is reported by its public upstream collection under section 2.1; Marin's own unpublished preprocessing output is not a private dataset obtained from a third party or provider/on-behalf crawling.

Were crawlers used directly by the provider or on its behalf for training inputs? **No**, as confirmed by the provider. Marin consulted with Common Crawl about the Focus Crawl but did not operate or commission it. Crawler identifiers, collection behavior and a provider-crawl domain attachment are therefore not applicable. Independently compiled public datasets are reported in section 2.1.

### 2.4. User data

Were user interactions with the provider's AI models used for training? **No**, as confirmed by the provider.

Were user interactions with the provider's other services/products used? **No**, as confirmed by the provider.

The corpus includes public conversation-derived datasets and synthetic dialogues described in sections 2.1 and 2.5. These are distinct from user-interaction data collected through the provider's own services.

### 2.5. Synthetic data

Was synthetic data created by the provider or on its behalf for training? **Yes.** Modality: **text**. The provider confirms that its synthetic data was also made public and that the public recipe covers all synthetic data used for Mochi. No synthetic data was added outside that recipe.

Documented public Marin/contributor-produced synthetic sources in the prepared corpus include:

| Dataset or recipe | Documented generators | Generation and selection |
| --- | --- | --- |
| [Marin identity data](https://huggingface.co/datasets/marin-community/identity-data#generation) | GLM-5.2 | Generated question phrasing from programmatically composed identity facets and contexts, paired with constrained canonical answers. |
| [Marin KernelGym rollouts](https://huggingface.co/datasets/marin-community/glm-5.2-kernelgym-rollouts) | [GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) | CUDA/Triton kernel-optimization trajectories with execution/correctness feedback; the ingestion recipe removes sampler bookkeeping and filters specified truncations. |
| [Public Penfever agent-trace recipe](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/lib/marin/src/marin/datakit/download/penfever_rollouts.py) | MiniMax-M2.7; Qwen3.5-122B-A10B-FP8; GLM-5.2 | 142 public trace datasets across four generator/context/backend cohorts. The recipe lists each provider repository and revision. |
| [SWE-ZERO-12M trajectories](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories#methodology) | [mini-coder-1.7b](https://huggingface.co/ricdomolm/mini-coder-1.7b) | Software-engineering agent trajectories over public GitHub pull-request snapshots. |

These datasets' cards and the [pinned source recipes](https://github.com/marin-community/marin/tree/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/lib/marin/src/marin/datakit/download) provide generation details. Generator names above follow the documented labels; they do not assert unrecorded weight revisions. **[EDITORIAL: add available generator-model training-content summary links from the public model documentation. Synthetic-data inventory completeness is confirmed, not an outstanding provider question.]**

AI-assisted quality scoring or annotation alone does not establish that model-generated content was used as a training target. Identify generation of training content separately from preprocessing metadata.

### 2.6. Other sources of data

Were other sources used? **[CONFIRM C05: Yes/No.]** If yes, describe the sources and content, including any offline collection, commissioned human data or human-generated reinforcement-learning data: **[CONFIRM C05].**

## 3. Data processing aspects

### 3.1. Respect of reservations of rights from text and data mining

Is the provider a signatory to the GPAI Code of Practice commitments on copyright? **[CONFIRM C07: Yes/No; if Yes, chapter and effective date.]** Signing the Code is voluntary. This draft makes no claim of adherence.

For the historical training data, the provider relied on upstream dataset providers' collection and rights-reservation handling, as confirmed by the provider. Marin did not operate or commission a crawl. This describes the approach taken; it does not assert independent verification of every upstream practice or permission for every underlying work. The public source recipes identify the upstream collections and dataset-specific access requirements.

Copyright policy: [draft policy](COPYRIGHT_POLICY.md). **[CONFIRM C07/C11: adoption and permanent public link, if a public policy summary is provided.]**

### 3.2. Removal of illegal content

ML-based classifiers were used to identify and filter illegal content from the training data, as confirmed by the provider. This describes a filtering measure; it does not claim that all illegal content was detected or removed.
