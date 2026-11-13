# Evidence and references

Research snapshot: 4 November 2026. The source checkout retains the detailed local audits; raw operational records are omitted from this review zip. Later provider confirmations in REMAINING_TODOS.md supersede historical open questions in those audits.

## Training content

- [Datakit blog](https://marin.community/blog/2026/09/11/marin-data-pipeline-overview/) describes processing and provenance.
- [Public postprocessed source counts](https://openathena.ai/assets/images/blog/marin-data-pipeline-overview/topic-quality-sankey.json) match the reviewed frozen store at **23,106,103,007,622 text tokens**. Public figure SHA256: `75258b06e4fcbc66b10e55319a79ed14ce6be04d93489179631fafa8117f3b6a`.
- [Public bucket metadata](https://openathena.ai/assets/images/blog/marin-data-pipeline-overview/bucket-grid.json), [pinned mixture recipe](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.py) and [source registry](https://github.com/marin-community/marin/blob/64a9483cf1a928e6db97b83130d8a5d171a4aead/lib/marin/src/marin/datakit/sources.py) support the inventory.
- Seven original packaged datasets exceed the 3% reporting threshold after combining subsets. Their subtotal is **18,301,410,547,202 tokens**. With the denominator bounded by this subtotal and the full corpus, 3% lies between **549,042,316,416 and 693,183,090,229 tokens**. Each listed dataset exceeds the upper bound; the largest other packaged dataset is below the lower bound. See the full list in [the summary](drafts/TRAINING_CONTENT.md).
- Available-corpus volume and cumulative sampled exposure differ. Through step 378000, the reviewed training schedule presents **17,439,916,032,000 tokens**, including sampling/repetition.
- [Focus Crawl extraction recipe](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/lib/marin/src/marin/datakit/download/common_crawl_focus.py). The provider confirms independently created public WARCs and reproducible, withheld Marin extraction.

## Compute

For the retained lineage through step 378000, the reviewed context-corrected forward/backward estimate is **2.6842652638833303 × 10²⁴ FLOP**, rounded to **2.7 × 10²⁴**. An independent additional-operation calculation gives approximately **2.7140 × 10²⁴**.

| Phase | Context | Estimated phase FLOP |
| --- | --- | --- |
| 0–257553 | 4096 | 1.7793 × 10²⁴ |
| 257553–372000 | 16384 | 8.4839 × 10²³ |
| 372000–378000 | 65536 | 5.6586 × 10²² |

The calculation counts active MoE computation and phase-specific attention, excludes activation recomputation under the reviewed guidance, and does not discount BF16. The 48 checkpoint exports are snapshots of a lineage; they do not multiply its compute by 48. A reviewed training-metric cross-check matched the analytic convention. The larger **2.7998 × 10²⁴** estimate concerns planned step 390251 and must not be attributed to released step 378000.

The engineering estimate is below the **10²⁵ FLOP** presumption. Applicable cumulative regulatory scope and any designation remain separate checks. See [Article 51](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-51) and the [Commission GPAI guidelines](https://ec.europa.eu/newsroom/dae/redirection/document/118340). No inference parity or generation validation is established by this calculation or by export integrity checks.

## Primary requirements

- [Article 53](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-53): GPAI duties and the limited OSS exception.
- [Article 54](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-54): representation and the OSS exception.
- [Article 3](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-3): market-placement and making-available definitions.
- [Official training-summary template and explanatory notice](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models), [template PDF](https://ec.europa.eu/newsroom/dae/redirection/document/118480), and [FAQ](https://digital-strategy.ec.europa.eu/en/faqs/template-general-purpose-ai-model-providers-summarise-their-training-content).
- [GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai): voluntary adherence route.
- [Official OpenMDW 1.1 license](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1).

## Peer forms

- [NVIDIA Nemotron public training-content summary](https://developer.nvidia.com/downloads/docs/public-summary-of-training-content-for-nvidia-nemotron.pdf).
- [Reflection EU public training-content summary](https://reflection.ai/documents/eu-public-summary-of-training-content.pdf).

These are drafting examples, not regulatory approvals or evidence that another provider's practices establish Mochi's compliance.
