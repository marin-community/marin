# Marin Codebase Transform Outline

This document provides a comprehensive overview of the source files in the Marin data processing pipeline, focusing on inputs and outputs for each module.

---

## src/marin/classifiers

### classifiers/types.py
Defines core TypedDict types used throughout the classifier system: `LabeledExample` (text + label pairs), `Document` (id, source, text), and `Attribute` (id, source, attributes dict). These types are used as inputs and outputs across all classifier and processing modules.

### classifiers/utils.py
Provides the core dataset creation and labeling infrastructure for quality classifiers. Takes Dolma-format documents (jsonl.gz) and optional attribute files as inputs, applies user-defined labeling functions to generate training datasets with text-label pairs as outputs. The `create_dataset` function performs distributed sampling and merging via Ray, supporting label ensembling and random sampling with configurable rates. The `label_documents` function creates new attribute files by applying custom functions to documents and existing attributes. Utility functions include dataset splitting (train/val), shuffling, formatting, and reservoir sampling for fixed-size dataset creation.

### classifiers/bert/training.py
Trains BERT-based sequence classification models on TPUs using HuggingFace Transformers and Ray. Takes jsonl.gz training data files with text-label pairs as input, performs dataset merging, formatting, train/val splitting, and tokenization. Outputs trained BERT models with tokenizers and label indices to GCS/local paths. Includes support for checkpoint resumption from GCS, distributed training via XLA multiprocessing, and WandB logging. The `train_model` function orchestrates the full pipeline on Ray workers with TPU resources.

### classifiers/bert/utils.py
Provides utilities for BERT classifier training. The `format_example` function converts labeled examples to JSON format for BERT training. The `BertDataset` class is a PyTorch Dataset wrapper that loads jsonl.gz files with text-label pairs, builds label indices, and provides tokenized examples for training.

### classifiers/custom/registry.py
Registry system for versioned custom attribute functions. Functions take a Document, list of Attributes, and optional kwargs as inputs, returning attribute dictionaries as outputs. The `max_quality_score` registered function computes the maximum score across multiple input attributes. Functions are registered via decorator for use in ExecutorSteps.

### classifiers/fasttext/training.py
Trains fastText supervised classification models using Ray. Takes jsonl.gz files with text-label pairs as input, merges shards, formats examples in fastText format (`__label__<label> <text>`), splits into train/val, and outputs trained fastText models (.bin files) to specified paths. Supports configurable fastText hyperparameters and multi-threaded training.

### classifiers/fasttext/utils.py
Utilities for fastText model training. The `preprocess` function strips newlines from text for fastText compatibility. The `format_example` function converts labeled examples to fastText format (`__label__<label> <preprocessed_text>`).

### classifiers/hf/launch_ray_training.py
Ray launcher for distributed HuggingFace classifier training on TPUs. Takes `HFTrainingConfig` and `ResourceConfig` as inputs, spawns XLA multiprocessing training via Ray remote actors with TPU resources. Loads datasets, performs train/test splits, and delegates to `train_classifier` for actual training. Requires torch_xla and sets PJRT_DEVICE environment variable for TPU detection.

### classifiers/hf/monkey_patch_flash_attn.py
Monkey-patches ModernBERT's attention mechanism to use XLA flash attention kernels. Replaces the default eager attention implementation with TPU-optimized flash_attention from torch_xla. Takes query/key/value tensors and attention masks as inputs, outputs attention results using efficient kernel implementations. Applied via `apply_flash_attn_monkey_patch()` to improve TPU training performance.

### classifiers/hf/train_classifier.py
Trains HuggingFace encoder models (e.g., GTE, ModernBERT) for sequence classification tasks using the Trainer API. Takes jsonl.gz dataset files as input, loads via HuggingFace datasets library, tokenizes with configurable max_length, and trains regression or classification models. Outputs trained models and tokenizers to GCS with automatic upload. Includes custom DataCollator for padding, compute_metrics with precision/recall/F1 tracking, and support for regression tasks with label clipping. Designed for distributed training on TPUs with XLA backend.

---

## src/marin/cluster

### cluster/monitoring.py
Cluster health monitoring utilities for TPU nodes and Ray clusters. Takes GCP project IDs and zone lists as inputs, queries TPU metrics (device counts, preemptibility, health status) and Ray cluster metrics (node status, resource utilization) via gcloud and Ray CLI commands. Outputs structured metrics dictionaries with TPU/Ray statistics and human-readable health summaries. Supports optional WandB logging for monitoring dashboards. The `monitor_cluster_health` function runs continuous monitoring loops across multiple zones.

### cluster/dashboard_proxy.py
Flask-based HTTP proxy server for accessing multiple Ray cluster dashboards through a unified interface. Takes cluster information (names, IPs, port mappings) as inputs, parses Ray status output to extract resource utilization, and serves an HTML dashboard UI with cluster status cards. Proxies HTTP requests to individual Ray dashboards running on remote clusters. Outputs formatted resource usage percentages and human-readable numbers (e.g., 61.7k CPUs). The `DashboardProxy` class manages routing and the web UI.

### cluster/cleanup.py
Automated maintenance operations for TPU clusters. Takes GCP project/zone information and cleanup intervals as inputs, performs periodic cleanup of preempted TPU nodes and stale lockfiles on TPU workers. Outputs lists of deleted node names and cleanup statistics. The `cleanup_tpu_processes` function SSHes to TPU workers to kill processes and remove lockfiles. The `run_cleanup_loop` function provides cron-style scheduled cleanup, and `submit_cleanup_cron_job` submits cleanup as a long-running Ray job.

### cluster/config.py
Ray cluster configuration management and Jinja2 template rendering. Takes cluster configuration dictionaries (CONFIGS, GENERATION_CONFIGS) and template files as inputs, generates Ray cluster YAML configuration files for various regions and TPU generations (v4, v5e, v5p, v6e). Outputs `RayClusterConfig` dataclass instances with typed cluster parameters and rendered YAML files to the infra/ directory. The `update_cluster_configs` function generates all configs from templates, and `find_config_by_region` locates configs for specific regions.

### cluster/gcp.py
Google Cloud Platform integration for managing TPU nodes and compute instances. Takes GCP project/zone information and resource filters as inputs, executes gcloud commands to list, create, delete, and query TPU nodes and compute instances. Outputs JSON-formatted resource lists, IP addresses, and health status information. The `cleanup_preempted_tpus` function removes terminated nodes in parallel, `find_head_node_ip` locates Ray head nodes, and `ssh_to_tpu` provides SSH access to TPU workers. All operations use the `run_gcloud_command` wrapper for error handling.

### cluster/ray.py
Ray cluster management, SSH tunneling, and job orchestration. Takes cluster configuration files and dashboard configs as inputs, discovers active Ray clusters via gcloud, establishes SSH tunnels for remote access, and manages Ray jobs. Outputs SSH tunnel processes with port mappings, dashboard URLs via Flask proxy, and cluster utilization metrics. The `ray_dashboard` context manager provides single or multi-cluster dashboard access. Functions for listing nodes/workers/tasks/actors, submitting/stopping jobs, backing up/restoring job state, and adding manual TPU workers. All Ray commands use `run_ray_command` wrapper with timeout and error handling.

---

## src/marin/generation

### generation/dataset.py
Dataset sampling and label-based score extraction for training data preparation. Takes jsonl.gz files with label or generated_text fields as inputs, applies label-based sampling with configurable weights via `DatasetSampler` class. The `sample_file` Ray remote function samples examples based on label weights with exponential backoff for resilience. The `convert_labeled_documents_to_scores` function parses generated text through regex-based score extraction to convert labels to numeric scores. Outputs rebalanced jsonl.gz files and score distribution statistics dictionaries.

### generation/llm_generation.py
Abstract wrapper around vLLM for text generation. The `BaseLLMProvider` abstract class defines the interface for LLM providers. The `vLLMProvider` concrete implementation takes model names, engine kwargs (tensor parallelism, eager execution), and generation kwargs (temperature, max tokens) as inputs. The `generate` method takes batches of prompts as input and outputs lists of generated text strings using vLLM's LLM engine.

### generation/templates.py
Prompt template definitions for text generation tasks. Contains a single constant `STEP_BY_STEP_TEMPLATE` that instructs models to reason step-by-step and place final answers in LaTeX `\boxed{}` format. Takes template strings with `{example}` placeholders as input and outputs formatted prompt strings.

### generation/ray_utils.py
Ray scheduling utilities for distributed inference on TPUs/GPUs. The `scheduling_strategy_fn` function takes tensor parallel size, scheduling strategy ("STRICT_PACK"/"PACK"), TPU type, and head node flags as inputs, creates Ray placement groups with TPU and CPU bundles. Outputs `PlacementGroupSchedulingStrategy` objects for Ray task scheduling. The `get_ray_remote_args_scheduling_strategy_fn` returns callables that generate scheduling strategy dicts for Ray Data pipelines.

### generation/chunk_utils.py
Text chunking utilities with multiple strategies. Takes text documents (jsonl files with "text" field) and chunking configurations as inputs. The `ChunkStrategy` enum defines three approaches: CHAR (fixed-size windows), PARAGRAPH (line-split), PASSAGE (sentence-merged). The `chunk_iterator` function splits text into chunks, `chunk_text` transforms single examples into multiple chunked examples preserving metadata. The `chunk_with_config` function orchestrates distributed chunking via Ray, outputting chunked examples with updated IDs (with chunk index) and source document metadata.

### generation/inference.py
Large-scale text generation inference orchestration using Ray Data. Takes `TextGenerationInferenceConfig` with IO paths, model parameters, prompting settings, and hardware configurations as inputs. The `run_inference` Ray remote function reads input datasets, applies templating/chunking, calls vLLM inference via Ray Data, and writes outputs. Supports checkpoint-based resumption by tracking finished document IDs. Outputs inference results in configurable formats (one-to-one filename mapping or UUID-based naming). The `set_ray_data_config` function configures Ray Data execution context with ordering and error handling.

### generation/pipeline.py
Text generation pipeline integrating LLM providers with prompt templating. The `TextGeneration` class takes batches of examples with configurable prompt columns as inputs, applies templates (single broadcast or per-example), and calls LLM providers. The `vLLMTextGeneration` class extends with vLLM-specific features: token-based truncation to max_doc_tokens, optional chat template application for conversation formatting, and tokenizer integration. Takes model configurations (engine/generation kwargs), templates, and chat flags as inputs. Outputs batches augmented with `generated_text` column and optional `prompt` column.

---

## src/marin/processing

### processing/tokenize/tokenize.py
Main tokenization infrastructure using Levanter and Ray for distributed tokenization. The `HfDatasetSpec` class specifies HuggingFace datasets with optional subsets. The `TokenizeConfig` class configures tokenization for local files (jsonl/json/parquet), while `HfTokenizeConfig` handles HuggingFace datasets directly. The `tokenize` Ray remote function takes directory paths, tokenizer names, and cache configurations as inputs, discovers files with appropriate extensions, estimates optimal processor counts, and outputs tokenized Levanter cache files using the Levanter infrastructure.

### processing/tokenize/download_pretokenized.py
Downloads pre-tokenized Levanter caches from HuggingFace. The `PretokenizedCacheDownloadConfig` specifies HuggingFace repository IDs, revisions, and tokenizer names as inputs. The `download_pretokenized_cache` function creates ExecutorSteps that download caches from HF, outputting downloaded cache directories with PretokenizedCacheDownloadConfig metadata.

### processing/tokenize/data_configs.py
Creates Levanter dataset configurations from tokenized data. Takes TokenizerSteps and dataset specifications as inputs, converts them to Levanter source configs via `step_to_lm_mixture_component`. The `lm_data_config` function creates configs for single training sets with optional validation sets. The `lm_mixture_data_config` creates configs from mixtures of multiple datasets with weights, and `lm_varying_mixture_data_config` supports time-varying weights. Functions output `LMMixtureDatasetConfig` objects for Levanter training, with utilities for adding validation sets and interpolating mixture weights.

### processing/wikipedia/convert.py
Converts Wikipedia XML/HTML to Markdown format. The `clean_text` function takes HTML/XML strings as input, removes citations and normalizes line spacing. The `html2md` function uses BeautifulSoup and pandoc to convert HTML to Markdown, outputting cleaned Markdown-formatted text strings.

### processing/open_web_math/extract.py
Main HTML extraction pipeline for mathematical content from web pages. The `extract_text` function takes HTML strings and configuration dictionaries as inputs, filters unwanted HTML elements (buttons, images, dense links) via `filter_tree`, applies regex-based tag replacement, and extracts plain text with preserved math equations. Outputs plain text strings and metadata dictionaries indicating math types found (LaTeX, MathML, AsciiMath).

### processing/open_web_math/latex_processing.py
Processes and extracts LaTeX/MathML/AsciiMath from HTML documents. The `extract_math` function takes HTML trees as input, detects MathJax/KaTeX configurations via `get_math_config`, extracts delimited math using detected delimiters (inline/display), converts MathML to LaTeX via `mml_to_latex`, and wraps math with metadata tags. The `replace_math_tags_with_dollar_signs` function converts custom tags to standard $ delimiters. Outputs modified HTML trees and math extraction metadata dictionaries.

### processing/open_web_math/text_normalizer.py
Text normalization utilities for standardizing unicode and removing noise. The `normalize` function takes text strings and language codes as inputs, applies case normalization, accent stripping, number/punctuation standardization, and math normalization. The `normalize_for_dedup` function provides fast normalization specifically for deduplication. Utility functions include `replace_unicode_punct` for converting unicode punctuation to ASCII, `strip_accents` for removing accents, and `normalize_spacing_for_tok` for spacing around punctuation. Outputs normalized text strings.

### processing/open_web_math/line_processing.py
Per-line text processing for content quality control. Takes lists of text lines and configuration dictionaries as inputs. The `remove_empty_headers` function removes headers with no content, `remove_edit_buttons` strips Wikipedia edit markers, `remove_chinese_characters` filters lines with Chinese content, and `remove_boilerplate` removes common boilerplate using word frequency analysis. Outputs filtered lists of text lines.

### processing/open_web_math/tree_processing.py
HTML DOM tree manipulation and cleaning operations. Takes HTMLTree objects as inputs and performs various cleaning operations: `remove_buttons` removes button elements, `remove_dense_links` removes containers with >80% links, `flatten` removes single-child divs/spans, `extract_headings` formats heading tags, `extract_tables` formats HTML tables. Outputs modified HTMLTree objects.

### processing/open_web_math/utils.py
Utility classes and functions for OpenWebMath processing. The `ReplacementManager` class manages tagged text replacements for preserving special content during processing. The `Config` class loads YAML configurations with probabilistic sampling support. Utility functions include `has_style` for checking CSS styles and `word_wrap` for text wrapping to specified widths. Takes various text and configuration inputs, outputs managed replacements and wrapped text.

### processing/open_web_math/manual_filter.py
Applies OpenWebMath manual filtering rules and quality checks. The `manual_url_filter` function takes URL strings and document text as inputs, checks URLs against blocklists via `is_good_url`, counts LaTeX formulas via `count_latex_formulas`, and detects high ratios of accented characters via `has_accented_char`. Outputs boolean filter decisions and optionally modified text.

### processing/classification/utils.py
General utility functions for classification workflows. The `download_file` function takes file paths (GCS/local) as inputs and downloads files with format handling. The `download_huggingface_file_with_backoff` function downloads from HuggingFace with exponential backoff retry logic. The `download_gcs_file_with_backoff` downloads from Google Cloud Storage with retries. The `make_serializable` function converts Python objects to JSON-serializable format. Outputs downloaded file paths and serialized JSON objects.

### processing/classification/classifier.py
Classification inference framework using Ray actors. The `BaseClassifier` abstract class defines the interface for all classifiers. The `DummyClassifier` returns fixed scores for testing. The `FasttextClassifier` takes text documents, model names, and configurations as inputs, loads fastText models, and outputs classification scores and attributes via Ray actors.

### processing/classification/inference.py
Ray-based distributed inference pipeline for document classification. Takes dataset paths (jsonl.gz), model names, and `InferenceConfig` objects as inputs. The `convert_batch_dict_to_output_rows` function converts batch predictions to output rows. Orchestrates distributed classification using Ray Data pipelines with classifier actors, outputting classified documents with attribute annotations in jsonl.gz format.

### processing/classification/consolidate.py
Consolidates documents and applies quality filters and deduplication. Takes tokenized documents, attribute files, and filter configurations as inputs. The `FilterType` enum distinguishes quality filters from deduplication filters. Processes documents in parallel via Ray, applying attribute-based filtering and duplicate removal. Outputs filtered document sets.

### processing/classification/dataset_utils.py
Dataset I/O utilities for streaming and batch reading. The `read_dataset_streaming` function takes dataset file paths as inputs and streams rows using HuggingFace datasets library. The `write_dataset_streaming` function writes rows to dataset files. The `make_json_serializable` function converts row data to JSON-safe format. Outputs dataset iterators and written files.

### processing/classification/checkpoint_utils.py
Checkpoint management for inference state tracking. The `get_finished_ids` function takes output file paths as inputs and retrieves sets of already-processed document IDs. The `get_id_from_row` function extracts IDs from rows using path tuples. The `has_id_column` function checks if rows contain ID fields. Outputs sets of processed IDs and extracted values.

### processing/classification/autoscaler.py
Autoscaling Ray actor pools for load balancing. The `AutoscalingActorPoolConfig` class provides configuration for dynamic actor scaling with min/max actors and scaling thresholds. Takes configuration dictionaries and queue objects as inputs, outputs managed actor pools with automatic scaling based on queue depth.

### processing/classification/config/inference_config.py
Configuration dataclasses for inference workflows. The `DatasetSchemaConfig` class defines input/output column names and ID field paths. The `RuntimeConfig` class specifies Ray memory and resource configurations. The `InferenceConfig` class provides main inference configuration with model paths, input/output paths, and runtime settings. Takes configuration parameters as inputs, outputs structured configuration objects.

### processing/classification/custom/custom_attribute.py
Custom attribute computation for documents. The `CustomAttributeConfig` class configures custom attribute functions. Takes document paths, existing attribute paths, and function definitions as inputs. Applies user-defined functions to compute new attributes from document text and existing attributes via `label_documents` from classifiers/utils.py. Outputs new computed attribute files in jsonl.gz format.

### processing/classification/eval/annotations_server.py
Gradio-based web server for viewing documents and attributes. Takes input files and attribute files as inputs, provides the `sample_and_display` function for randomly sampling and displaying documents with annotations. Outputs an interactive web interface for human evaluation of classifier outputs.

### processing/classification/eval/compare_classifiers.py
Compares predictions between classifiers using Ray. The `process_file` Ray remote function takes ground truth files, prediction files, and threshold values as inputs, computes classification metrics (precision, recall, F1) by comparing predictions. Outputs classification performance metrics.

### processing/classification/fasttext/train_fasttext.py
Training script for fastText quality classifiers. The `TrainFasttextClassifierConfig` class takes dataset configurations and fastText hyperparameters as inputs. Uses the `marin.classifiers.fasttext.training.train_model` function to train models on labeled datasets. Outputs trained fastText classifier models (.bin files) to specified paths.

### processing/classification/bert/train_bert.py
Training script for BERT quality classifiers. The `TrainBertClassifierConfig` class takes dataset configurations and BERT hyperparameters as inputs. Uses the `marin.classifiers.bert.training.train_model` function to train models on labeled datasets. Outputs trained BERT classifier models with tokenizers to specified paths.

### processing/pubmed/convert.py
Converts PubMed XML articles to Markdown format. The `xml2md` function takes PubMed XML strings as inputs, parses XML structure to extract titles, abstracts, and body sections, and outputs Markdown-formatted article text with preserved structure.

---

## src/marin/tokenize

### tokenize/slice_cache.py
Cache slicing functionality for Levanter caches. The `SliceCacheConfig` dataclass takes source dataset configurations, target token counts, output paths, and tokenizer specifications as inputs. The `_do_slice_cache` function loads source caches, shuffles documents with PRNG, incrementally samples documents in batches until reaching target token count, and writes sampled documents to new caches. The `slice_cache` function creates ExecutorStep objects for execution in Ray pipelines. Outputs LmDatasetSourceConfigBase pointing to sliced caches and generates README.md files documenting the cache with factsheet information.

---

## src/marin/transform

### transform/conversation/adapters.py
Adapters for converting conversation datasets to OpenAI chat format. Takes dataset rows in various formats (single column multi-turn, instruction-response pairs, etc.) as inputs. Maintains a registry of dataset-specific adapters for datasets like OpenHermes-2.5, MetaMathQA, Tulu, UltraChat, Magpie, WildChat, and LMSYS-Chat. Each adapter function parses the source format and outputs lists of OpenAIChatMessage objects with role/content fields standardized to OpenAI format.

### transform/conversation/conversation_to_dolma.py
Converts conversation data to Dolma format. Takes jsonl.gz files with messages field (already in OpenAI format) as inputs. Concatenates messages with role labels (U for user, A for assistant) into a single text field. Outputs jsonl.gz files in Dolma format with id, source, text, and metadata fields.

### transform/conversation/preference_data_adapters.py
Adapters for preference datasets (DPO, RM training). Takes preference dataset rows with chosen/rejected columns as inputs. Converts chosen and rejected message pairs to standard format with OpenAIChatMessage lists. Outputs standardized preference examples with both chosen and rejected message sequences.

### transform/conversation/transform_conversation.py
Main transformation script for SFT conversation datasets from HuggingFace. Takes HuggingFace dataset names and adapter names as inputs. Downloads datasets from HuggingFace/GCS, applies dataset-specific adapters to convert to OpenAI format, generates hash-based IDs, includes metadata (source dataset, split, row index), and outputs sharded jsonl.gz files in Dolma conversation format with configurable sharding.

### transform/conversation/transform_preference_data.py
Transforms preference datasets from HuggingFace for DPO/RM training. Takes HuggingFace preference dataset names and adapter names as inputs. Downloads datasets, applies preference adapters to extract chosen/rejected pairs, generates IDs, and outputs sharded jsonl.gz files with preference pair metadata. Similar structure to transform_conversation.py but for preference learning data.

### transform/dolmino/filter_dolmino.py
Filters Dolmino dataset by document length. Takes Dolma-formatted jsonl.gz files as inputs. Applies minimum length filtering in chunks via Ray remote functions. Outputs filtered jsonl.gz files in Dolma format with documents meeting length requirements, supports sharded output.

### transform/dolmino/transform_dclm_hq.py
Converts DCLM HQ dump (HTML) to Dolma format. Takes HTML content from DCLM HQ dataset in ZST compression as inputs. Uses `convert_page` for HTML-to-text/markdown extraction, handles WARC records, and outputs jsonl.gz files with extracted text content in Dolma format.

### transform/evaluation/eval_to_dolma.py
Converts evaluation files to Dolma format. Takes jsonl.gz files with prompt and response fields as inputs. Concatenates prompt and response with newline separator into text field. Outputs jsonl.gz files in Dolma format for evaluation datasets.

### transform/fasttext/transform.py
Converts FastText formatted files to Dolma JSONL format. Takes FastText format files (label + text per line) as inputs. Generates unique IDs based on content hash and line number, includes timestamp metadata. Outputs jsonl.gz files in Dolma format.

### transform/fineweb/process_parquet_fw.py
Converts FineWeb parquet files to Dolma format with HTML-to-markdown conversion. Takes FineWeb parquet files and WARC files from S3 as inputs. Implements complex multi-step processing: reads parquet metadata, extracts from WARC files, converts HTML to markdown using configurable extraction methods (trafilatura, readability, resiliparse). Outputs jsonl.gz files in Dolma format with markdown, text, and optional HTML fields.

### transform/legal/transform_australianlegalcorpus.py
Converts Australian Legal Corpus to Dolma format. Takes jsonl files from raw dataset as inputs. Extracts metadata (type, jurisdiction, source, citation, URL) and text content. Outputs jsonl.gz files in Dolma format with legal document metadata.

### transform/legal/transform_hupd.py
Converts Harvard USPTO Patent Dataset to Dolma format. Takes tar.gz files containing JSON documents as inputs. Combines title, abstract, claims, and full description into text field. Includes patent metadata (application number, filing date, decision, etc.). Outputs jsonl.gz files in Dolma format.

### transform/legal/transform_multilegalpile.py
Converts MultiLegalPile dataset to Dolma format. Takes jsonl.xz compressed files as inputs. Extracts text and metadata (type, jurisdiction). Outputs jsonl.gz files in Dolma format for multi-jurisdictional legal documents.

### transform/legal/transform_edgar.py
Converts SEC EDGAR parquet files to Dolma format. Takes parquet files with multiple section columns (15 sections of 10-K forms) as inputs. Combines all sections into single text field with section headers. Includes filing metadata (CIK, accession number, filing date). Outputs jsonl.gz files in Dolma format.

### transform/lingoly/to_dolma.py
Converts Lingoly dataset to Dolma format. Takes zip file containing test.jsonl as input. Concatenates preamble, context, and questions with size limits per document (max 5 questions). Outputs jsonl files (uncompressed) in Dolma format.

### transform/medical/lavita_to_dolma.py
Converts Lavita medical QA datasets to Dolma format. Takes parquet files from HuggingFace (3 subsets: all-processed, medmcqa, pubmed-qa) as inputs. Applies format-specific transformations for each subset. Outputs parquet files in Dolma format with medical QA pairs.

### transform/simple_html_to_md/process.py
Converts HTML content in files to markdown. Takes jsonl.gz files with HTML text and fineweb_metadata fields as inputs. Uses configurable extraction methods (readability, trafilatura, resiliparse) via `convert_page` function. Includes error handling for extraction failures. Outputs jsonl.gz files with markdown content in Dolma format.

### transform/stackexchange/filter_stackexchange.py
Filters StackExchange data by vote threshold and removes duplicates. Takes Dolma-formatted jsonl.gz files as inputs. Extracts vote counts from metadata, filters questions below threshold, removes duplicate questions by ID. Outputs filtered jsonl.gz files.

### transform/stackexchange/transform_stackexchange.py
Converts StackExchange dump to markdown format. Takes jsonl.gz files in Dolma format with question/answer metadata as inputs. Creates markdown templates with questions, answers (with vote counts), and tags. Supports optional answer shuffling by seed. Outputs jsonl.gz files with markdown-formatted Q&A in Dolma format.

### transform/ar5iv/transform.py
HTML cleaning and markdown conversion utilities for ar5iv. Provides multiple HTML cleaning functions: remove authors, equations, references, tables, figures, captions, etc. Takes HTML strings as inputs. The `html2md` function converts cleaned HTML to markdown using pandoc. Outputs cleaned markdown text. Used by transform_ar5iv.py for actual processing.

### transform/ar5iv/transform_ar5iv.py
Converts ar5iv HTML dump to markdown. Takes jsonl.gz files with ar5iv HTML content as inputs. Applies HTML cleaning functions from transform.py to remove references, clean equations, handle academic paper formatting. Converts to markdown preserving paper structure. Outputs jsonl.gz files with extracted markdown text in Dolma format.

### transform/wikipedia/transform_wikipedia.py
Converts Wikipedia dump HTML to markdown. Takes ndjson files with article HTML content as inputs. Removes references, moves infobox to end, cleans equations, applies post-processing quality filters (digit/word/special character percentage thresholds). Outputs jsonl.gz files with markdown content and metadata (URL, wiki_id) in Dolma format.

### transform/common_pile/filter_by_extension.py
Filters Common Pile datasets by metadata extension. Takes jsonl(.gz) files in Dolma format as inputs. Normalizes extensions, filters against allowed extension list, optional dropping of entries with missing extensions. Outputs filtered jsonl(.gz) files containing only documents with allowed file extensions.

---

## src/marin/web

### web/convert.py
HTML to text/markdown conversion using multiple extraction methods. The `convert_page` main dispatcher function takes HTML strings, optional URLs, extraction methods ("trafilatura", "readability", "resiliparse", "legacy"), and optional config objects as inputs. Method-specific functions (`convert_page_with_trafilatura`, `convert_page_with_readability`, `convert_page_with_resiliparse`, `convert_page_legacy`) apply different extraction libraries. The `make_links_absolute` helper converts relative URLs to absolute. Outputs dictionaries containing title, content (text/markdown), processed HTML, and optional URL/date/byline fields.

### web/lookup_cc.py
Interface for querying and fetching from Common Crawl archives. The `search_cc_index` function takes URLs and index names (default "2023-40") as inputs, queries Common Crawl Index server. Outputs lists of JSON records with WARC metadata (offset, length, filename). The `fetch_page_from_cc` function takes WARC records as inputs, fetches content from Common Crawl S3 storage, auto-detects encoding with chardet. Outputs decoded HTML strings. Configured for localhost:8080 testing with production fallback to index.commoncrawl.org.

### web/rpv2.py
Utilities for RedPajama v2 dataset from together.xyz. Provides `RPV2_CRAWLS` list of 108 Common Crawl snapshots (2014-2023). The `iterate_rpv2_file` function yields quality signals and duplicates from files. The `list_rpv2_parts` function generates RedPajama v2 file references. The `gopher_rules_pass_with_rpv2` function takes quality signals dictionaries as inputs and applies Gopher quality filtering rules. The `all_urls` generator yields quality signal and duplicate file URLs for specified snapshots, shards, languages, and parts. Outputs tuples of (doc_id, quality_signals_with_duplicate_flag) and boolean pass/fail decisions with reason strings.

### web/utils.py
Utility function for extracting DOM content and converting to markdown. The `extract_content_from_dom` function takes HTML strings, keyword argument dictionaries for resiliparse_dom, and HtmlToMarkdownConfig objects as inputs. Extracts main content using custom Resiliparse fork, converts to markdown, strips null characters and whitespace. Outputs markdown strings with extracted main content. Note: uses custom fork that returns simplified DOM, not compatible with official Resiliparse package.

---

## src/marin/crawl

### crawl/common/schemas.py
Data structure definitions for crawl operations. The `HtmlExtractionConfig` dataclass configures HTML extraction from WARC files with parquet inputs, URL modifiers, metadata extraction, and WARC references. The `DolmaFormattedRecord` TypedDict defines standard record format with id, source, format, html, and metadata fields. Used as inputs/outputs throughout crawl modules.

### crawl/common/utils.py
HTML decoding utility. The `decode_html` function takes HTML bytes as input, attempts UTF-8 decoding first, then detects encoding using chardet for fallback. Outputs decoded HTML strings.

### crawl/common/convert_to_html.py
Extracts HTML from WARC files referenced in parquet. Takes parquet files with WARC paths and URL references as inputs. Downloads WARCs from S3, extracts HTML for matching URLs using warcio library. Supports custom URL modifiers and metadata extraction via config. Outputs Dolma-formatted jsonl.gz files with HTML content and metadata.

### crawl/fetch_links.py
Fetches web content from URLs. Takes parquet files with URL lists as inputs. Downloads URLs using multi-threaded HTTP fetching with configurable concurrency and timeout. Implements per-domain rate limiting (1 req/sec default) and adaptive backoff for 429/503/5xx errors (exponential backoff, increases delay on errors). Outputs parquet files with status codes, headers, and content. Also fetches and stores robots.txt files for each domain.

### crawl/get_outlinks_from_html.py
Extracts outbound links from HTML documents. Takes Dolma-format jsonl.gz files with HTML as inputs. Parses HTML with BeautifulSoup, extracts anchor tags, filters to valid HTML links, canonicalizes URLs. Identifies internal vs external links and links in main content area. Outputs jsonl.gz files with link metadata (source_url, link_target, is_internal, is_in_content, anchor_text).

### crawl/convert_responses_parquet_to_warc.py
Converts fetched HTTP responses to WARC format. Takes parquet files with HTTP responses (status, headers, content) as inputs. Converts to standard WARC-Response records with gzip compression. Handles request/response pairs with proper WARC headers and payload digests. Outputs gzip-compressed WARC files.

### crawl/deduplicate_outlinks.py
BigQuery-based exact deduplication of outlinks. Takes jsonl files with link_target fields as inputs. Creates external tables in BigQuery, deduplicates by link_target using SQL, shuffles results. Outputs deduplicated jsonl files exported to GCS in shards.

### crawl/deduplicate_outlinks_against_cc.py
Bloom filter-based deduplication against Common Crawl. Takes jsonl files with outlinks as inputs. Loads Common Crawl URL bloom filter, filters out links already in CC. Outputs jsonl files with only novel/new links not present in Common Crawl.

### crawl/minhash/index_for_deduplication.py
Creates MinHash index for fuzzy deduplication. Takes dataset paths as inputs, uses datatrove to compute MinHash signatures. Two-stage pipeline: signature generation then bucket indexing. Outputs indexed buckets for fuzzy duplicate detection.

### crawl/minhash/deduplicate_against_index.py
Fuzzy deduplication against MinHash index. Takes input dataset paths and indexed dataset paths as inputs. Four-stage pipeline: computes signatures, builds buckets, clusters duplicates, filters dataset. Outputs deduplicated jsonl files, removed ID lists, and exclusion logs.

### crawl/sample_from_nonunique_outlinks.py
Samples from outlink sets with duplicates. Takes jsonl files with outlinks as inputs. Performs rejection sampling of k items from n items with repetition, deduplicates by link_target. Groups URLs by domain in round-robin shards. Outputs parquet shards with sampled, deduplicated, domain-grouped URLs.

### crawl/sample_from_unique_outlinks.py
Samples from deduplicated outlink sets. Takes jsonl files with unique outlinks as inputs. Sequential sampling (no rejection needed), skips first `start_from` items, collects `num_to_sample` items. Groups by domain for balanced distribution. Outputs parquet files with sampled URLs.

### crawl/count_outlinks.py
Counts unique outlinks using HyperLogLog. Takes jsonl files with outlinks as inputs. Approximate counting with HyperLogLog for memory efficiency, optional exact counting with sets. Processes files in parallel via Ray. Outputs unique outlink count statistics.

### crawl/count_tokens.py
Token counting for text datasets. Takes parquet/jsonl files with text fields as inputs. Tokenizes using GPT-2 tokenizer, caches results per file. Processes files via Ray with cached_or_construct_output. Outputs aggregated token statistics.

### crawl/get_fineweb_edu_crawl_yield.py
FineWeb-Edu quality filtering pipeline. Takes WARC files as inputs. Extracts text using trafilatura, applies filters: URL filtering, language ID (fastText), gopher repetition/quality, C4 quality rules. Scores with FineWeb-Edu classifier (3.0+ threshold for passing). Outputs passing/failing parquet files with scores, metadata, and filter statistics.

### crawl/get_open_web_math_crawl_yield.py
OpenWebMath quality filtering pipeline for math content. Takes WARC files as inputs. Extracts text using resiliparse, applies math detection pre-filter (keywords, LaTeX commands, fastText classifier). Filters by language ID, perplexity (KenLM), mathscore thresholds. Manual URL filtering for known bad domains. Outputs passing/failing parquet files with math scores, perplexity, language, and detailed metadata.

### crawl/get_finemath_crawl_yield.py
FineMath quality filtering pipeline combining FineWeb-Edu scoring with math filtering. Takes WARC files as inputs. Extracts text using resiliparse, applies FineWeb-Edu-like quality filters (language, repetition, gopher, C4). Scores with FineWeb-Edu classifier. Outputs passing/failing parquet files with quality scores and filter metadata.

### crawl/fineweb_edu/convert_fineweb_edu_to_html.py
Extracts HTML from FineWeb-Edu WARC files. Takes FineWeb-Edu parquet files with WARC references as inputs. Groups by source WARC, downloads WARC files from S3, extracts HTML for matching URLs. Handles invalid S3 paths by rewriting HuggingFace disk paths to Common Crawl S3 paths. Outputs Dolma-format jsonl.gz files with HTML content.

### crawl/fineweb_edu/get_urls_and_fineweb_scores_from_warcs.py
Scores Common Crawl WARCs with FineWeb-Edu classifier. Takes sampled CC WARC paths as inputs. Downloads and extracts WARCs, uses trafilatura for text extraction, scores with HuggingFace FineWeb-Edu classifier. Outputs parquet files with URLs and quality scores (0-5 scale).

### crawl/fineweb_edu/get_urls_and_scores_from_fineweb_edu_html.py
Scores FineWeb-Edu HTML files. Takes Dolma-format jsonl.gz files with HTML as inputs. Extracts text from HTML, scores with FineWeb-Edu classifier. Outputs parquet files with URLs and quality scores.

### crawl/fineweb_edu/resample_fineweb_edu_urls_by_quality_score.py
Resamples and splits FineWeb-Edu URLs by quality scores. Takes parquet files with URLs and scores as inputs. Discretizes scores into 6 bins (0-5), splits by domain for train/test separation. Balances training set with equal samples per bin, creates CC-distributed and balanced test sets. Outputs stratified parquet files for training and evaluation.

### crawl/open_web_math/convert_open_web_math_to_html.py
Extracts HTML from OpenWebMath WARC files. Takes OpenWebMath parquet files with WARC references as inputs. Groups by source WARC, downloads and decodes WARC files from S3. Generates IDs from URL+date hash. Outputs Dolma-format jsonl.gz files with HTML content and metadata.

### crawl/open_web_math/get_urls_and_openwebmath_scores_from_warcs.py
Scores Common Crawl WARCs for math content. Takes CC WARC paths as inputs. Downloads WARCs with retry logic, extracts text, scores with FastText math classifier. Outputs parquet files with URLs and math scores.

### crawl/open_web_math/get_urls_and_scores_from_openwebmath_html.py
Scores OpenWebMath HTML files for math content. Takes Dolma-format jsonl.gz files with HTML as inputs. Extracts text from HTML, scores with FastText math classifier, includes found_math detection flag. Outputs parquet files with URLs, scores, and math detection status.

### crawl/open_web_math/consolidate_open_web_math_shards.py
Consolidates small JSONL shards into larger files. Takes directories of small jsonl.gz files as inputs. Groups 1000 shards per output file. Processes in parallel via Ray. Outputs consolidated jsonl.gz files with reduced file count.

### crawl/open_web_math/resample_openwebmath_urls_by_quality_score.py
Resamples and splits OpenWebMath URLs by quality scores. Takes parquet files with URLs and math scores as inputs. Clips scores to [0, 1], applies regex adjustment for math detection, discretizes into 10 bins (0.0-1.0 in 0.1 increments). Domain-based train/test split, balances training set, creates CC-distributed and balanced test sets. Outputs stratified parquet files for training and evaluation.

---

## Common Patterns Across Modules

All modules follow consistent architectural patterns:

1. **Ray for Distribution**: Use `@ray.remote` decorators for parallel processing
2. **Idempotency**: Use `@cached_or_construct_output` with SUCCESS files for resumability
3. **Cloud Storage**: Use fsspec for unified access to GCS, S3, and local filesystems
4. **Dolma Format**: Standardized output format with id, text, source, and metadata fields
5. **Configuration**: Dataclasses with draccus for CLI support and type safety
6. **Sharding**: Process data in shards/chunks with configurable concurrency
7. **Error Handling**: Graceful degradation with logging and retry mechanisms
