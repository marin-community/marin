# Decontamination Guide for Dolma Dedupe (Paragraph Mode)

This document explains how to use Dolma's deduplication tool in **decontamination** mode, with **paragraph-level** span tagging and **n-gram** matching. It covers the two-phase workflow, the difference between document vs. paragraph modes, n-gram matching logic, fallback behavior, and how to interpret the JSONL attribute output.

---

## 1. Two-Phase Decontamination Workflow

When you enable decontamination (`--decontaminate`), Dolma runs in two sequential phases:

1. **Phase 1: Build the Bloom filter**
   - Input: the *decontamination dataset* (e.g., your development set).
   - Command flags include `--no-bloom_filter.read_only` so that every paragraph (or n-gram span) from these documents is **added** to the Bloom filter.
   - Result: a binary Bloom filter file (e.g. `decontaminated_bloom_filter.bin`).

2. **Phase 2: Test the target corpus**
   - Input: your *main dataset* (e.g., training or test corpus).
   - Command flags include `--bloom_filter.read_only` so that no new entries are added; the tool **only tests** each span against the prebuilt filter.
   - Output: JSONL attribute files marking all spans that match the decontamination set.

The Marin helper script automates staging files into a temporary directory, running these two steps under the hood, and then uploading the attribute files back to your storage target.

---

## 2. Document Mode vs. Paragraph Mode

Dolma dedupe supports two primary tagging modes:

- **Document-Level ("URL") Mode**
  - Flag an entire JSONL record (all its text) as duplicate vs. non-duplicate.
  - Enables you to exclude or drop whole documents.
  - Invoked via `--dedupe.documents.attribute_name <name>`.

- **Paragraph-Level Mode**
  - Split each document's `text` by newline characters (`\n`) into paragraphs.
  - Tag **individual paragraphs** as duplicates based on exact-match or n-gram logic.
  - Invoked via `--dedupe.paragraphs.attribute_name <name>`.

In decontamination pipelines, paragraph mode is typically preferred because it pinpoints exactly which sections (spans) of a large document overlap with your protected set.

---

## 3. N-Gram Matching Mode

When you enable n-gram matching under paragraph mode, Dolma will:

1. **Tokenize** each paragraph into Unicode word tokens (per UTR29).
2. **Slide** a window of length `ngram_length` (e.g. 8) across these tokens, advancing by `stride` each time.
3. **Test** each extracted n-gram against the Bloom filter.
4. **Compute** a `matched_fraction` = (# of matching n-grams) / (total n-grams in paragraph).
5. **Flag** the paragraph if `matched_fraction >= overlap_threshold` (e.g. 0.7).

### Key Flags

```bash
--dedupe.paragraphs.by_ngram.ngram_length    8
--dedupe.paragraphs.by_ngram.stride          0
--dedupe.paragraphs.by_ngram.overlap_threshold 0.7
```

- **ngram_length**: number of tokens per window.
- **stride**: step size (0 = every adjacent window).
- **overlap_threshold**: minimum fraction of n-grams that must hit the filter to mark the *whole paragraph*.

### Stride Details
- A `stride` of `0` produces an n-gram at every adjacent token offset (maximal overlap).
- A `stride` > `0` skips that many tokens between each generated n-gram window (e.g., `stride=2` moves the window by two tokens each time, reducing overlap and total n-grams).

### Skip-Short-Paragraph Behavior
The underlying Rust struct also exposes a `skip_short_paragraphs` flag:
```rust
pub struct NgramDedupeConfig {
    pub ngram_length: usize,
    pub stride: usize,
    pub overlap_threshold: f32,
    pub skip_short_paragraphs: Option<bool>, // skip paragraphs shorter than ngram_length + stride
}
```
When `skip_short_paragraphs` is enabled, any paragraph whose token count is less than `ngram_length + stride` is automatically skipped for n-gram checks (falling back to full-paragraph matching or being ignored), ensuring very short paragraphs don't generate zero n-grams.

---

## 4. Short-Paragraph Fallback

Per the merged PR, if a paragraph has _fewer tokens_ than `ngram_length`, Dolma **falls back** to an exact-match check of the entire paragraph:

- If the full paragraph string is already in the Bloom filter → emit span with `score = 1.0`.
- Otherwise, do not emit any span.

**Disclaimer:** Only **whole paragraphs** are ever flagged. If a small snippet or substring (for example, an 8-token phrase) appears verbatim inside a much larger paragraph but does _not_ constitute at least the specified `overlap_threshold` fraction of that paragraph's n-grams, the paragraph will **not** be marked as duplicate. Matched fractions are always computed over the entire paragraph length.

This ensures that you don't erroneously mark or drop sections of long paragraphs just because they contain brief overlaps.

---

## 5. Attribute JSONL Output Structure

Each input document produces one or more attribute JSONL files (one per input shard).  Each line is an `OutputSpec` record:

```json
{
  "id": "<document-id>",
  "attributes": {
    "<attribute_name>": [
      [start_char, end_char, score],
      ...
    ]
  },
  "source": "<original source path>"
}
```

- `start_char` / `end_char`: character offsets delimiting the **entire paragraph** you flagged.
- `score`:
  - For n-gram paragraphs, the `matched_fraction` (e.g. 0.78).
  - For short-paragraph fallback or exact-match mode, always `1.0`.

**Example:**
```json
{
  "id": "doc123",
  "attributes": {
    "train_test_overlap": [
      [0, 200, 0.78],      # paragraph 1: 78% of its 8-grams matched dev set
      [400, 580, 1.00]     # paragraph 3 was shorter than 8 tokens and matched exactly
    ]
  },
  "source": "gs://bucket/mmlu/test.jsonl.gz"
}
```

---

## 6. Post-Processing & Metrics

Dolma itself does **not** emit a single "X% contamination" number.  However, you can compute:

1. For each document:
   - Sum up `end_char - start_char` for all spans → total flagged chars.
   - Divide by the document's full length (in chars or tokens) → per-doc contamination ratio.

2. Aggregate across your corpus (e.g. weighted by doc length) → overall contamination percentage.

**Tip:** if you want "what fraction of the dev set appears in my train set," simply reverse the roles (build filter from train, test dev in read-only mode) and compute the same metric.

---

## 7. Example Command

```bash
dolma dedupe \
  --documents "/path/to/docs/**/*.jsonl.gz" \
  --dedupe.paragraphs.attribute_name train_test_overlap \
  --dedupe.skip_empty \
  --dedupe.min_length 0 \
  --dedupe.min_words 0 \
  --dedupe.paragraphs.by_ngram.ngram_length 8 \
  --dedupe.paragraphs.by_ngram.stride 0 \
  --dedupe.paragraphs.by_ngram.overlap_threshold 0.7 \
  --bloom_filter.file decontaminated_bloom_filter.bin \
  --bloom_filter.read_only \
  --processes 4
```

This assumes you've already built `decontaminated_bloom_filter.bin` by running dedupe over your dev set _without_ `--read_only`.

---

## 8. Further Reading
- Upstream docs: https://github.com/allenai/dolma/blob/main/docs/deduplication.md
- Dolma CLI reference: `dolma dedupe --help`

## 9. Document-Mode Decontamination

By default, the Marin wrapper runs Dolma dedupe in **paragraph mode** (via `--dedupe.paragraphs.attribute_name`) so that individual paragraphs (or n-grams) are tested in the two-phase decontamination workflow. If you switch to **document mode**:

```bash
--dedupe.documents.attribute_name <attribute_name>
```

Dolma will treat each JSONL record as a single key (typically extracted via a JSONPath on a document field) and perform decontamination on the full document string. In this mode:

- **No n-gram matching**: n-gram flags (`--dedupe.paragraphs.by_ngram.*`) do _not_ apply—each document is hashed once and only exact-string membership is tested.
- **Output**: one span per document is emitted when a match is found, e.g. `[0, full_text_length, 1.0]` to mark the entire record.

### Decontamination Phases in Document Mode

1. **Phase 1 – Build filter**: with `--no-bloom_filter.read_only`, every document key from the decontamination set is inserted into the Bloom filter.
2. **Phase 2 – Test target**: with `--bloom_filter.read_only`, each target document key is tested; if present, the whole document is flagged.

This mode is ideal when you want to drop or tag entire documents that overlap with your protected set, rather than just paragraphs.

## 10. Train/Test Overlap Considerations

When you run the decontaminate pipeline in **paragraph mode** to measure overlap between a development (or training) shard and a held-out test set, here's what happens:

1. **Phase 1** (build filter):
   - The script stages each paragraph (or its n-grams) from the dev shard, and with `--no-bloom_filter.read_only` each key is **inserted** into the Bloom filter.

2. **Phase 2** (test target):
   - The test documents are staged, and with `--bloom_filter.read_only` each paragraph is split into n-grams (or fallback to the full string if the paragraph is too short).
   - Dolma tests each n-gram key; it counts how many n-grams hit the filter, computes `matched_ngrams / total_ngrams`, and if that fraction ≥ `--dedupe.paragraphs.by_ngram.overlap_threshold`, the paragraph is flagged.

**Small-paragraphs caveat**

- Very short test questions produce few or even just one n-gram window, triggering the _fallback_ to a full-string membership test.
- A single hash test can more easily collide (false positive), especially if the text is very short or highly repetitive.

You can mitigate this by:
- Raising the n-gram `overlap_threshold` or increasing `stride` so you require stronger evidence.
- Turning on `--dedupe.paragraphs.by_ngram.skip_short_paragraphs` to skip fragments shorter than `ngram_length + stride` entirely.
- Increasing `--dedupe.min_length` or `--dedupe.min_words` so tiny snippets are excluded from the test phase.

### Document Mode Alternative

If instead you switch to **document mode** with
```bash
--dedupe.documents.attribute_name <attr>
```

- **No n-grams** are generated. Dolma treats each entire JSONL record (e.g. the full `text` field) as a single key.
- In Phase 1, you would insert each full document string into the filter; in Phase 2, you test each test record's full string under read-only mode.

**Substring handling**

- A test sample that is only a substring of a larger dev document will **not** match—because it does not equal the complete string that was inserted.
- Document mode thus eliminates the risk of short-text false positives, but it also cannot detect partial overlap; only exact, entire-document duplicates will be flagged.

Use paragraph-mode (with tuned threshold, stride, or short-paragraph skipping) when you care about partial/textual overlap. Use document-mode when you only want to drop or tag entire records that exactly reappear in your protected set.
