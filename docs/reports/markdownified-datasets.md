# Dataset Card: Markdownified Datasets

## Overview
This dataset collection consists of several large-scale text corpora that have been processed and tokenized through a custom fork of Resiliparse. The datasets are primarily intended for language model training and research purposes.

## Dataset Statistics

| Dataset | Token Count | Approximate Size | Source |
|---------|------------|------------------|--------|
| Wiki | 8,587,224,558 | 8.59B tokens | Wikipedia |
| Arxiv No Problem | 2,742,463,924 | 2.74B tokens | ArXiv |
| Arxiv Warning | 19,552,307,274 | 19.6B tokens | ArXiv |
| Stack Exchange | [Missing data] | [Missing data] | Stack Exchange Network |

## Processing Methodology

These datasets were processed using our custom fork of Resiliparse which simplified the raw HTML DOM, the simplified DOM was then processed by our custom implementation of Markdownify. The exact modifications and enhancements made to the original Resiliparse are documented in the next section, the processing pipeline appears to have:

1. Extracted text from various sources (Wikipedia, ArXiv, Stack Exchange)
2. Simplified the raw HTML DOM through our custom fork of Resiliparse
3. Process the simplified DOM with our custom Markdownify implementation to covert DOM to Markdown.

## Usage Notes

When using these datasets, people should be aware of:
- The potential differences in quality between the "No Problem" and "Warning" subsets of Arxiv.
- These are Markdownified version of the raw dataset and have not been filtered for quality.

# Resiliparse Custom Fork

This fork extends the original **Resiliparse** HTML-to-text extractor with a new helper, `extract_simplified_dom`, that yields a _cleaned & simplified_ HTML snippet instead of plain text.  The goal is to preserve **minimal HTML structure** (e.g. headings, paragraphs, links, lists) while still removing boiler-plate, scripts, tracking pixels, etc.

### Behaviour at a Glance

| Mode                | Original `extract_plain_text` | New `extract_simplified_dom` |
|---------------------|--------------------------------|------------------------------|
|Output               |Plain text only                |Simplified HTML               |
|Preserves `<p><h1>`… |Optional via `minimal_html`     |Always (unless filtered)      |
|Whitespace handling  |Collapses to single space/`\n`  |Follows DOM tree indentation  |
|Link handling        |`href` rendered as text         |Anchor kept as `<a>`          |
|List handling        |Bullets/indices converted to •  |`<ul>/<ol>` retained          |
|Boiler-plate removal |Yes                             |Yes                           |

### Implementation Details

The core enhancement is the implementation of `extract_simplified_dom` which leverages Lexbor's DOM serialization capabilities to maintain HTML structure while still applying the filtering and content extraction logic from the original `extract_plain_text` function.

1. **DOM Serialization**:
   - Added `serialize_node` function that converts DOM nodes to their HTML string representation
   - Utilizes Lexbor's `lxb_html_serialize_tree_str` to create a faithful representation of the DOM structure

2. **Modified Extraction Logic**:
   - Preserves important semantic HTML tags rather than stripping all markup
   - Follows the same element filtering rules as `extract_plain_text` (skipping script, style, etc.)
   - Maintains structural relationships between elements

3. **Configuration Options**:
   - Maintains the same parameter interface as `extract_plain_text` for consistency
   - Allows for the same customization of content extraction (links, alt texts, form fields, etc.)

This enhancement maintains full backward compatibility with the existing Resiliparse API while extending its capabilities for applications that need more structure than plain text extraction provides.

### Acknowledgements

Huge thanks to **Janek Bevendorff** for the original Resiliparse project and to
Lexbor contributors for the blazing-fast HTML parser/serializer. 
