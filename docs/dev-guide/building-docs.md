# Building and Maintaining Documentation

This guide explains how to build, test, and maintain the Marin documentation.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 or higher
- uv (Python package manager)
- Git

## Installation

1. Install the documentation dependencies:
   ```bash
   uv sync --package marin --group docs
   ```

## Building Documentation

### Local Development

1. Start the local development server:
   ```bash
   uv run mkdocs serve
   ```
   This will start a local server at `http://127.0.0.1:8000` where you can preview your changes in real-time.

2. Build the documentation:
   ```bash
   uv run mkdocs build
   ```
   This will create a `site` directory containing the built documentation.

### Production Build

For production builds, use:
```bash
uv run mkdocs build --clean
```

## Documentation Structure

The documentation follows the [Diátaxis](https://diataxis.fr/) framework with four main sections:

1. **Tutorials** (`docs/tutorials/`)
   - Step-by-step guides
   - Getting started guides
   - Learning-oriented content

2. **Technical References** (`docs/references/`)
   - API documentation
   - Configuration options
   - Technical specifications

3. **Explanations** (`docs/explanations/`)
   - Background information
   - Design decisions
   - Best practices


We deviate slightly, adding design docs as a fifth section.

## Writing Documentation

### Markdown Guidelines

1. Use proper heading hierarchy:
   ```markdown
   # Main Title
   ## Section
   ### Subsection
   ```

2. Include code blocks with language specification:
   ```markdown
   ```python
   def example():
       return "Hello, World!"
   ```
   ```

3. Use admonitions for important notes:
   ```markdown
   !!! note
       This is an important note.
   ```

### API Documentation

For Python code documentation:
1. Use Google-style docstrings
2. Include type hints
3. Document parameters and return values

Example:
```python
def process_data(data: List[str]) -> Dict[str, int]:
    """Process a list of strings into a frequency dictionary.

    Args:
        data: List of strings to process

    Returns:
        Dictionary mapping strings to their frequencies
    """
    return Counter(data)
```

## Testing Documentation

1. Check for broken links:
   ```bash
   uv run mkdocs build --strict
   ```

2. Check GitHub source links after moving, deleting, or relinking docs pages:
   ```bash
   uv run python infra/check_docs_source_links.py
   ```

3. Validate markdown:
   ```bash
   uv run mkdocs build --strict --verbose
   ```

## Deployment

The documentation is automatically deployed when changes are pushed to the main branch. The deployment process:

1. Builds the documentation
2. Validates the build
3. Deploys to the configured hosting service

## Contributing

1. Create a new branch for your changes
2. Make your changes
3. Preview the docs locally with `uv run mkdocs serve`
4. Run `./infra/pre-commit.py --all-files --fix`
5. Run `uv run mkdocs build --strict`
6. Submit a pull request whose body references an issue with `Fixes #NNNN` or `Part of #NNNN`

For the full contributor workflow, including targeted test guidance, see
[Contributing to Marin](contributing.md).

## Common Issues

### Broken Links
- Use relative links for internal documentation
- Use absolute URLs for external links
- Run `uv run python infra/check_docs_source_links.py` after moving, deleting, or relinking docs pages that reference GitHub paths
- Run `uv run mkdocs build --strict` after docs edits to catch navigation and Markdown link failures

### Build Errors
- Check for syntax errors in markdown
- Verify all required dependencies are installed
- Check for missing files or incorrect paths

## Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Diátaxis Documentation Framework](https://diataxis.fr/)
