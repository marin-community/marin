# Building and Maintaining Documentation

This guide explains how to build, test, and maintain the Marin documentation.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.12 or higher
- pip (Python package manager)
- Git

## Installation

1. Install the documentation dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

## Building Documentation

### Local Development

1. Start the local development server:
   ```bash
   mkdocs serve
   ```
   This will start a local server at `http://127.0.0.1:8000` where you can preview your changes in real-time.

2. Build the documentation:
   ```bash
   mkdocs build
   ```
   This will create a `site` directory containing the built documentation.

### Production Build

For production builds, use:
```bash
mkdocs build --clean
```

## Documentation Structure

The documentation follows the [Diátaxis](https://diataxis.fr/) framework with four main sections:

1. **Tutorials** (`docs/tutorials/`)
   - Step-by-step guides
   - Getting started guides
   - Learning-oriented content

2. **How-to Guides** (`docs/how-to-guides/`)
   - Task-oriented guides
   - Problem-solving guides
   - Practical instructions

3. **Technical Reference** (`docs/reference/`)
   - API documentation
   - Configuration options
   - Technical specifications

4. **Explanation** (`docs/explanation/`)
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
   mkdocs build --strict
   ```

2. Validate markdown:
   ```bash
   mkdocs build --strict --verbose
   ```

## Deployment

The documentation is automatically deployed when changes are pushed to the main branch. The deployment process:

1. Builds the documentation
2. Validates the build
3. Deploys to the configured hosting service

## Contributing

1. Create a new branch for your changes
2. Make your changes
3. Test locally using `mkdocs serve`
4. Submit a pull request

## Common Issues

### Broken Links
- Use relative links for internal documentation
- Use absolute URLs for external links
- Test links after making changes

### Build Errors
- Check for syntax errors in markdown
- Verify all required dependencies are installed
- Check for missing files or incorrect paths

## Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Diátaxis Documentation Framework](https://diataxis.fr/)
